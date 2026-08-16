"""Telemetry event stream and the graceful-stop path.

The stream is a contract with consumers that do not import this package (the
monitor, the web view, analysis scripts), so these tests read the file back as
plain JSON rather than through the emitter's own types.
"""

import json

import pytest

from balatro_train.config import (EnvConfig, LogConfig, PolicyConfig, PPOConfig,
                                  TrainConfig)
from balatro_train.ppo import PPOTrainer
from balatro_train.telemetry import (ALL_TYPES, SCHEMA_VERSION, EventType,
                                     RunSession, TelemetryEmitter,
                                     close_emitter, emit, set_emitter)


def _cfg(tmp_path, total=2 * 16 * 64, run_name="test"):
    return TrainConfig(
        env=EnvConfig(name="mock", num_envs=16, seed=0,
                      kwargs={"reward_mode": "bandit"}),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=total, rollout_len=64, minibatch_size=512,
                      update_epochs=1, lr=1e-3),
        log=LogConfig(run_name=run_name, log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "runs"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )


def read_events(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_type(events, event_type):
    return [e for e in events if e["type"] == event_type]


@pytest.fixture(autouse=True)
def _no_leaked_emitter():
    """Every test leaves the process-wide emitter as it found it."""
    yield
    close_emitter()


# ------------------------------------------------------------------- emitter


def test_emitter_writes_conformant_ndjson(tmp_path):
    path = str(tmp_path / "events.jsonl")
    emitter = TelemetryEmitter(path)
    emitter.emit(EventType.ROLLOUT, sps=1234, loss=0.5)
    emitter.emit(EventType.EPISODE_END, r=1.0, won=True)
    emitter.close()

    events = read_events(path)
    assert [e["type"] for e in events] == [EventType.ROLLOUT, EventType.EPISODE_END]
    assert [e["seq"] for e in events] == [1, 2]
    for event in events:
        assert event["v"] == SCHEMA_VERSION
        assert event["type"] in ALL_TYPES
        assert isinstance(event["t"], float)
    assert events[0]["data"] == {"sps": 1234, "loss": 0.5}


def test_emit_is_a_noop_without_an_emitter(tmp_path):
    """Nothing installed means no crash and no file -- telemetry is optional."""
    close_emitter()
    emit(EventType.ROLLOUT, sps=1)


def test_set_emitter_closes_the_previous_one(tmp_path):
    first = TelemetryEmitter(str(tmp_path / "a.jsonl"))
    first.emit(EventType.ROLLOUT, sps=1)
    set_emitter(first)
    set_emitter(TelemetryEmitter(str(tmp_path / "b.jsonl")))

    # The replaced emitter was drained and closed, not abandoned mid-queue.
    assert first._handle is None
    assert len(read_events(str(tmp_path / "a.jsonl"))) == 1


# ------------------------------------------------------------------- session


def test_for_run_reuses_the_directory_and_keeps_created(tmp_path):
    """Resuming keeps one identity: same dir, same creation stamp."""
    runs = str(tmp_path / "runs")
    first = RunSession.for_run("m3", runs, config={"a": 1})
    created = first.read_meta()["created"]

    second = RunSession.for_run("m3", runs, config={"a": 2})
    meta = second.read_meta()
    assert second.run_dir == first.run_dir
    assert meta["created"] == created
    assert meta["resumed"] is not None
    assert meta["config"] == {"a": 2}


def test_for_run_clears_a_stale_stop_request(tmp_path):
    """A leftover STOP file must not halt the next run the moment it starts."""
    runs = str(tmp_path / "runs")
    session = RunSession.for_run("m3", runs)
    session.request_stop("earlier run")
    assert session.stop_requested()

    assert not RunSession.for_run("m3", runs).stop_requested()


# ------------------------------------------------------------------- trainer


def test_trainer_writes_the_expected_event_stream(tmp_path):
    trainer = PPOTrainer(_cfg(tmp_path))
    trainer.train()

    events = read_events(trainer.session.events_path)
    types = {e["type"] for e in events}
    assert {EventType.SESSION_START, EventType.ROLLOUT, EventType.EPISODE_END,
            EventType.CHECKPOINT_SAVED, EventType.SESSION_END} <= types

    rollouts = by_type(events, EventType.ROLLOUT)
    assert len(rollouts) == 2
    # The monitor's panels read these fields; they come straight from the
    # stats dict the trainer already builds for TensorBoard.
    for key in ("iteration", "global_step", "sps", "loss", "entropy",
                "approx_kl", "ep_return_mean", "ep_win_rate", "shaping_beta"):
        assert key in rollouts[-1]["data"], key

    ends = by_type(events, EventType.EPISODE_END)
    assert ends, "no episodes finished"
    assert {"step", "r", "l", "ante", "won", "from_snapshot"} <= set(ends[0]["data"])

    session_end = by_type(events, EventType.SESSION_END)[0]["data"]
    assert session_end["status"] == "completed"


def test_stop_file_exits_cleanly_with_a_final_checkpoint(tmp_path):
    """A stop is not a kill: the run still lands its final checkpoint."""
    trainer = PPOTrainer(_cfg(tmp_path, total=10**9))
    trainer.session.request_stop("test")
    trainer.train()

    assert trainer.global_step == 0, "stop should be seen before any collection"

    events = read_events(trainer.session.events_path)
    session_end = by_type(events, EventType.SESSION_END)[0]["data"]
    assert session_end["status"] == "stopped"

    final = by_type(events, EventType.CHECKPOINT_SAVED)[-1]["data"]["path"]
    assert final == session_end["final_checkpoint"]
    assert (tmp_path / "runs" / "test" / "ckpt_0.pt").exists()

    meta = trainer.session.read_meta()
    assert meta["active"] is False
    assert meta["status"] == "stopped"


def test_telemetry_can_be_switched_off(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.log.telemetry = False
    trainer = PPOTrainer(cfg)
    trainer.train()

    import os
    assert not os.path.exists(trainer.session.events_path)
