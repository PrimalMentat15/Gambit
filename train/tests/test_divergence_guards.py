"""Update guards: target_kl early-stops an update, and a divergent update aborts
training without writing a checkpoint."""

import json
import math

import pytest

from balatro_train.config import (EnvConfig, LogConfig, PolicyConfig, PPOConfig,
                                  TrainConfig)
from balatro_train.ppo import PPOTrainer, TrainingDiverged, divergence_reason

# 16 envs x 64 steps = 1024 samples / minibatch 512 = 2 minibatches, x 4 epochs.
FULL_STEPS = 8

HEALTHY = {"loss": -0.0116, "pg_loss": -0.0011, "v_loss": 0.07, "entropy": 2.58,
           "approx_kl": 0.0118, "approx_kl_max": 0.0176}


def _cfg(tmp_path, total=16 * 64, **ppo):
    return TrainConfig(
        env=EnvConfig(name="mock", num_envs=16, seed=0, kwargs={"reward_mode": "bandit"}),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=total, rollout_len=64, minibatch_size=512,
                      update_epochs=4, lr=1e-3, anneal_lr=False, ent_coef=0.003, **ppo),
        log=LogConfig(run_name="test", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=1, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )


def _one_update(cfg):
    trainer = PPOTrainer(cfg)
    trainer.collect()
    return trainer.update()


# ------------------------------------------------------------ divergence_reason

def test_healthy_stats_are_not_divergent():
    assert divergence_reason(HEALTHY, kl_ceiling=0.1) is None
    assert divergence_reason(HEALTHY, kl_ceiling=None) is None


@pytest.mark.parametrize("key", ["loss", "pg_loss", "v_loss", "entropy",
                                 "approx_kl", "approx_kl_max"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_is_divergent_even_without_a_ceiling(key, bad):
    reason = divergence_reason({**HEALTHY, key: bad}, kl_ceiling=None)
    assert reason and key in reason


def test_negative_entropy_is_divergent_but_rounding_is_not():
    assert divergence_reason({**HEALTHY, "entropy": -10.7}, kl_ceiling=None)
    assert divergence_reason({**HEALTHY, "entropy": -1e-6}, kl_ceiling=None) is None


def test_kl_ceiling_ignores_a_lone_minibatch_outlier():
    # m12 session 2, iter 4231: one minibatch's KL estimate spiked to 0.116
    # (heavy-tailed estimator, healthy policy, kl0 ~0) while the iteration's
    # mean was 0.0068. A max-based ceiling aborted a run that was at its best.
    outlier = {**HEALTHY, "approx_kl": 0.0068, "approx_kl_max": 0.1164}
    assert divergence_reason(outlier, kl_ceiling=0.1) is None


def test_kl_ceiling_catches_a_runaway_mean():
    # The compiled m12 collapse: mean KL 3.49 the iteration after it took off.
    runaway = {**HEALTHY, "approx_kl": 3.49, "approx_kl_max": 7.2}
    assert divergence_reason(runaway, kl_ceiling=0.1)
    assert divergence_reason(runaway, kl_ceiling=None) is None


# --------------------------------------------------------------------- target_kl

def test_update_runs_every_minibatch_when_target_kl_is_off(tmp_path):
    out = _one_update(_cfg(tmp_path))
    assert out["update_steps"] == FULL_STEPS
    assert out["kl_early_stop"] == 0.0
    assert out["approx_kl_max"] >= out["approx_kl"]


def test_pre_step_kl_measures_rollout_update_agreement(tmp_path):
    # Before the first step the update recomputes the rollout's own logprobs
    # with the same weights, so eager fp32 agrees to rounding error. Training
    # then moves the policy, so the update's mean/max KL are far larger.
    out = _one_update(_cfg(tmp_path))
    assert abs(out["approx_kl_pre"]) < 1e-6
    assert out["approx_kl_max"] > 100 * abs(out["approx_kl_pre"])


def test_target_kl_stops_the_update_early(tmp_path):
    # Before the first step the weights ARE the rollout policy (KL ~ 0), so at
    # least one step always runs; lr 1e-3 then moves it far past 1e-6.
    out = _one_update(_cfg(tmp_path, target_kl=1e-6))
    assert out["kl_early_stop"] == 1.0
    assert 1 <= out["update_steps"] < FULL_STEPS
    assert out["approx_kl_max"] > 1e-6
    for key in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac"):
        assert math.isfinite(out[key]), key


def test_loose_target_kl_never_fires(tmp_path):
    out = _one_update(_cfg(tmp_path, target_kl=10.0))
    assert out["update_steps"] == FULL_STEPS
    assert out["kl_early_stop"] == 0.0


# ------------------------------------------------------------------ abort path

def test_divergence_aborts_without_writing_any_checkpoint(tmp_path):
    # save_every_iters=1 would save after this very iteration, and a normal
    # exit saves again after the loop: neither may happen for a broken update.
    trainer = PPOTrainer(_cfg(tmp_path, total=4 * 16 * 64, kl_ceiling=1e-12))
    with pytest.raises(TrainingDiverged, match="kl_ceiling"):
        trainer.train()
    assert trainer.iteration == 1
    assert not list((tmp_path / "ckpt").rglob("*.pt"))

    run_dir = tmp_path / "runs" / "test"
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "diverged"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    end = [e for e in events if e["type"] == "session_end"]
    assert end and end[-1]["data"]["status"] == "diverged"
    # The divergent iteration's stats are still emitted, so the blowup is visible.
    assert any(e["type"] == "rollout" for e in events)


def test_divergence_message_names_the_resumed_checkpoint(tmp_path):
    seed = PPOTrainer(_cfg(tmp_path / "seed")).save_checkpoint(tmp_path / "seed.pt")
    trainer = PPOTrainer(_cfg(tmp_path / "run", total=4 * 16 * 64, kl_ceiling=1e-12))
    trainer.load_checkpoint(seed)
    with pytest.raises(TrainingDiverged) as exc:
        trainer.train()
    assert str(seed) in str(exc.value)


def test_kl_stop_streak_aborts_after_consecutive_early_stops(tmp_path):
    # target_kl 1e-6 early-stops every iteration; a streak of 2 must abort on
    # the 2nd, after iteration 1's checkpoint and without saving iteration 2.
    cfg = _cfg(tmp_path, total=4 * 16 * 64, target_kl=1e-6, kl_stop_streak=2)
    trainer = PPOTrainer(cfg)
    with pytest.raises(TrainingDiverged, match="2 iterations in a row"):
        trainer.train()
    assert trainer.iteration == 2
    saved = sorted(p.name for p in (tmp_path / "ckpt").rglob("*.pt"))
    assert saved == [f"ckpt_{16 * 64}.pt"]


def test_isolated_early_stops_do_not_abort(tmp_path):
    # Early stops that never run back-to-back (the healthy pattern: 5 isolated
    # stops in ~2800 m12 iterations) must not trip a streak of 2.
    trainer = PPOTrainer(_cfg(tmp_path, total=4 * 16 * 64, kl_stop_streak=2))
    real_update = trainer.update

    def alternating():
        out = real_update()
        out["kl_early_stop"] = float(trainer.iteration % 2)
        return out

    trainer.update = alternating
    trainer.train()
    assert trainer.iteration == 4
