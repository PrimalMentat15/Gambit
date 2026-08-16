"""Snapshot start-state mixing (M3): pool file format, env mixing semantics
(mock parity + the real Rust sim), trainer anneal/logging plumbing, and the
resume-from-M1-checkpoint smoke test."""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from balatro_train import snapshot_pool
from balatro_train.config import (CurriculumConfig, EnvConfig, LogConfig,
                                  PolicyConfig, PPOConfig, SnapshotAnnealConfig,
                                  SnapshotConfig, TrainConfig, load_config)
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.ppo import PPOTrainer, snapshot_fraction_at

from conftest import random_legal_actions

# ------------------------------------------------------------ pool file


def _fake_pool(path, antes):
    """A pool file whose payloads are opaque (mock envs never decode them)."""
    snapshot_pool.write_pool(path, [(a, bytes([a]) * 8) for a in antes])
    return path


def test_pool_file_roundtrip(tmp_path):
    p = tmp_path / "pool.bin"
    entries = [(3, b"abc"), (4, b"defg"), (3, b"\x00" * 5), (7, b"x")]
    snapshot_pool.write_pool(p, entries)
    assert snapshot_pool.read_pool_antes(p) == [3, 4, 3, 7]
    assert snapshot_pool.read_pool(p) == entries

    with pytest.raises(ValueError):
        snapshot_pool.write_pool(tmp_path / "e.bin", [])
    with pytest.raises(ValueError):
        snapshot_pool.write_pool(tmp_path / "b.bin", [(0, b"x")])
    with pytest.raises(ValueError):
        snapshot_pool.write_pool(tmp_path / "b2.bin", [(3, b"")])
    (tmp_path / "junk.bin").write_bytes(b"NOTAPOOLxxxx")
    with pytest.raises(ValueError):
        snapshot_pool.read_pool_antes(tmp_path / "junk.bin")

    m = snapshot_pool.write_manifest(p, {"total": 4})
    assert m.exists() and m.name == "pool.bin.manifest.json"


def test_snapshot_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "snapshots:\n"
        "  enabled: true\n"
        "  pool_path: pools/x.bin\n"
        "  fraction: 0.25\n"
        "  anneal:\n"
        "    start_step: 100\n"
        "    end_step: 200\n"
        "    final_fraction: 0.05\n"
    )
    cfg = load_config(p)
    s = cfg.snapshots
    assert s.enabled and s.pool_path == "pools/x.bin" and s.fraction == 0.25
    assert s.anneal.start_step == 100 and s.anneal.end_step == 200
    assert s.anneal.final_fraction == 0.05
    assert SnapshotConfig().enabled is False  # off by default

    # The committed m3 config parses and has mixing enabled.
    m3 = load_config(Path(__file__).parent.parent / "configs" / "m3.yaml")
    assert m3.snapshots.enabled and m3.snapshots.fraction == 0.3
    assert m3.snapshots.anneal.final_fraction == 0.0
    assert m3.snapshots.anneal.end_step < m3.ppo.total_timesteps


def test_anneal_math():
    cfg = SnapshotConfig(enabled=True, fraction=0.3,
                         anneal=SnapshotAnnealConfig(start_step=100, end_step=200,
                                                     final_fraction=0.1))
    assert snapshot_fraction_at(0, cfg) == 0.3
    assert snapshot_fraction_at(100, cfg) == 0.3
    assert snapshot_fraction_at(150, cfg) == pytest.approx(0.2)
    assert snapshot_fraction_at(200, cfg) == pytest.approx(0.1)
    assert snapshot_fraction_at(10**9, cfg) == pytest.approx(0.1)
    # end <= start: constant fraction.
    flat = SnapshotConfig(enabled=True, fraction=0.4)
    assert snapshot_fraction_at(10**9, flat) == 0.4
    # disabled: always 0.
    assert snapshot_fraction_at(0, SnapshotConfig(enabled=False, fraction=0.4)) == 0.0


# ------------------------------------------------------- mock env parity


def _drain_episodes(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    obs, masks = env.reset(list(range(env.num_envs)))
    episodes = []
    for _ in range(steps):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, infos = env.step(actions)
        episodes += [i["episode"] for i in infos if "episode" in i]
    return episodes, obs


def test_mock_fraction_0_and_1_and_win_ante(tmp_path):
    pool = _fake_pool(tmp_path / "p.bin", [3, 3, 4, 5])

    env = MockBalatroEnv(8)
    with pytest.raises(ValueError):
        env.set_snapshot_fraction(0.5)  # no pool yet
    assert env.load_snapshot_pool(pool) == 4
    with pytest.raises(ValueError):
        env.set_snapshot_fraction(1.5)

    # fraction 1: every auto-reset episode is a snapshot start.
    env.set_snapshot_fraction(1.0)
    eps, _ = _drain_episodes(env, 500)
    assert eps, "no episodes finished"
    assert all("from_snapshot" in e for e in eps)
    # First episode per env is a fresh reset() start; later ones are mixed.
    assert any(e["from_snapshot"] for e in eps)
    # Snapshot episodes start mid-run (ante >= 3), so they end at ante >= 3.
    assert all(e["ante"] >= 3 for e in eps if e["from_snapshot"])

    # fraction 0: nothing comes from the pool.
    env.set_snapshot_fraction(0.0)
    eps, _ = _drain_episodes(env, 500)
    assert eps and not any(e["from_snapshot"] for e in eps)

    # win_ante 3: no pool entry has ante < 3 -> fresh fallback everywhere.
    env2 = MockBalatroEnv(8, win_ante=3)
    env2.load_snapshot_pool(pool)
    env2.set_snapshot_fraction(1.0)
    eps, _ = _drain_episodes(env2, 500)
    assert eps and not any(e["from_snapshot"] for e in eps)

    # win_ante 4: only the ante-3 entries qualify.
    env3 = MockBalatroEnv(8, win_ante=4)
    env3.load_snapshot_pool(pool)
    env3.set_snapshot_fraction(1.0)
    eps, _ = _drain_episodes(env3, 800)
    snap_eps = [e for e in eps if e["from_snapshot"]]
    assert snap_eps
    assert all(3 <= e["ante"] <= 4 for e in snap_eps)


def test_mock_fraction_0_bit_identical_to_no_pool(tmp_path):
    """Loading a pool at fraction 0 must not perturb the episode streams."""
    pool = _fake_pool(tmp_path / "p.bin", [3, 4])

    def run(with_pool):
        env = MockBalatroEnv(4)
        if with_pool:
            env.load_snapshot_pool(pool)
            env.set_snapshot_fraction(0.0)
        rng = np.random.default_rng(11)
        obs, masks = env.reset([5, 6, 7, 8])
        out = []
        for _ in range(300):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, r, d, _ = env.step(actions)
            out.append((r.copy(), d.copy(), obs["global"].copy()))
        return out

    for (r1, d1, g1), (r2, d2, g2) in zip(run(False), run(True)):
        assert (r1 == r2).all() and (d1 == d2).all() and (g1 == g2).all()


# ------------------------------------------------------------ trainer side


def _trainer_cfg(tmp_path, snapshots, total=4 * 8 * 8, curriculum=None):
    return TrainConfig(
        env=EnvConfig(name="mock", num_envs=8, seed=0,
                      kwargs={"reward_mode": "bandit", "bandit_episode_len": 8}),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=total, rollout_len=8, minibatch_size=64,
                      update_epochs=1, lr=1e-3),
        curriculum=curriculum or CurriculumConfig(),
        snapshots=snapshots,
        log=LogConfig(run_name="snaptest", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )


@pytest.mark.timeout(120)
def test_trainer_applies_fraction_and_logs_sources(tmp_path):
    pool = _fake_pool(tmp_path / "p.bin", [3, 3, 4])
    snaps = SnapshotConfig(
        enabled=True, pool_path=str(pool), fraction=1.0,
        # Anneal 1.0 -> 0.0 across the run's 4 iterations (64 steps/iter).
        anneal=SnapshotAnnealConfig(start_step=0, end_step=4 * 64,
                                    final_fraction=0.0),
    )
    trainer = PPOTrainer(_trainer_cfg(tmp_path, snaps))
    # Pool got loaded into the env at init.
    assert trainer.env._pool_antes is not None
    trainer.train()

    fracs = [h["snapshot_fraction"] for h in trainer.history]
    # _apply_snapshot_fraction runs BEFORE collect: iteration k covers
    # global steps (k-1)*64..k*64, logged with the fraction at (k-1)*64.
    assert fracs == pytest.approx([1.0, 0.75, 0.5, 0.25])
    # Bandit episodes are 8 env-steps here: from-snapshot episodes finish
    # from iteration 2 on and the per-source stats split them out.
    assert any(h.get("ep_from_snapshot_share", 0) > 0 for h in trainer.history)
    assert any("ep_return_snapshot" in h for h in trainer.history)
    assert any("ep_return_fresh" in h for h in trainer.history)
    for h in trainer.history:
        assert np.isfinite(h["loss"])


@pytest.mark.timeout(120)
def test_trainer_disabled_snapshots_add_no_stats(tmp_path):
    trainer = PPOTrainer(_trainer_cfg(tmp_path, SnapshotConfig(), total=64))
    trainer.train()
    assert all("snapshot_fraction" not in h for h in trainer.history)


def test_trainer_enabled_requires_pool_path(tmp_path):
    with pytest.raises(ValueError):
        PPOTrainer(_trainer_cfg(tmp_path, SnapshotConfig(enabled=True)))


@pytest.mark.timeout(120)
def test_eval_envs_never_get_the_pool(tmp_path, monkeypatch):
    """Promotion evals run on a separate env with fraction 0 (fresh starts):
    the eval env constructed inside _run_eval must not be the training env
    and must have no pool loaded."""
    pool = _fake_pool(tmp_path / "p.bin", [3])
    snaps = SnapshotConfig(enabled=True, pool_path=str(pool), fraction=1.0)
    cur = CurriculumConfig(enabled=True, start_ante=1, eval_every_steps=64,
                           promotion_eval_episodes=2, eval_num_envs=2,
                           milestone_every_steps=10**12)
    trainer = PPOTrainer(_trainer_cfg(tmp_path, snaps, total=64, curriculum=cur))

    seen_envs = []
    import balatro_train.eval as eval_mod

    real_make = eval_mod.make_vec_env

    def spy_make(name, num_envs, **kwargs):
        env = real_make(name, num_envs, **kwargs)
        seen_envs.append(env)
        return env

    monkeypatch.setattr(eval_mod, "make_vec_env", spy_make)
    trainer.train()
    assert seen_envs, "promotion eval never ran"
    for env in seen_envs:
        assert env is not trainer.env
        assert env._pool_antes is None and env._snapshot_fraction == 0.0


# ------------------------------------------------------------ real sim env

sim_available = True
try:
    import balatro_sim  # noqa: F401
except ImportError:
    sim_available = False

needs_sim = pytest.mark.skipif(not sim_available, reason="balatro_sim not built")


def _harvest_sim_pool(path, num_envs=8, want=6, ante=2, max_steps=4000):
    """Bot-drive the sim and snapshot envs entering SHOP/BLIND_SELECT at
    `ante`; write a real pool file."""
    from balatro_train.sim_env import SimBalatroEnv

    env = SimBalatroEnv(num_envs)
    env.reset(list(range(100, 100 + num_envs)))
    prev = [env.run_info(i)["state"] for i in range(num_envs)]
    entries, seen = [], set()
    for _ in range(max_steps):
        env.step(env.bot_actions())
        for i in range(num_envs):
            info = env.run_info(i)
            st = info["state"]
            if (st != prev[i] and st in ("SHOP", "BLIND_SELECT")
                    and info["ante"] == ante):
                key = (info["seed"], info["ante"], info["round"], st)
                if key not in seen:
                    seen.add(key)
                    entries.append((ante, env.snapshot(i)))
            prev[i] = st
        if len(entries) >= want:
            break
    assert len(entries) >= want, f"harvested only {len(entries)}/{want}"
    snapshot_pool.write_pool(path, entries[:want])
    return path


@needs_sim
@pytest.mark.timeout(300)
def test_sim_pool_load_and_bad_pool_rejected(tmp_path):
    from balatro_train.sim_env import SimBalatroEnv

    pool = _harvest_sim_pool(tmp_path / "p.bin", want=4)
    env = SimBalatroEnv(4)
    with pytest.raises(ValueError):
        env.set_snapshot_fraction(0.3)  # no pool yet
    assert env.load_snapshot_pool(str(pool)) == 4
    assert env.snapshot_pool_len == 4
    env.set_snapshot_fraction(0.3)
    assert env.snapshot_fraction == 0.3
    with pytest.raises(ValueError):
        env.set_snapshot_fraction(-0.1)

    # The Rust loader validates payloads: a fake pool must be rejected.
    fake = _fake_pool(tmp_path / "fake.bin", [3, 4])
    with pytest.raises(ValueError):
        SimBalatroEnv(4).load_snapshot_pool(str(fake))


@needs_sim
@pytest.mark.timeout(300)
def test_sim_mixing_determinism_and_fractions(tmp_path):
    from balatro_train.sim_env import SimBalatroEnv

    pool = _harvest_sim_pool(tmp_path / "p.bin", want=6, ante=2)

    def run(fraction, steps=1200, load_pool=True):
        rng = np.random.default_rng(17)
        env = SimBalatroEnv(8)
        if load_pool:
            env.load_snapshot_pool(str(pool))
            env.set_snapshot_fraction(fraction)
        obs, masks = env.reset(list(range(50, 58)))
        trace, eps = [], []
        for _ in range(steps):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, r, d, infos = env.step(actions)
            trace.append((r.copy(), d.copy(), obs["global"].copy()))
            eps += [i["episode"] for i in infos if "episode" in i]
        return trace, eps

    # fraction 1: from-snapshot episodes appear and are flagged.
    t1, eps1 = run(1.0)
    assert eps1, "no episodes finished"
    assert all("from_snapshot" in e for e in eps1)
    assert any(e["from_snapshot"] for e in eps1)

    # Determinism: same seeds + same pool + same fraction -> identical.
    t2, eps2 = run(1.0)
    assert eps1 == eps2
    for (r1, d1, g1), (r2, d2, g2) in zip(t1, t2):
        assert (r1 == r2).all() and (d1 == d2).all() and (g1 == g2).all()

    # fraction 0 == no pool at all, bit-identical (and never from_snapshot).
    t0, eps0 = run(0.0)
    tn, _ = run(0.0, load_pool=False)
    assert not any(e["from_snapshot"] for e in eps0)
    for (r1, d1, g1), (r2, d2, g2) in zip(t0, tn):
        assert (r1 == r2).all() and (d1 == d2).all() and (g1 == g2).all()


@needs_sim
@pytest.mark.timeout(300)
def test_sim_win_ante_gates_pool_eligibility(tmp_path):
    from balatro_train.sim_env import SimBalatroEnv

    pool = _harvest_sim_pool(tmp_path / "p.bin", want=6, ante=2)

    def episodes(win_ante, steps=2500):
        rng = np.random.default_rng(23)
        env = SimBalatroEnv(8, win_ante=win_ante)
        env.load_snapshot_pool(str(pool))
        env.set_snapshot_fraction(1.0)
        obs, masks = env.reset(list(range(8)))
        eps = []
        for _ in range(steps):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, _r, _d, infos = env.step(actions)
            eps += [i["episode"] for i in infos if "episode" in i]
        return eps

    # win_ante 1: ante-2 snapshots are past the goal -> all fresh.
    eps = episodes(1)
    assert eps and not any(e["from_snapshot"] for e in eps)
    # win_ante 3: ante-2 snapshots are eligible and stay under the goal.
    eps = episodes(3)
    snap = [e for e in eps if e["from_snapshot"]]
    assert snap
    for e in eps:
        assert e["ante"] <= 4
        assert e["won"] == (e["ante"] == 4)


# ----------------------------------------- resume-from-M1 checkpoint smoke

M1_DIR = Path(__file__).parent.parent / "runs" / "m1"


def _latest_safe_m1_ckpt():
    """Newest m1 checkpoint that is not being written right now (the run is
    LIVE): prefer the newest file whose mtime is > 120s old."""
    if not M1_DIR.is_dir():
        return None
    ckpts = sorted(M1_DIR.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    fresh_cutoff = time.time() - 120
    safe = [p for p in ckpts if p.stat().st_mtime < fresh_cutoff]
    return safe[-1] if safe else None


@needs_sim
@pytest.mark.timeout(600)
def test_resume_from_m1_checkpoint_smoke(tmp_path):
    """Load the live M1 run's latest checkpoint READ-ONLY (copied to a temp
    dir first), enable snapshot mixing with a tiny bot-generated pool, and
    train 2 iterations on cpu: losses finite, no shape errors, win_ante and
    step counters restored."""
    import torch

    from balatro_train.config import _build

    src = _latest_safe_m1_ckpt()
    if src is None:
        pytest.skip("no m1 checkpoint available")
    ckpt_copy = tmp_path / src.name
    shutil.copyfile(src, ckpt_copy)  # never touch the live run's files

    ckpt = torch.load(ckpt_copy, map_location="cpu", weights_only=False)
    m1_cfg = _build(TrainConfig, ckpt["config"], "m1ckpt")
    assert {"policy", "optimizer", "ret_norm", "global_step", "iteration",
            "win_ante"} <= set(ckpt)

    pool = _harvest_sim_pool(tmp_path / "pool.bin", want=6, ante=2)
    n_envs, rollout = 8, 8
    cfg = TrainConfig(
        env=EnvConfig(name="sim", num_envs=n_envs, seed=0),
        policy=m1_cfg.policy,  # dims must match the checkpoint
        ppo=PPOConfig(total_timesteps=10**12, rollout_len=rollout,
                      minibatch_size=32, update_epochs=1, lr=1e-4,
                      anneal_lr=False),
        curriculum=CurriculumConfig(enabled=True, start_ante=1,
                                    ladder=[1, 2, 3, 5, 8],
                                    eval_every_steps=10**12,
                                    milestone_every_steps=10**12),
        snapshots=SnapshotConfig(enabled=True, pool_path=str(pool),
                                 fraction=0.5),
        log=LogConfig(run_name="resume_smoke", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )
    trainer = PPOTrainer(cfg)
    trainer.load_checkpoint(ckpt_copy)
    assert trainer.global_step == ckpt["global_step"] > 0
    assert trainer.win_ante == ckpt["win_ante"]
    assert trainer.env._env.win_ante == ckpt["win_ante"]
    assert trainer.snapshot_fraction == 0.5

    start_iter = trainer.iteration
    trainer.train(total_timesteps=trainer.global_step + 2 * n_envs * rollout)
    assert trainer.iteration == start_iter + 2
    assert len(trainer.history) == 2
    for h in trainer.history:
        assert np.isfinite(h["loss"]) and np.isfinite(h["v_loss"])
        assert np.isfinite(h["approx_kl"])
        assert h["snapshot_fraction"] == 0.5
