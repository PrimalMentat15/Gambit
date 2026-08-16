"""PPO integration: overfit on the mock's trivially-learnable reward, and
bit-exact determinism of seeded runs."""

import pytest
import torch

from balatro_train.config import (EnvConfig, LogConfig, PolicyConfig, PPOConfig,
                                  TrainConfig)
from balatro_train.ppo import PPOTrainer


def _cfg(tmp_path, reward_mode="bandit", total=24576, seed=1):
    return TrainConfig(
        env=EnvConfig(name="mock", num_envs=16, seed=0,
                      kwargs={"reward_mode": reward_mode}),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=total, rollout_len=64, minibatch_size=512,
                      update_epochs=4, lr=1e-3, ent_coef=0.003),
        log=LogConfig(run_name="test", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=seed,
        device="cpu",
        bf16_rollout=False,
    )


@pytest.mark.timeout(120)
def test_overfit_bandit_reward(tmp_path):
    """Bandit mode: DISCARD always pays +1 (max 32/episode). PPO must reach
    >= 90% of optimal return within a small CPU budget (<2 min)."""
    trainer = PPOTrainer(_cfg(tmp_path))
    history = trainer.train()
    best = max(h.get("ep_return_mean", 0.0) for h in history)
    assert best >= 0.9 * 32, f"best ep_return_mean {best:.2f} < 28.8"
    # report convergence point for the log
    for h in history:
        if h.get("ep_return_mean", 0.0) >= 0.9 * 32:
            print(f"\nconverged at iteration {h['iteration']}, "
                  f"step {h['global_step']} (ep_return {h['ep_return_mean']:.2f})")
            break


@pytest.mark.timeout(300)
def test_seeded_runs_are_identical(tmp_path):
    """Two trainers with identical config+seed produce identical losses."""

    def run(subdir):
        cfg = _cfg(tmp_path / subdir, reward_mode="shaped", total=3 * 16 * 64)
        trainer = PPOTrainer(cfg)
        history = trainer.train()
        return [(h["loss"], h["pg_loss"], h["v_loss"], h["entropy"]) for h in history]

    h1, h2 = run("a"), run("b")
    assert len(h1) == 3
    assert h1 == h2, "seeded reruns diverged"


def test_checkpoint_roundtrip(tmp_path):
    cfg = _cfg(tmp_path, reward_mode="shaped", total=16 * 64)
    trainer = PPOTrainer(cfg)
    trainer.train()
    path = trainer.save_checkpoint(tmp_path / "ck.pt")

    cfg2 = _cfg(tmp_path, reward_mode="shaped", total=16 * 64, seed=99)
    trainer2 = PPOTrainer(cfg2)
    trainer2.load_checkpoint(path)
    for p1, p2 in zip(trainer.policy.parameters(), trainer2.policy.parameters()):
        assert torch.equal(p1, p2)
    assert trainer2.global_step == trainer.global_step
    if trainer.ret_norm:
        assert trainer2.ret_norm.rms.state_dict() == trainer.ret_norm.rms.state_dict()
