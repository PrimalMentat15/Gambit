"""Live smoke: the deck curriculum's torch-path wiring in a real trainer.

The schedule functions are unit-tested elsewhere; this checks the parts that
only exist inside PPOTrainer -- that `_apply_deck_curriculum` pushes a
distribution to the env, that a run with the curriculum enabled trains without
crashing, and that `_run_deck_matrix_eval` returns a per-deck grid. Uses the
sim env (the curriculum only means anything there) with a tiny policy and a
handful of iterations, on CPU.
"""

import numpy as np
import pytest

from balatro_train.config import (
    CurriculumConfig, EnvConfig, LogConfig, PolicyConfig, PPOConfig, TrainConfig,
)
from balatro_train.deck_curriculum import DeckCurriculumConfig, active_decks
from balatro_train.ppo import PPOTrainer


def _cfg(tmp_path, tier_steps):
    return TrainConfig(
        env=EnvConfig(name="sim", num_envs=8, seed=0),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=8 * 64 * 3, rollout_len=64,
                      minibatch_size=256, update_epochs=1, lr=1e-3,
                      anneal_lr=False),
        curriculum=CurriculumConfig(enabled=True, start_ante=8, ladder=[8],
                                    eval_every_steps=10**12,
                                    milestone_every_steps=10**12),
        deck_curriculum=DeckCurriculumConfig(enabled=True, tier_steps=tier_steps),
        log=LogConfig(run_name="dc_smoke", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1, device="cpu", bf16_rollout=False,
    )


@pytest.mark.timeout(180)
def test_curriculum_trains_and_pushes_distribution(tmp_path):
    # All tiers unlocked immediately (tier_steps in the past) so every deck is
    # live -- exercises the full 15-wide distribution.
    trainer = PPOTrainer(_cfg(tmp_path, tier_steps=[0, 0, 0, 0]))
    # Construction already called _apply_deck_curriculum once; the env's
    # distribution should be live. Train a few iterations without crashing.
    trainer.train()
    # After training, episodes across multiple decks should have been recorded.
    decks_seen = {d for (d, _st) in trainer.ep_wins_by_cohort}
    assert len(decks_seen) >= 2, f"only saw decks {decks_seen}"


@pytest.mark.timeout(180)
def test_deck_matrix_eval_returns_a_grid(tmp_path):
    trainer = PPOTrainer(_cfg(tmp_path, tier_steps=[0, 0, 0, 0]))
    live = active_decks(0, trainer.cfg.deck_curriculum)
    # Both the argmax (deployment) and sampled (brittleness comparator) passes
    # must return a full per-deck grid over the live decks.
    matrix = trainer._run_deck_matrix_eval(episodes=8)
    sampled = trainer._run_deck_matrix_eval(episodes=8, deterministic=False)
    for grid in (matrix, sampled):
        assert set(grid) == set(live), "matrix decks != active decks"
        assert all(0.0 <= wr <= 1.0 for wr in grid.values())


@pytest.mark.timeout(180)
def test_cohort_ema_survives_checkpoint_roundtrip(tmp_path):
    # The deck sampler reads per-cohort win-rate EMAs; they must survive a
    # resume or the deck mixture resets on every restart (the perturbation that
    # tipped Black's brittle argmax at ~1.8B).
    t1 = PPOTrainer(_cfg(tmp_path, tier_steps=[0, 0, 0, 0]))
    t1.ep_wins_by_cohort = {(0, 1): 0.7, (4, 1): 0.27, (13, 1): 0.4}
    t1.ep_count_by_cohort = {(0, 1): 500, (4, 1): 900, (13, 1): 300}
    path = t1.save_checkpoint()

    t2 = PPOTrainer(_cfg(tmp_path, tier_steps=[0, 0, 0, 0]))
    assert t2.ep_wins_by_cohort == {}, "fresh trainer should start empty"
    t2.load_checkpoint(path)
    assert t2.ep_wins_by_cohort == t1.ep_wins_by_cohort
    assert t2.ep_count_by_cohort == t1.ep_count_by_cohort


@pytest.mark.timeout(120)
def test_tiers_gate_which_decks_are_sampled(tmp_path):
    # Tiers far in the future -> only Red is live at step 0, so every episode
    # is Red and the cohort map has exactly one deck.
    trainer = PPOTrainer(_cfg(tmp_path, tier_steps=[10**12] * 4))
    trainer.train()
    decks_seen = {d for (d, _st) in trainer.ep_wins_by_cohort}
    assert decks_seen == {0}, f"expected only Red, saw {decks_seen}"
