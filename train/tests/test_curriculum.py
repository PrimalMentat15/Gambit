"""Ante-limit curriculum: env win_ante semantics (mock reference), config
keys, and the trainer's promotion scheduling."""

import numpy as np
import pytest

from balatro_train import encoding as E
from balatro_train.config import (CurriculumConfig, EnvConfig, LogConfig,
                                  PolicyConfig, PPOConfig, TrainConfig,
                                  load_config)
from balatro_train.encoding import ActionType, BlindKind, Phase
from balatro_train.mock_env import MockBalatroEnv, _Game
from balatro_train.ppo import PPOTrainer

# ---------------------------------------------------------------- env side


def _play_action(n=1):
    actions = {
        k: np.full((1, *shape), -1, dtype=dt) for k, (shape, dt) in E.ACTION_SPEC.items()
    }
    actions["action_type"][0] = ActionType.PLAY_HAND
    actions["cards"][0, :n] = np.arange(n)
    actions["n_cards"][0] = n
    return actions


def test_mock_win_ante_boss_clear_wins():
    """Clearing the win_ante boss ends the episode won with the +15 bonus."""
    game = _Game(seed=0, reward_mode="shaped", bandit_episode_len=32, win_ante=1)
    game.phase = Phase.PLAYING
    game.blind_kind = BlindKind.BOSS
    game.ante = 1
    game.blind_req = 100.0
    game.blind_scored = 100.0  # already at the bar: any PLAY gain clears it
    game.hands_left = 4
    game.hand = [(0, 0)] * 8

    reward, done, info = game.apply(_play_action(), 0, beta=1.0)
    assert done and info["episode"]["won"]
    assert reward >= 15.0


def test_mock_win_ante_8_unchanged():
    """Default win_ante: an ante-1 boss clear does NOT win."""
    game = _Game(seed=0, reward_mode="shaped", bandit_episode_len=32)
    game.phase = Phase.PLAYING
    game.blind_kind = BlindKind.BOSS
    game.ante = 1
    game.blind_req = 100.0
    game.blind_scored = 100.0
    game.hands_left = 4
    game.hand = [(0, 0)] * 8

    reward, done, info = game.apply(_play_action(), 0, beta=1.0)
    assert not done and "episode" not in info
    assert reward < 15.0


def test_set_win_ante_applies_at_next_reset():
    env = MockBalatroEnv(2, win_ante=8)
    env.reset([0, 1])
    env.set_win_ante(3)
    game = env._games[0]
    assert game.win_ante == 8, "in-flight episode must keep its goalpost"
    assert game.pending_win_ante == 3
    game._finish(0.0, True)  # episode end -> auto-reset
    assert game.win_ante == 3
    # A full reset applies it everywhere.
    env.reset([0, 1])
    assert all(g.win_ante == 3 for g in env._games)


def test_win_ante_validation():
    with pytest.raises(ValueError):
        MockBalatroEnv(1, win_ante=0)
    env = MockBalatroEnv(1)
    with pytest.raises(ValueError):
        env.set_win_ante(9)


# ------------------------------------------------------------- config side


def test_curriculum_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "curriculum:\n"
        "  enabled: true\n"
        "  start_ante: 2\n"
        "  ladder: [2, 3, 8]\n"
        "  promote_winrate: 0.5\n"
        "  eval_every_steps: 1000\n"
        "  promotion_eval_episodes: 8\n"
    )
    cfg = load_config(p)
    cur = cfg.curriculum
    assert cur.enabled and cur.start_ante == 2 and cur.ladder == [2, 3, 8]
    assert cur.promote_winrate == 0.5
    assert cur.eval_every_steps == 1000 and cur.promotion_eval_episodes == 8
    # Defaults survive partial specification.
    assert cur.milestone_every_steps == 25_000_000
    # Disabled by default.
    assert CurriculumConfig().enabled is False


# ------------------------------------------------------------ trainer side


def _cfg(tmp_path, curriculum, total=3 * 8 * 8):
    return TrainConfig(
        env=EnvConfig(name="mock", num_envs=8, seed=0,
                      kwargs={"reward_mode": "bandit"}),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=total, rollout_len=8, minibatch_size=64,
                      update_epochs=1, lr=1e-3),
        curriculum=curriculum,
        log=LogConfig(run_name="test", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )


@pytest.mark.timeout(120)
def test_trainer_promotes_along_ladder(tmp_path, monkeypatch):
    """With every promo eval reporting 100% wins, the trainer walks the
    ladder one rung per eval and pushes the new ante to the env."""
    cur = CurriculumConfig(enabled=True, start_ante=1, ladder=[1, 2, 3, 5, 8],
                           promote_winrate=0.7, eval_every_steps=64,
                           promotion_eval_episodes=4, eval_num_envs=4,
                           milestone_every_steps=10**12)
    trainer = PPOTrainer(_cfg(tmp_path, cur))
    assert trainer.win_ante == 1
    assert trainer.env._win_ante == 1  # applied before reset

    evals = []

    def fake_eval(win_ante, episodes):
        evals.append((trainer.global_step, win_ante, episodes))
        return {"win_rate": 1.0, "episodes": episodes, "ante_mean": 2.0,
                "return_mean": 0.0, "length_mean": 1.0, "ante_hist": {}}

    monkeypatch.setattr(trainer, "_run_eval", fake_eval)
    trainer.train()  # 3 iterations x 64 steps -> 3 promo evals

    assert [e[1] for e in evals] == [1, 2, 3], f"unexpected evals {evals}"
    assert all(e[2] == 4 for e in evals)
    assert trainer.win_ante == 5
    assert trainer.env._win_ante == 5
    # The logged stat reflects the goal DURING the iteration (promotion runs
    # after logging), so the last iteration still reports ante 3.
    assert trainer.history[-1]["win_ante"] == 3


@pytest.mark.timeout(120)
def test_trainer_promo_eval_end_to_end(tmp_path):
    """Un-mocked promotion eval runs against the real mock env (bandit
    episodes never win, so the trainer must NOT promote)."""
    cur = CurriculumConfig(enabled=True, start_ante=1, eval_every_steps=64,
                           promotion_eval_episodes=4, eval_num_envs=4,
                           milestone_every_steps=128, milestone_episodes=4)
    trainer = PPOTrainer(_cfg(tmp_path, cur, total=2 * 8 * 8))
    trainer.train()
    assert trainer.win_ante == 1
    # Milestone threshold (128) was crossed on iteration 2.
    assert trainer._next_milestone_eval > 128


def test_win_ante_checkpoint_roundtrip(tmp_path):
    cur = CurriculumConfig(enabled=True, start_ante=1, eval_every_steps=10**12,
                           milestone_every_steps=10**12)
    trainer = PPOTrainer(_cfg(tmp_path, cur, total=8 * 8))
    trainer.train()
    trainer.win_ante = 3
    path = trainer.save_checkpoint(tmp_path / "ck.pt")

    trainer2 = PPOTrainer(_cfg(tmp_path, cur, total=8 * 8))
    trainer2.load_checkpoint(path)
    assert trainer2.win_ante == 3
    assert trainer2.env._win_ante == 3
