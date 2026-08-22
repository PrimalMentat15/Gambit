"""Contract conformance of the REAL (Rust PyO3) vec-env — the same invariants
the mock tests pin, adapted where the two envs legitimately differ (the sim
allows sell/use inventory actions in more phases than the mock's
simplification, so the phase-exclusivity assertions here are the contract's,
not the mock's).

Skipped wholesale when the extension is not built
(`pip install ./sim/py`); the mock tests keep the training stack green.
"""

import numpy as np
import pytest

pytest.importorskip("balatro_sim")

from balatro_train import encoding as E
from balatro_train.encoding import ActionType, Phase
from balatro_train.env_api import VecEnv, make_vec_env
from balatro_train.sim_env import SimBalatroEnv

from conftest import random_legal_actions


def phase_of(obs, i):
    onehot = obs["global"][i, E.GLOBAL_PHASE_OFF : E.GLOBAL_PHASE_OFF + E.N_PHASES]
    assert onehot.sum() == 1.0
    return Phase(int(onehot.argmax()))


def test_protocol_and_specs():
    env = make_vec_env("sim", 4)
    assert isinstance(env, VecEnv)
    assert env.num_envs == 4
    obs, masks = env.reset([0, 1, 2, 3])
    E.validate_batch(E.OBS_SPEC, obs, 4)
    E.validate_batch(E.MASK_SPEC, masks, 4)
    # identical seeds must yield identical runs
    o2, _ = make_vec_env("sim", 4).reset([0, 1, 2, 3])
    for k in obs:
        assert (obs[k] == o2[k]).all(), k


def test_random_legal_agent_never_rejected_and_covers_phases():
    rng = np.random.default_rng(7)
    env = SimBalatroEnv(num_envs=32)
    obs, masks = env.reset(list(range(32)))
    phases_seen = set()
    for _ in range(500):
        for i in range(32):
            phases_seen.add(phase_of(obs, i))
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, reward, done, infos = env.step(actions)
        assert reward.dtype == np.float32 and done.dtype == np.bool_
        assert np.isfinite(reward).all()
    assert phases_seen >= {Phase.BLIND_SELECT, Phase.PLAYING, Phase.ROUND_EVAL,
                           Phase.SHOP, Phase.PACK}


def test_mask_semantics():
    rng = np.random.default_rng(3)
    env = SimBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(16)))
    for _ in range(300):
        for i in range(16):
            phase = phase_of(obs, i)
            at = masks["action_type_mask"][i]
            hand_len = int(obs["hand_len"][i])
            assert at.any()
            # card mask never exceeds hand_len
            assert not masks["card_select_mask"][i, hand_len:].any()
            # reserved action types (MOVE_JOKER+) never legal
            assert not at[ActionType.MOVE_JOKER :].any()
            # phase-gated action types
            if phase != Phase.PACK:
                assert not masks["pack_target_mask"][i].any()
                assert not at[ActionType.PICK_PACK] and not at[ActionType.SKIP_PACK]
            if phase != Phase.SHOP:
                assert not masks["shop_target_mask"][i].any()
                assert not at[ActionType.BUY_SHOP] and not at[ActionType.REROLL]
                assert not at[ActionType.LEAVE_SHOP]
            if phase != Phase.ROUND_EVAL:
                assert not at[ActionType.CASH_OUT]
            else:
                assert at[ActionType.CASH_OUT]
            if phase != Phase.BLIND_SELECT:
                assert not at[ActionType.SELECT_BLIND] and not at[ActionType.SKIP_BLIND]
            # pointer guarantees
            if at[ActionType.PLAY_HAND] or at[ActionType.DISCARD]:
                assert masks["card_select_mask"][i].any()
            if at[ActionType.BUY_SHOP]:
                assert masks["shop_target_mask"][i].any()
            if at[ActionType.PICK_PACK]:
                assert masks["pack_target_mask"][i].any()
            if at[ActionType.SELL_JOKER]:
                assert masks["joker_target_mask"][i].any()
            if at[ActionType.USE_CONSUMABLE] or at[ActionType.SELL_CONSUMABLE]:
                assert masks["consumable_target_mask"][i].any()
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, _infos = env.step(actions)


def test_global_deck_stake_and_dtypes():
    rng = np.random.default_rng(9)
    env = SimBalatroEnv(num_envs=8)
    obs, masks = env.reset(list(range(8)))
    deck = slice(E.GLOBAL_DECK_OFF, E.GLOBAL_DECK_OFF + E.N_DECKS)
    for _ in range(100):
        # The deck block is exactly one-hot in every env, every step: the deck
        # is fixed for the whole episode and must never read as "no deck".
        assert (obs["global"][:, deck].sum(axis=1) == 1.0).all()
        stake = obs["global"][:, E.GLOBAL_STAKE]
        assert ((stake >= 0.0) & (stake <= 1.0)).all()
        for k in obs:
            assert np.isfinite(obs[k]).all(), k
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, _infos = env.step(actions)


def test_illegal_actions_rejected():
    env = SimBalatroEnv(num_envs=2)
    env.reset([0, 1])

    def base_actions():
        a = {k: np.full((2, *s), -1, dtype=d) for k, (s, d) in E.ACTION_SPEC.items()}
        a["action_type"] = np.zeros(2, dtype=np.int64)
        a["n_cards"] = np.zeros(2, dtype=np.int64)
        return a

    # PLAY_HAND during blind select
    a = base_actions()
    a["action_type"][:] = ActionType.PLAY_HAND
    with pytest.raises(ValueError):
        env.step(a)
    # stray param on a paramless action
    env.reset([0, 1])
    a = base_actions()
    a["action_type"][:] = ActionType.SELECT_BLIND
    a["joker_target"][0] = 0
    with pytest.raises(ValueError):
        env.step(a)


def test_determinism_bit_identical():
    rng1, rng2 = np.random.default_rng(11), np.random.default_rng(11)
    outs = []
    for rng in (rng1, rng2):
        env = SimBalatroEnv(num_envs=8)
        obs, masks = env.reset(list(range(8)))
        trace = []
        for _ in range(200):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, r, d, _ = env.step(actions)
            trace.append((r.copy(), d.copy(), {k: v.copy() for k, v in obs.items()}))
        outs.append(trace)
    for (r1, d1, o1), (r2, d2, o2) in zip(*outs):
        assert (r1 == r2).all() and (d1 == d2).all()
        for k in o1:
            assert (o1[k] == o2[k]).all(), k


def test_auto_reset_and_episode_info():
    rng = np.random.default_rng(5)
    env = SimBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(16)))
    done_seen = 0
    for _ in range(800):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, done, infos = env.step(actions)
        for i in range(16):
            if done[i]:
                done_seen += 1
                ep = infos[i]["episode"]
                assert set(ep) >= {"r", "l", "ante", "won"}
                assert ep["l"] >= 1 and ep["ante"] >= 1
                # auto-reset: row i is already a fresh ante-1 episode
                assert obs["global"][i, E.GLOBAL_ANTE_OFF] == 1.0
                assert phase_of(obs, i) == Phase.BLIND_SELECT
            else:
                assert "episode" not in infos[i]
    assert done_seen > 0, "random agent should lose within 800 steps"


def test_shaping_beta_scales_shaping_terms_only():
    def shaped_reward(beta):
        rng = np.random.default_rng(2)
        env = SimBalatroEnv(num_envs=8)
        env.set_shaping_beta(beta)
        obs, masks = env.reset([100 + i for i in range(8)])
        tot, wins = 0.0, 0
        for _ in range(400):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, r, _d, infos = env.step(actions)
            tot += float(np.asarray(r, dtype=np.float64).sum())
            wins += sum(1 for i in infos if i.get("episode", {}).get("won"))
        return tot - 15.0 * wins, wins

    r_full, w_full = shaped_reward(1.0)
    r_tenth, w_tenth = shaped_reward(0.1)
    assert w_full == w_tenth  # beta must not change the dynamics
    assert r_full > 0
    assert abs(r_tenth - 0.1 * r_full) < 1e-4


def test_snapshot_restore_roundtrip_mid_episode():
    rng = np.random.default_rng(21)
    env = SimBalatroEnv(num_envs=4)
    obs, masks = env.reset([7, 8, 9, 10])
    for _ in range(60):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, _ = env.step(actions)
    snaps = [env.snapshot(i) for i in range(4)]
    assert all(isinstance(s, bytes) and len(s) > 0 for s in snaps)

    # Continue for a recorded stretch...
    stream_rng = np.random.default_rng(77)
    ref = []
    o, m = env.observe()
    for _ in range(50):
        actions = random_legal_actions(stream_rng, o, m)
        o, m, r, d, _ = env.step(actions)
        ref.append((actions, r.copy(), d.copy(), {k: v.copy() for k, v in o.items()}))

    # ...restore all four envs and replay the exact same actions.
    for i in range(4):
        env.restore(i, snaps[i])
    o, m = env.observe()
    for actions, r_ref, d_ref, o_ref in ref:
        o, m, r, d, _ = env.step(actions)
        assert (r == r_ref).all() and (d == d_ref).all()
        for k in o:
            assert (o[k] == o_ref[k]).all(), k


def test_redeterminize_keeps_observables():
    rng = np.random.default_rng(31)
    env = SimBalatroEnv(num_envs=2)
    obs, masks = env.reset([1, 2])
    for _ in range(40):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, _ = env.step(actions)
    before_o, before_m = env.observe()
    env.redeterminize(0, 999)
    after_o, after_m = env.observe()
    for k in before_o:
        assert (before_o[k] == after_o[k]).all(), k
    for k in before_m:
        assert (before_m[k] == after_m[k]).all(), k


def test_bot_runs_through_step_path():
    env = SimBalatroEnv(num_envs=16)
    env.reset(list(range(16)))
    episodes = 0
    antes = []
    for _ in range(1500):
        actions = env.bot_actions()
        E.validate_batch(E.ACTION_SPEC, actions, 16, "bot actions")
        _o, _m, _r, done, infos = env.step(actions)
        for i in range(16):
            if done[i]:
                episodes += 1
                antes.append(infos[i]["episode"]["ante"])
    assert episodes > 0
    # A competent baseline must at least reach ante 2 on average.
    assert float(np.mean(antes)) >= 2.0


def test_run_seed_is_valid_balatro_seed():
    env = SimBalatroEnv(num_envs=2)
    env.reset([123, 456])
    for i in range(2):
        s = env.run_seed(i)
        assert len(s) == 8 and all(c.isdigit() or c.isupper() for c in s)


@pytest.mark.timeout(300)
def test_ppo_smoke_on_sim(tmp_path):
    """Short PPO run against the real env: losses finite, entropy sane,
    checkpoint saves and loads. (The 3-minute smoke run uses
    configs/sim_debug.yaml; this is its CI-sized cousin.)"""
    import torch

    from balatro_train.config import (EnvConfig, LogConfig, PolicyConfig,
                                      PPOConfig, TrainConfig)
    from balatro_train.ppo import PPOTrainer

    cfg = TrainConfig(
        env=EnvConfig(name="sim", num_envs=16, seed=0),
        policy=PolicyConfig(d_model=32, n_layers=1, n_heads=2, ffn_mult=2,
                            global_hidden=32, value_hidden=32),
        ppo=PPOConfig(total_timesteps=4 * 16 * 32, rollout_len=32,
                      minibatch_size=256, update_epochs=2, lr=1e-3),
        log=LogConfig(run_name="simtest", log_dir=str(tmp_path / "runs"),
                      ckpt_dir=str(tmp_path / "ckpt"), tensorboard=False,
                      save_every_iters=0, log_every_iters=10**9),
        seed=1,
        device="cpu",
        bf16_rollout=False,
    )
    trainer = PPOTrainer(cfg)
    history = trainer.train()
    assert len(history) == 4
    for h in history:
        assert np.isfinite(h["loss"]) and np.isfinite(h["v_loss"])
        assert np.isfinite(h["approx_kl"])
        assert 0.0 < h["entropy"] < 20.0
    path = trainer.save_checkpoint()
    trainer2 = PPOTrainer(cfg)
    trainer2.load_checkpoint(path)
    for p1, p2 in zip(trainer.policy.parameters(), trainer2.policy.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.timeout(120)
def test_win_ante_curriculum_on_sim():
    """win_ante=1: the heuristic bot's episodes never pass ante 2, wins are
    exactly the ante-1 boss clears (post-boss ante 2) and pay the +15."""
    env = SimBalatroEnv(num_envs=16, win_ante=1)
    env.reset(list(range(16)))
    episodes, wins = 0, 0
    for _ in range(1500):
        obs, masks, reward, done, infos = env.step(env.bot_actions())
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                episodes += 1
                assert ep["ante"] <= 2
                assert ep["won"] == (ep["ante"] == 2)
                if ep["won"]:
                    wins += 1
                    assert ep["r"] >= 15.0
        if episodes >= 64 and wins > 0:
            break
    assert episodes > 0 and wins > 0

    # set_win_ante applies at the NEXT reset: a full reset applies it now.
    env.set_win_ante(8)
    env.reset(list(range(16)))
    assert env._env.win_ante == 8
