"""Mock env contract conformance: specs, mask semantics, determinism, auto-reset."""

import numpy as np

from balatro_train import encoding as E
from balatro_train.encoding import ActionType, Phase
from balatro_train.env_api import VecEnv, make_vec_env
from balatro_train.mock_env import MockBalatroEnv

from conftest import random_legal_actions


def phase_of(obs, i):
    onehot = obs["global"][i, E.GLOBAL_PHASE_OFF : E.GLOBAL_PHASE_OFF + E.N_PHASES]
    assert onehot.sum() == 1.0
    return Phase(int(onehot.argmax()))


def test_protocol_and_specs():
    env = make_vec_env("mock", 4)
    assert isinstance(env, VecEnv)
    assert env.num_envs == 4
    obs, masks = env.reset([0, 1, 2, 3])
    E.validate_batch(E.OBS_SPEC, obs, 4)
    E.validate_batch(E.MASK_SPEC, masks, 4)


def test_random_legal_agent_never_rejected():
    """M0-style invariant: uniformly-random legal actions are always accepted
    by the strict validator, across hundreds of steps covering all phases."""
    rng = np.random.default_rng(7)
    env = MockBalatroEnv(num_envs=32)
    obs, masks = env.reset(list(range(32)))
    phases_seen = set()
    for _ in range(300):
        for i in range(32):
            phases_seen.add(phase_of(obs, i))
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, reward, done, infos = env.step(actions)
        assert reward.dtype == np.float32 and done.dtype == np.bool_
    # a healthy mock must exercise every non-terminal phase
    assert phases_seen >= {Phase.BLIND_SELECT, Phase.PLAYING, Phase.ROUND_EVAL,
                           Phase.SHOP, Phase.PACK}


def test_mask_semantics_per_phase():
    rng = np.random.default_rng(3)
    env = MockBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(16)))
    for _ in range(200):
        for i in range(16):
            phase = phase_of(obs, i)
            at = masks["action_type_mask"][i]
            hand_len = int(obs["hand_len"][i])
            # card mask sized exactly to hand_len
            assert not masks["card_select_mask"][i, hand_len:].any()
            if phase == Phase.PLAYING:
                assert masks["card_select_mask"][i, :hand_len].all()
            # pack targets only in pack phase; shop targets only in shop phase
            if phase != Phase.PACK:
                assert not masks["pack_target_mask"][i].any()
                assert not at[ActionType.PICK_PACK] and not at[ActionType.SKIP_PACK]
            if phase != Phase.SHOP:
                assert not masks["shop_target_mask"][i].any()
                assert not at[ActionType.BUY_SHOP] and not at[ActionType.REROLL]
            # phase-exclusive action types
            if phase == Phase.ROUND_EVAL:
                legal = set(np.flatnonzero(at))
                assert legal == {ActionType.CASH_OUT}
            else:
                assert not at[ActionType.CASH_OUT]
            # guarantee: legal action types have a legal target
            assert at.any()
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


def test_illegal_actions_rejected():
    env = MockBalatroEnv(num_envs=2)
    obs, masks = env.reset([0, 1])

    def base_actions():
        a = {k: np.full((2, *s), -1, dtype=d) for k, (s, d) in E.ACTION_SPEC.items()}
        a["action_type"] = np.zeros(2, dtype=np.int64)
        a["n_cards"] = np.zeros(2, dtype=np.int64)
        return a

    # illegal action type (PLAY_HAND during blind select)
    a = base_actions()
    a["action_type"][:] = ActionType.PLAY_HAND
    try:
        env.step(a)
        raise AssertionError("illegal action_type accepted")
    except ValueError:
        pass
    # stray param on a paramless action
    env.reset([0, 1])
    a = base_actions()
    a["action_type"][:] = ActionType.SELECT_BLIND
    a["joker_target"][0] = 0
    try:
        env.step(a)
        raise AssertionError("stray param accepted")
    except ValueError:
        pass


def test_determinism_same_seeds_same_actions():
    rng1, rng2 = np.random.default_rng(11), np.random.default_rng(11)
    outs = []
    for rng in (rng1, rng2):
        env = MockBalatroEnv(num_envs=8)
        obs, masks = env.reset(list(range(8)))
        trace = []
        for _ in range(100):
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
    env = MockBalatroEnv(num_envs=8)
    obs, masks = env.reset(list(range(8)))
    done_seen = 0
    for _ in range(500):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, done, infos = env.step(actions)
        for i in range(8):
            if done[i]:
                done_seen += 1
                ep = infos[i]["episode"]
                assert set(ep) >= {"r", "l", "ante", "won"}
                assert ep["l"] >= 1
                # auto-reset: row i is already a fresh ante-1 episode
                assert obs["global"][i, E.GLOBAL_ANTE_OFF] == 1.0
                assert phase_of(obs, i) == Phase.BLIND_SELECT
            else:
                assert "episode" not in infos[i]
    assert done_seen > 0, "random agent should lose at least once in 500 steps"


def test_shaping_beta_scales_shaped_rewards():
    def shaped_reward(beta):
        """Total reward minus (unscaled) win bonuses, over a fixed action stream."""
        rng = np.random.default_rng(2)
        env = MockBalatroEnv(num_envs=4)
        env.set_shaping_beta(beta)
        obs, masks = env.reset([10, 11, 12, 13])
        tot, wins = 0.0, 0
        for _ in range(200):
            actions = random_legal_actions(rng, obs, masks)
            obs, masks, r, _d, infos = env.step(actions)
            tot += float(r.sum())
            wins += sum(1 for i in infos if i.get("episode", {}).get("won"))
        return tot - 15.0 * wins, wins

    r_full, w_full = shaped_reward(1.0)
    r_tenth, w_tenth = shaped_reward(0.1)
    # identical rng => identical action stream => identical outcomes
    assert w_full == w_tenth
    assert r_full > 0
    # shaping terms (1)+(2) scale linearly with beta; win bonus does not
    assert abs(r_tenth - 0.1 * r_full) < 1e-6
