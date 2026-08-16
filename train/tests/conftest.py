import numpy as np
import pytest
import torch

from balatro_train.config import PolicyConfig
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch

SMALL_POLICY = PolicyConfig(
    d_model=32, n_layers=1, n_heads=2, ffn_mult=2, global_hidden=32, value_hidden=32
)


@pytest.fixture
def small_policy():
    torch.manual_seed(0)
    return BalatroPolicy(SMALL_POLICY)


@pytest.fixture
def env16():
    env = MockBalatroEnv(num_envs=16)
    env.reset(list(range(16)))
    return env


def rollout_states(num_envs=16, steps=30, seed=0, reward_mode="shaped"):
    """Drive the mock with a uniform-random legal policy; yield (obs, masks)
    pairs covering many phases. Uses numpy only (no policy)."""
    rng = np.random.default_rng(seed)
    env = MockBalatroEnv(num_envs=num_envs, reward_mode=reward_mode)
    obs, masks = env.reset([seed * 1000 + i for i in range(num_envs)])
    states = [(obs, masks)]
    for _ in range(steps):
        actions = random_legal_actions(rng, obs, masks)
        obs, masks, _r, _d, _i = env.step(actions)
        states.append((obs, masks))
    return states


def random_legal_actions(rng, obs, masks):
    """Sample a uniformly random legal composite action per env (numpy)."""
    from balatro_train import encoding as E
    from balatro_train.encoding import ActionType

    n = masks["action_type_mask"].shape[0]
    actions = {
        k: np.full((n, *shape), -1, dtype=dt) for k, (shape, dt) in E.ACTION_SPEC.items()
    }
    actions["action_type"] = np.zeros(n, dtype=np.int64)
    actions["n_cards"] = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = ActionType(int(rng.choice(np.flatnonzero(masks["action_type_mask"][i]))))
        actions["action_type"][i] = t
        for field, mask_key, needed in (
            ("joker_target", "joker_target_mask", t in E.ACTION_NEEDS_JOKER),
            ("consumable_target", "consumable_target_mask", t in E.ACTION_NEEDS_CONSUMABLE),
            ("shop_target", "shop_target_mask", t in E.ACTION_NEEDS_SHOP),
            ("pack_target", "pack_target_mask", t in E.ACTION_NEEDS_PACK),
        ):
            if needed:
                actions[field][i] = int(rng.choice(np.flatnonzero(masks[mask_key][i])))
        if t in E.ACTION_NEEDS_CARDS:
            avail = np.flatnonzero(masks["card_select_mask"][i])
            lo = E.CARD_PICK_MIN[t]
            hi = min(E.MAX_CARD_PICKS, avail.size)
            k = int(rng.integers(lo, hi + 1)) if hi >= lo else 0
            picks = rng.choice(avail, size=k, replace=False) if k else []
            actions["cards"][i, :k] = picks
            actions["n_cards"][i] = k
    return actions


def policy_step(policy, env, obs, masks, deterministic=False, need_aux=False):
    """One policy-driven env step; returns everything."""
    obs_t = obs_to_torch(obs, torch.device("cpu"))
    masks_t = obs_to_torch(masks, torch.device("cpu"))
    actions, logp, ent, value, aux = policy.act(
        obs_t, masks_t, deterministic=deterministic, need_aux=need_aux
    )
    actions_np = actions_to_numpy(actions)
    step_out = env.step(actions_np)
    return actions_np, logp, ent, value, aux, obs_t, masks_t, actions, step_out
