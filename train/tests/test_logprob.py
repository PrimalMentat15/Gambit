"""Joint logprob consistency + the joint distribution is properly normalized."""

import itertools

import numpy as np
import torch

from balatro_train import encoding as E
from balatro_train.encoding import ActionType
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch

from conftest import SMALL_POLICY


def test_recompute_matches_sampling_time():
    """evaluate_actions under the SAME stored masks reproduces the joint
    logprob (and entropy and value) computed at sampling time."""
    torch.manual_seed(3)
    policy = BalatroPolicy(SMALL_POLICY)
    env = MockBalatroEnv(num_envs=24)
    obs, masks = env.reset(list(range(24)))
    device = torch.device("cpu")
    for _ in range(60):
        obs_t, masks_t = obs_to_torch(obs, device), obs_to_torch(masks, device)
        actions, logp, ent, value, _ = policy.act(obs_t, masks_t)
        logp2, ent2, value2, _ = policy.evaluate_actions(obs_t, masks_t, actions)
        assert torch.allclose(logp, logp2, atol=1e-6), (logp - logp2).abs().max()
        assert torch.allclose(ent, ent2, atol=1e-6)
        assert torch.allclose(value, value2, atol=1e-6)
        obs, masks, _r, _d, _infos = env.step(actions_to_numpy(actions))


def _synthetic_state(policy):
    """One env with 3 selectable hand cards; PLAY/DISCARD/SELECT_BLIND legal."""
    obs = {k: np.zeros((1, *s), dtype=d) for k, (s, d) in E.OBS_SPEC.items()}
    obs["hand_len"][0] = 3
    obs["hand"][0, :3, E.CARD_RANK_OFF] = 1.0
    masks = {k: np.zeros((1, *s), dtype=d) for k, (s, d) in E.MASK_SPEC.items()}
    masks["action_type_mask"][0, [ActionType.PLAY_HAND, ActionType.DISCARD,
                                  ActionType.SELECT_BLIND]] = True
    masks["card_select_mask"][0, :3] = True
    return obs_to_torch(obs, torch.device("cpu")), obs_to_torch(masks, torch.device("cpu"))


def _all_composite_actions():
    """Every legal composite action for the synthetic state."""
    acts = []
    empty = {k: np.full((1, *s), -1, dtype=d) for k, (s, d) in E.ACTION_SPEC.items()}
    empty["n_cards"] = np.zeros(1, dtype=np.int64)

    a = {k: v.copy() for k, v in empty.items()}
    a["action_type"] = np.array([ActionType.SELECT_BLIND], dtype=np.int64)
    acts.append(a)
    for t in (ActionType.PLAY_HAND, ActionType.DISCARD):
        for n in (1, 2, 3):
            for seq in itertools.permutations(range(3), n):
                a = {k: v.copy() for k, v in empty.items()}
                a["action_type"] = np.array([t], dtype=np.int64)
                a["cards"][0, :n] = seq
                a["n_cards"][0] = n
                acts.append(a)
    return acts  # 1 + 2 * (3 + 6 + 6) = 31 ordered pick sequences


def test_joint_probability_sums_to_one():
    """Sum of exp(joint logprob) over ALL legal composite actions == 1.

    This pins down the autoregressive factorization: sub-step masks, STOP
    semantics (forced stop at 3 available cards), and the pick-min rule."""
    torch.manual_seed(4)
    policy = BalatroPolicy(SMALL_POLICY)
    obs_t, masks_t = _synthetic_state(policy)
    total = 0.0
    for a in _all_composite_actions():
        at = obs_to_torch(a, torch.device("cpu"))
        logp, _ent, _v, _ = policy.evaluate_actions(obs_t, masks_t, at)
        total += float(logp.detach().exp())
    assert abs(total - 1.0) < 1e-5, total
