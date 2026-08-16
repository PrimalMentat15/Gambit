"""Entropy decomposition: joint = sum of conditional (per-sub-step) entropies."""

import numpy as np
import torch

from balatro_train import encoding as E
from balatro_train.encoding import ActionType, Phase
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch

from conftest import SMALL_POLICY


def _manual_entropy(dist):
    """Entropy from the distribution's log-probs by direct formula."""
    logp = dist.logp.double()
    p = logp.exp()
    terms = torch.where(p > 0, p * logp, torch.zeros_like(p))
    return -terms.sum(-1).float()


def test_joint_entropy_is_sum_of_parts_and_parts_match_manual():
    torch.manual_seed(5)
    policy = BalatroPolicy(SMALL_POLICY)
    env = MockBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(16)))
    device = torch.device("cpu")
    for _ in range(40):
        obs_t, masks_t = obs_to_torch(obs, device), obs_to_torch(masks, device)
        actions, _logp, ent, _v, aux = policy.act(obs_t, masks_t, need_aux=True)
        parts = aux["entropy_parts"]
        # decomposition: joint == sum of conditional parts
        total = sum(parts.values())
        assert torch.allclose(ent, total, atol=1e-6)
        # the action-type part matches a direct -sum(p log p) computation
        manual_type = _manual_entropy(aux["dists"]["action_type"])
        assert torch.allclose(parts["action_type"], manual_type, atol=1e-5)
        # pointer parts: zero when the head is unused, manual entropy when used
        at = actions["action_type"]
        for group, needs in (
            ("joker", E.ACTION_NEEDS_JOKER),
            ("consumable", E.ACTION_NEEDS_CONSUMABLE),
            ("shop", E.ACTION_NEEDS_SHOP),
            ("pack", E.ACTION_NEEDS_PACK),
        ):
            needed = torch.tensor([ActionType(int(a)) in needs for a in at])
            assert (parts[group][~needed] == 0.0).all()
            if needed.any():
                manual = _manual_entropy(aux["dists"][group])
                assert torch.allclose(parts[group][needed], manual[needed], atol=1e-5)
        # card part is zero for actions that take no card sub-steps
        card_needed = torch.tensor([ActionType(int(a)) in E.ACTION_NEEDS_CARDS for a in at])
        assert (parts["cards"][~card_needed] == 0.0).all()
        obs, masks, _r, _d, _infos = env.step(actions_to_numpy(actions))


def test_forced_choice_has_zero_entropy():
    """ROUND_EVAL states admit exactly one action (CASH_OUT): joint entropy 0."""
    torch.manual_seed(6)
    policy = BalatroPolicy(SMALL_POLICY)
    env = MockBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(16)))
    device = torch.device("cpu")
    checked = 0
    for _ in range(300):
        obs_t, masks_t = obs_to_torch(obs, device), obs_to_torch(masks, device)
        actions, logp, ent, _v, _aux = policy.act(obs_t, masks_t)
        for i in range(16):
            phase = obs["global"][i, E.GLOBAL_PHASE_OFF : E.GLOBAL_PHASE_OFF + E.N_PHASES].argmax()
            if Phase(int(phase)) == Phase.ROUND_EVAL:
                assert float(ent[i]) == 0.0
                assert float(logp[i]) == 0.0
                checked += 1
        obs, masks, _r, _d, _infos = env.step(actions_to_numpy(actions))
        if checked >= 20:
            break
    assert checked > 0, "never reached ROUND_EVAL; mock dynamics broken?"
