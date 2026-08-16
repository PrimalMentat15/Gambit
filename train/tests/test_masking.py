"""Masking invariants: sampled actions are always legal; -inf masking exact."""

import numpy as np
import torch

from balatro_train import encoding as E
from balatro_train.encoding import ActionType
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.policy import BalatroPolicy, MaskedCategorical, actions_to_numpy, obs_to_torch

from conftest import SMALL_POLICY


def test_masked_categorical_exact():
    torch.manual_seed(0)
    logits = torch.randn(64, 7) * 10
    mask = torch.rand(64, 7) < 0.5
    mask[:, 0] |= ~mask.any(dim=-1)  # >=1 legal
    dist = MaskedCategorical(logits, mask)
    probs = dist.probs
    # illegal entries have probability EXACTLY zero; legal renormalize to 1
    assert (probs[~mask] == 0.0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(64), atol=1e-6)
    # legal logprobs equal log_softmax restricted to the legal subset
    for i in range(64):
        legal = torch.where(mask[i])[0]
        ref = torch.log_softmax(logits[i, legal], -1)
        assert torch.allclose(dist.logp[i, legal], ref, atol=1e-6)
    # single-legal-option rows: logprob exactly 0, entropy exactly 0
    one = torch.zeros(4, 7, dtype=torch.bool)
    one[:, 3] = True
    d1 = MaskedCategorical(torch.randn(4, 7), one)
    assert (d1.logp[:, 3] == 0.0).all()
    assert (d1.entropy() == 0.0).all()
    # sampling never leaves the mask
    samples = torch.multinomial(probs, 200, replacement=True)
    assert mask.gather(1, samples).all()


def test_policy_samples_always_legal():
    """The strict mock validator sees thousands of policy-sampled composite
    actions across all phases and never rejects one (M0: zero invalid)."""
    torch.manual_seed(1)
    policy = BalatroPolicy(SMALL_POLICY)
    env = MockBalatroEnv(num_envs=32)
    obs, masks = env.reset(list(range(32)))
    device = torch.device("cpu")
    for step in range(150):
        actions, logp, ent, _v, _aux = policy.act(
            obs_to_torch(obs, device), obs_to_torch(masks, device)
        )
        a = actions_to_numpy(actions)
        # direct assertions on top of the env's own validation
        at = a["action_type"]
        assert masks["action_type_mask"][np.arange(32), at].all()
        assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
        assert (ent >= -1e-6).all()
        for i in range(32):
            t = ActionType(int(at[i]))
            n = int(a["n_cards"][i])
            if t in E.ACTION_NEEDS_CARDS:
                assert E.CARD_PICK_MIN[t] <= n <= E.MAX_CARD_PICKS
                picks = a["cards"][i, :n]
                assert len(set(picks.tolist())) == n
                assert masks["card_select_mask"][i, picks].all()
            else:
                assert n == 0 and (a["cards"][i] == -1).all()
        obs, masks, _r, _d, _infos = env.step(a)  # raises on any violation


def test_argmax_actions_also_legal():
    torch.manual_seed(2)
    policy = BalatroPolicy(SMALL_POLICY)
    env = MockBalatroEnv(num_envs=16)
    obs, masks = env.reset(list(range(100, 116)))
    device = torch.device("cpu")
    for _ in range(80):
        actions, *_ = policy.act(
            obs_to_torch(obs, device), obs_to_torch(masks, device), deterministic=True
        )
        obs, masks, _r, _d, _infos = env.step(actions_to_numpy(actions))
