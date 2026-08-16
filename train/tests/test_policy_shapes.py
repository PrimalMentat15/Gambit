"""Policy forward shapes/dtypes, incl. under bf16 autocast; param count sanity."""

import numpy as np
import torch

from balatro_train import encoding as E
from balatro_train.config import PolicyConfig
from balatro_train.mock_env import MockBalatroEnv
from balatro_train.policy import BalatroPolicy, obs_to_torch

from conftest import SMALL_POLICY


def _state(n=8):
    env = MockBalatroEnv(num_envs=n)
    obs, masks = env.reset(list(range(n)))
    dev = torch.device("cpu")
    return obs_to_torch(obs, dev), obs_to_torch(masks, dev)


def test_forward_shapes_and_dtypes():
    torch.manual_seed(0)
    policy = BalatroPolicy(SMALL_POLICY)
    n = 8
    obs_t, masks_t = _state(n)
    actions, logp, ent, value, _ = policy.act(obs_t, masks_t)
    for k, (shape, _dt) in E.ACTION_SPEC.items():
        assert actions[k].shape == (n, *shape), k
        assert actions[k].dtype == torch.int64, k
    assert logp.shape == (n,) and ent.shape == (n,) and value.shape == (n,)
    assert logp.dtype == torch.float32 and value.dtype == torch.float32

    logp2, ent2, value2, _ = policy.evaluate_actions(obs_t, masks_t, actions)
    assert logp2.shape == (n,) and ent2.shape == (n,) and value2.shape == (n,)
    assert logp2.requires_grad and value2.requires_grad
    v = policy.get_value(obs_t, masks_t)
    assert v.shape == (n,)


def test_forward_under_bf16_autocast():
    """The rollout inference path must run under bf16 autocast (CPU flavor
    here; CUDA uses the same torch.autocast API)."""
    torch.manual_seed(0)
    policy = BalatroPolicy(SMALL_POLICY)
    n = 8
    obs_t, masks_t = _state(n)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actions, logp, ent, value, _ = policy.act(obs_t, masks_t)
    assert torch.isfinite(logp.float()).all()
    assert torch.isfinite(value.float()).all()
    assert (ent.float() >= -1e-3).all()
    # sampled actions remain exactly legal under reduced precision
    at = actions["action_type"].numpy()
    assert masks_t["action_type_mask"].numpy()[np.arange(n), at].all()
    for k in E.ACTION_SPEC:
        assert actions[k].dtype == torch.int64


def test_default_policy_param_count_in_budget():
    policy = BalatroPolicy(PolicyConfig())
    n = policy.num_params()
    # plan budget: ~1.5-2.5M, deliberately small
    assert 1_000_000 < n < 3_000_000, n
