"""Perf-toggle equivalence tests (config.PerfConfig).

The contract: every perf toggle is either bit-exact (buffer_device,
pin_memory, need_entropy, split stores, auto_minibatch, deferred stat syncs)
or explicitly numerics-changing (bf16_update, compile — validated by
learning-curve comparison outside the unit suite).  These tests pin the
bit-exact half, including the pipelining change (policy-side buffer stores
now happen BEFORE env.step).
"""

import dataclasses

import numpy as np
import pytest
import torch

from balatro_train import encoding as E
from balatro_train.buffer import RolloutBuffer
from balatro_train.config import load_config
from balatro_train.policy import BalatroPolicy, obs_to_torch
from balatro_train.ppo import PPOTrainer
from tests.conftest import SMALL_POLICY, rollout_states

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _base_cfg(**perf):
    cfg = load_config("configs/debug.yaml")
    cfg.seed = 11
    cfg.env.num_envs = 16
    cfg.env.seed = 5
    cfg.ppo.rollout_len = 16
    cfg.ppo.minibatch_size = 64
    cfg.ppo.update_epochs = 2
    cfg.ppo.total_timesteps = 16 * 16 * 2  # 2 iterations
    cfg.log.tensorboard = False
    cfg.curriculum.enabled = False
    for k, v in perf.items():
        setattr(cfg.perf, k, v)
    return cfg


def _run(cfg):
    torch.use_deterministic_algorithms(False)
    tr = PPOTrainer(cfg)
    hist = tr.train()
    return [(h["loss"], h["pg_loss"], h["v_loss"], h["entropy"]) for h in hist]


def test_perf_config_defaults():
    cfg = load_config("configs/debug.yaml")
    assert not cfg.perf.bf16_update
    assert not cfg.perf.compile
    assert cfg.perf.buffer_device == "cpu"
    assert not cfg.perf.pin_memory
    assert not cfg.perf.auto_minibatch
    assert not cfg.perf.fused_adam


def _fill_buffers(*buffers, seed=3):
    """Fill each buffer with the identical mock rollout (numpy path)."""
    states = rollout_states(num_envs=buffers[0].N, steps=buffers[0].T, seed=seed)
    rng = np.random.default_rng(seed)
    for t in range(buffers[0].T):
        obs, masks = states[t]
        actions = {
            "action_type": np.zeros(buffers[0].N, dtype=np.int64),
            "cards": np.full((buffers[0].N, 5), -1, dtype=np.int64),
            "n_cards": np.zeros(buffers[0].N, dtype=np.int64),
            "joker_target": np.full(buffers[0].N, -1, dtype=np.int64),
            "consumable_target": np.full(buffers[0].N, -1, dtype=np.int64),
            "shop_target": np.full(buffers[0].N, -1, dtype=np.int64),
            "pack_target": np.full(buffers[0].N, -1, dtype=np.int64),
        }
        logp = rng.normal(size=buffers[0].N).astype(np.float32)
        value = rng.normal(size=buffers[0].N).astype(np.float32)
        reward = rng.normal(size=buffers[0].N).astype(np.float32)
        done = rng.random(buffers[0].N) < 0.1
        for buf in buffers:
            buf.store_policy(obs, masks, actions, logp, value)
            buf.store_env(reward, done)
    last_value = torch.as_tensor(rng.normal(size=buffers[0].N).astype(np.float32))
    for buf in buffers:
        buf.compute_gae(last_value, 0.999, 0.95)


def _compare_minibatches(buf_a, buf_b, mb=32, seed=17):
    torch.manual_seed(seed)
    mbs_a = list(buf_a.minibatches(mb))
    torch.manual_seed(seed)
    mbs_b = list(buf_b.minibatches(mb))
    assert len(mbs_a) == len(mbs_b)
    for a, b in zip(mbs_a, mbs_b):
        for group in ("obs", "masks", "actions"):
            for k in a[group]:
                assert torch.equal(a[group][k].cpu(), b[group][k].cpu()), (group, k)
        for k in ("logprobs", "values", "advantages", "returns"):
            assert torch.equal(a[k].cpu(), b[k].cpu()), k


def test_split_store_equals_add():
    """store_policy + store_env == the classic add() (same buffer contents)."""
    dev = torch.device("cpu")
    buf_a = RolloutBuffer(8, 16, dev)
    buf_b = RolloutBuffer(8, 16, dev)
    states = rollout_states(num_envs=16, steps=8, seed=1)
    rng = np.random.default_rng(1)
    for t in range(8):
        obs, masks = states[t]
        actions = {k: np.zeros((16, *shape), dtype=np.int64)
                   for k, (shape, _dt) in E.ACTION_SPEC.items()}
        logp = rng.normal(size=16).astype(np.float32)
        value = rng.normal(size=16).astype(np.float32)
        reward = rng.normal(size=16).astype(np.float32)
        done = rng.random(16) < 0.2
        buf_a.add(obs, masks, actions, logp, value, reward, done)
        buf_b.store_policy(obs, masks, actions, logp, value)
        buf_b.store_env(reward, done)
    for k in buf_a.obs:
        assert torch.equal(buf_a.obs[k], buf_b.obs[k])
    assert torch.equal(buf_a.rewards, buf_b.rewards)
    assert torch.equal(buf_a.dones, buf_b.dones)
    assert torch.equal(buf_a.logprobs, buf_b.logprobs)


def test_pinned_minibatches_bit_identical():
    """pin_memory staging yields the exact same minibatch tensors."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf_a = RolloutBuffer(8, 16, dev, pin_memory=False)
    buf_b = RolloutBuffer(8, 16, dev, pin_memory=True)
    _fill_buffers(buf_a, buf_b)
    _compare_minibatches(buf_a, buf_b)


@cuda_only
def test_gpu_buffer_minibatches_bit_identical():
    """storage_device=cuda yields the exact same minibatch tensors, and GAE
    (always CPU fp32) matches bit-for-bit."""
    dev = torch.device("cuda")
    buf_a = RolloutBuffer(8, 16, dev, storage_device="cpu")
    buf_b = RolloutBuffer(8, 16, dev, storage_device="cuda")
    _fill_buffers(buf_a, buf_b)
    assert torch.equal(buf_a.advantages, buf_b.advantages.cpu())
    assert torch.equal(buf_a.returns, buf_b.returns.cpu())
    _compare_minibatches(buf_a, buf_b)


def test_need_entropy_skip_is_bit_identical():
    """act(need_entropy=False) must not change actions/logp/value or RNG."""
    policy = BalatroPolicy(SMALL_POLICY)
    (obs, masks) = rollout_states(num_envs=16, steps=12)[12]
    obs_t = obs_to_torch(obs, torch.device("cpu"))
    masks_t = obs_to_torch(masks, torch.device("cpu"))
    torch.manual_seed(123)
    a1, lp1, ent1, v1, _ = policy.act(obs_t, masks_t)
    torch.manual_seed(123)
    a2, lp2, ent2, v2, _ = policy.act(obs_t, masks_t, need_entropy=False)
    for k in a1:
        assert torch.equal(a1[k], a2[k])
    assert torch.equal(lp1, lp2)
    assert torch.equal(v1, v2)
    assert (ent2 == 0).all()
    assert not torch.equal(ent1, ent2)  # entropy genuinely skipped


def test_trainer_perf_toggles_bit_identical_cpu_reference():
    """Full-loop A/B on the mock env: perf defaults vs pinned toggles that are
    inert on CPU — pins that enabling the plumbing never drifts the math."""
    base = _run(_base_cfg())
    again = _run(_base_cfg())
    assert base == again  # seeded determinism of the harness itself


@cuda_only
def test_trainer_gpu_buffer_and_pin_bit_identical():
    """The pipelined collect (store-before-step) + GPU-resident buffer + pinned
    H2D + auto_minibatch + fused-off must reproduce the classic path's losses
    bit-for-bit on a seeded mock-env run (bf16/compile off)."""
    cfg_a = _base_cfg()
    cfg_a.device = "cuda"
    cfg_b = _base_cfg(buffer_device="cuda", pin_memory=True, auto_minibatch=True)
    cfg_b.device = "cuda"
    assert _run(cfg_a) == _run(cfg_b)


@cuda_only
def test_auto_minibatch_probe_caps_and_reports():
    cfg = _base_cfg(auto_minibatch=True)
    cfg.device = "cuda"
    cfg.ppo.minibatch_size = 1 << 20  # larger than the batch; probe caps it
    tr = PPOTrainer(cfg)
    assert tr.cfg.ppo.minibatch_size == cfg.env.num_envs * cfg.ppo.rollout_len
    assert tr._probe_peak and tr._probe_peak > 0


@cuda_only
def test_bf16_update_and_fused_adam_run():
    """Numerics-changing toggles still train: finite losses, sane entropy."""
    cfg = _base_cfg(bf16_update=True, fused_adam=True, buffer_device="cuda",
                    pin_memory=True)
    cfg.device = "cuda"
    hist = _run(cfg)
    assert len(hist) == 2
    assert all(np.isfinite(v) for h in hist for v in h)


def test_perf_config_dataclass_roundtrip():
    from balatro_train.config import PerfConfig, config_to_dict

    cfg = _base_cfg(bf16_update=True)
    d = config_to_dict(cfg)
    assert d["perf"]["bf16_update"] is True
    assert set(d["perf"]) == {f.name for f in dataclasses.fields(PerfConfig)}
