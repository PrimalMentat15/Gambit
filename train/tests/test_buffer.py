"""Rollout buffer round-trip fidelity + hand-computed GAE."""

import numpy as np
import torch

from balatro_train import encoding as E
from balatro_train.buffer import ReturnNormalizer, RolloutBuffer, RunningMeanStd


def _random_batch(spec, rng, n):
    out = {}
    for k, (shape, dt) in spec.items():
        if dt == np.dtype(np.float32):
            out[k] = rng.standard_normal((n, *shape)).astype(np.float32)
        elif dt == np.dtype(np.int64):
            out[k] = rng.integers(-1, 10, size=(n, *shape)).astype(np.int64)
        else:
            out[k] = rng.random((n, *shape)) < 0.5
    return out


def test_round_trip_fidelity():
    T, N = 6, 5
    rng = np.random.default_rng(0)
    buf = RolloutBuffer(T, N, torch.device("cpu"))
    stored = []
    for t in range(T):
        obs = _random_batch(E.OBS_SPEC, rng, N)
        masks = _random_batch(E.MASK_SPEC, rng, N)
        actions = _random_batch(E.ACTION_SPEC, rng, N)
        # tag each transition with its unique flat index for identification
        flat = np.arange(t * N, (t + 1) * N, dtype=np.float32)
        logp = flat.copy()
        buf.add(obs, masks, actions, logp, rng.standard_normal(N).astype(np.float32),
                rng.standard_normal(N).astype(np.float32), rng.random(N) < 0.2)
        stored.append((obs, masks, actions))
    buf.compute_gae(torch.zeros(N), gamma=0.99, lam=0.95)

    seen = []
    for mb in buf.minibatches(minibatch_size=7):
        for row in range(mb["logprobs"].shape[0]):
            flat_idx = int(mb["logprobs"][row].item())
            t, n = divmod(flat_idx, N)
            seen.append(flat_idx)
            obs, masks, actions = stored[t]
            for k in E.OBS_SPEC:
                assert (mb["obs"][k][row].numpy() == obs[k][n]).all(), k
            for k in E.MASK_SPEC:
                assert (mb["masks"][k][row].numpy() == masks[k][n]).all(), k
            for k in E.ACTION_SPEC:
                assert (mb["actions"][k][row].numpy() == actions[k][n]).all(), k
    # every transition appears exactly once across minibatches
    assert sorted(seen) == list(range(T * N))


def test_gae_hand_computed():
    """gamma=0.5, lam=0.5, T=3: worked example (see comments)."""
    buf = RolloutBuffer(3, 1, torch.device("cpu"))
    obs = {k: np.zeros((1, *s), dtype=d) for k, (s, d) in E.OBS_SPEC.items()}
    masks = {k: np.zeros((1, *s), dtype=d) for k, (s, d) in E.MASK_SPEC.items()}
    acts = {k: np.zeros((1, *s), dtype=d) for k, (s, d) in E.ACTION_SPEC.items()}
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    dones = [False, True, False]
    for t in range(3):
        buf.add(obs, masks, acts, np.zeros(1, np.float32),
                np.array([values[t]], np.float32),
                np.array([rewards[t]], np.float32),
                np.array([dones[t]]))
    buf.compute_gae(torch.tensor([2.0]), gamma=0.5, lam=0.5)
    # t=2: delta = 3 + 0.5*2.0 - 1.5 = 2.5            -> adv2 = 2.5
    # t=1: done -> delta = 2 + 0 - 1.0 = 1.0           -> adv1 = 1.0
    # t=0: delta = 1 + 0.5*1.0 - 0.5 = 1.0
    #      adv0 = 1.0 + 0.5*0.5*adv1 = 1.25
    expect_adv = torch.tensor([[1.25], [1.0], [2.5]])
    assert torch.allclose(buf.advantages, expect_adv, atol=1e-6)
    assert torch.allclose(buf.returns, expect_adv + torch.tensor(values).unsqueeze(1))


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(1)
    rms = RunningMeanStd()
    chunks = [rng.standard_normal(50) * 3 + 2 for _ in range(10)]
    for c in chunks:
        rms.update(c)
    allx = np.concatenate(chunks)
    assert abs(rms.mean - allx.mean()) < 1e-6
    assert abs(rms.var - allx.var()) < 1e-4


def test_return_normalizer_state_roundtrip():
    rng = np.random.default_rng(2)
    norm = ReturnNormalizer(4, gamma=0.99)
    for _ in range(20):
        norm(rng.standard_normal(4).astype(np.float32), rng.random(4) < 0.1)
    state = norm.state_dict()
    norm2 = ReturnNormalizer(4, gamma=0.99)
    norm2.load_state_dict(state)
    r = rng.standard_normal(4).astype(np.float32)
    d = np.zeros(4, dtype=bool)
    assert (norm(r.copy(), d) == norm2(r.copy(), d)).all()
