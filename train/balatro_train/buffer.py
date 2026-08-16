"""Rollout buffer (dict obs + masks + composite actions), GAE, normalizers.

The buffer stores, per transition: the full observation dict, ALL mask
arrays, the composite action dict, the joint logprob, value, reward, done.
Masks are stored verbatim and replayed to `evaluate_actions` during the PPO
epochs — they are never recomputed from obs (the classic silent masking bug,
M0-gated).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from balatro_train import encoding as E

_TORCH_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
}


class RolloutBuffer:
    """Fixed-size (T, N) storage; minibatches are moved to ``device`` lazily.

    ``storage_device`` selects where obs/masks/actions/logprobs/values (and,
    post-GAE, advantages/returns) live:

    * ``"cpu"`` (default) — classic layout; with ``pin_memory=True`` the big
      dict tensors are pinned and minibatch fetches stage through pinned
      memory with ``non_blocking`` H2D copies.
    * ``"cuda"`` — GPU-resident: per-step stores become device-side copies
      (queued async, so they overlap the CPU env step) and minibatch fetch
      is a pure GPU gather — no per-epoch H2D of the whole batch.

    rewards/dones always stay on CPU (they arrive as numpy from the env) and
    GAE is always computed on CPU in fp32, so stored advantages/returns are
    bit-identical across storage devices.
    """

    def __init__(self, rollout_len: int, num_envs: int, device: torch.device,
                 storage_device: str | torch.device | None = None,
                 pin_memory: bool = False):
        self.T = rollout_len
        self.N = num_envs
        self.device = device
        self.storage_device = torch.device(storage_device or "cpu")
        self.pin = bool(
            pin_memory
            and self.storage_device.type == "cpu"
            and torch.cuda.is_available()
        )
        alloc = lambda spec: {
            k: torch.zeros((self.T, self.N, *shape), dtype=_TORCH_DTYPE[dt],
                           device=self.storage_device, pin_memory=self.pin)
            for k, (shape, dt) in spec.items()
        }
        self.obs = alloc(E.OBS_SPEC)
        self.masks = alloc(E.MASK_SPEC)
        self.actions = alloc(E.ACTION_SPEC)
        self.logprobs = torch.zeros(self.T, self.N, device=self.storage_device)
        self.values = torch.zeros(self.T, self.N, device=self.storage_device)
        self.rewards = torch.zeros(self.T, self.N)
        self.dones = torch.zeros(self.T, self.N, dtype=torch.bool)
        self.advantages = torch.zeros(self.T, self.N)
        self.returns = torch.zeros(self.T, self.N)
        self._stage: dict[tuple, torch.Tensor] = {}  # pinned minibatch staging
        self.step = 0

    def nbytes(self) -> int:
        """Bytes of the big storage (obs + masks + actions + logp/values)."""
        dicts = (*self.obs.values(), *self.masks.values(), *self.actions.values())
        return sum(t.numel() * t.element_size() for t in dicts) + \
            2 * self.logprobs.numel() * self.logprobs.element_size()

    def store_policy(self, obs, masks, actions, logprob, value) -> None:
        """Store the policy-side half of step ``self.step``: obs/masks the
        policy saw, the sampled actions, joint logprob and value.  Accepts
        numpy dicts or torch dicts (either device); with cuda storage pass
        the device tensors so the copies stay on-GPU."""
        t = self.step
        assert t < self.T, "buffer full"
        for store, batch in ((self.obs, obs), (self.masks, masks), (self.actions, actions)):
            for k, v in batch.items():
                store[k][t].copy_(torch.as_tensor(v))
        self.logprobs[t].copy_(torch.as_tensor(logprob).float())
        self.values[t].copy_(torch.as_tensor(value).float())

    def store_env(self, reward, done) -> None:
        """Store the env-side half (reward, done) and advance the step."""
        t = self.step
        assert t < self.T, "buffer full"
        self.rewards[t] = torch.as_tensor(reward).float().cpu()
        self.dones[t] = torch.as_tensor(done).cpu()
        self.step = t + 1

    def add(self, obs, masks, actions, logprob, value, reward, done) -> None:
        """Store one vec-step. obs/masks numpy dicts; the rest torch/numpy (N,)."""
        self.store_policy(obs, masks, actions, logprob, value)
        self.store_env(reward, done)

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE(gamma, lam). ``dones[t]`` = episode ended ON step t (the stored
        reward is terminal; the successor state is a fresh episode, so its
        value is not bootstrapped).  Always computed on CPU in fp32 (values
        are pulled from storage in one copy), so results are bit-identical
        regardless of ``storage_device``."""
        assert self.step == self.T, "buffer not full"
        last_value = last_value.float().cpu()
        values = self.values.cpu()
        not_done = (~self.dones).float()
        adv = torch.zeros(self.N)
        advantages = torch.zeros(self.T, self.N)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else values[t + 1]
            delta = self.rewards[t] + gamma * next_value * not_done[t] - values[t]
            adv = delta + gamma * lam * not_done[t] * adv
            advantages[t] = adv
        self.advantages = advantages.to(self.storage_device)
        self.returns = (advantages + values).to(self.storage_device)

    def _fetch(self, flat: torch.Tensor, idx: torch.Tensor, key: tuple) -> torch.Tensor:
        """Gather rows ``idx`` of a flattened store and move them to device.
        ``idx`` must already live on the store's device."""
        if flat.device.type != "cpu":  # GPU-resident: pure device gather
            return flat[idx]
        if not self.pin:
            return flat[idx].to(self.device)
        # pinned staging -> async H2D (the consumer ops sync via the stream)
        stage = self._stage.get(key)
        if stage is None or stage.shape[0] < idx.shape[0]:
            stage = torch.empty((idx.shape[0], *flat.shape[1:]), dtype=flat.dtype,
                                pin_memory=True)
            self._stage[key] = stage
        out = stage[: idx.shape[0]]
        torch.index_select(flat, 0, idx, out=out)
        return out.to(self.device, non_blocking=True)

    def minibatches(
        self, minibatch_size: int, generator: torch.Generator | None = None
    ) -> Iterator[dict]:
        """Yield shuffled flat minibatches as dicts of device tensors."""
        total = self.T * self.N
        perm = torch.randperm(total, generator=generator)
        flat = lambda x: x.reshape(total, *x.shape[2:])
        stage_free = None  # completion event of the previous fetch's H2D copies
        for start in range(0, total, minibatch_size):
            if stage_free is not None:
                # pinned staging is reused per fetch: wait until the previous
                # async H2D copies have drained before overwriting the stage.
                stage_free.synchronize()
            idx = perm[start : start + minibatch_size]
            if self.storage_device.type != "cpu":
                idx = idx.to(self.storage_device)  # one H2D per minibatch
            to_dev = lambda x, key: self._fetch(flat(x), idx, key)
            mb = {
                "obs": {k: to_dev(v, ("obs", k)) for k, v in self.obs.items()},
                "masks": {k: to_dev(v, ("masks", k)) for k, v in self.masks.items()},
                "actions": {k: to_dev(v, ("act", k)) for k, v in self.actions.items()},
                "logprobs": to_dev(self.logprobs, ("logprobs",)),
                "values": to_dev(self.values, ("values",)),
                "advantages": to_dev(self.advantages, ("advantages",)),
                "returns": to_dev(self.returns, ("returns",)),
            }
            if self.pin and self.device.type == "cuda":
                stage_free = torch.cuda.Event()
                stage_free.record()  # marks just the H2D copies (pre-yield)
            yield mb

    def reset(self) -> None:
        self.step = 0


class RunningMeanStd:
    """Numerically stable running mean/var (Welford, batched)."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        b_mean, b_var, b_count = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta**2 * self.count * b_count / tot) / tot
        self.count = tot

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


class ReturnNormalizer:
    """Running return normalization (CleanRL/gym ``NormalizeReward`` scheme).

    Tracks a per-env discounted return accumulator, keeps running stats of
    it, and scales rewards by 1/std(returns).  State is checkpointed.
    """

    def __init__(self, num_envs: int, gamma: float, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.accum = np.zeros(num_envs, dtype=np.float64)
        self.rms = RunningMeanStd()

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.accum = self.accum * self.gamma + rewards
        self.rms.update(self.accum)
        self.accum[dones] = 0.0
        return (rewards / np.sqrt(self.rms.var + self.epsilon)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"accum": self.accum.copy(), "rms": self.rms.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        # The per-env accumulator only transfers between equal-width runs;
        # when resuming with a different num_envs (e.g. a small-vec smoke
        # resume of a 2048-env checkpoint) keep the zeros — the envs were
        # freshly reset anyway. The running stats (what actually scales
        # rewards) always transfer.
        if state["accum"].shape == self.accum.shape:
            self.accum = state["accum"].copy()
        self.rms.load_state_dict(state["rms"])
