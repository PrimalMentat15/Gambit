"""Wall-clock phase breakdown + torch.profiler benchmark of the PPO loop.

Measures, per training iteration at the config's num_envs / rollout_len:

* collect: obs->tensor H2D, policy.act, actions->numpy D2H, env.step,
  info bookkeeping + return-norm, buffer.add, GAE.
* update: minibatch fetch (CPU gather + H2D), forward+loss, backward,
  optimizer step — per epoch x minibatch, at a configurable minibatch size
  (so the update path can be profiled at reduced sizes when VRAM is shared
  with a live run; results include per-sample scaling to extrapolate).

Usage (from train/):
    python scripts/bench_ppo.py --config configs/m3.yaml \
        --ckpt /path/to/ckpt_copy.pt --iters 2 [--mb 4096] [--torch-profile]

Timing uses torch.cuda.synchronize() at section boundaries, which serializes
async work — end-to-end steps/s from this script is a *lower bound*; use
--no-sync for an uninstrumented end-to-end number.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from balatro_train.config import load_config
from balatro_train.policy import actions_to_numpy, obs_to_torch
from balatro_train.ppo import PPOTrainer


class Timers:
    def __init__(self, sync: bool):
        self.sync = sync
        self.acc: dict[str, float] = defaultdict(float)
        self._t0 = None

    def start(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self, name: str):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.acc[name] += time.perf_counter() - self._t0
        self._t0 = time.perf_counter()


def timed_collect(tr: PPOTrainer, tm: Timers) -> None:
    """PPOTrainer.collect() with per-section timers (kept in sync manually)."""
    gpu_buffer = tr.buffer.storage_device.type == "cuda"
    tr.buffer.reset()
    for _ in range(tr.cfg.ppo.rollout_len):
        tm.start()
        obs_t = tr._to_device(tr.obs)
        masks_t = tr._to_device(tr.masks)
        tm.stop("collect/obs_to_tensor")
        with tr._autocast():
            actions_t, logprob, _ent, value, _aux = tr.policy.act(
                obs_t, masks_t, need_entropy=False)
        tm.stop("collect/policy_act")
        actions_np = actions_to_numpy(actions_t)
        tm.stop("collect/actions_to_numpy")
        if gpu_buffer:
            tr.buffer.store_policy(obs_t, masks_t, actions_t,
                                   logprob.float(), value.float())
        else:
            tr.buffer.store_policy(tr.obs, tr.masks, actions_np,
                                   logprob.float(), value.float())
        tm.stop("collect/buffer_store")
        next_obs, next_masks, reward, done, infos = tr.env.step(actions_np)
        tm.stop("collect/env_step")
        tr.global_step += tr.env.num_envs
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                tr.ep_returns.append(ep["r"])
        stored_reward = tr.ret_norm(reward, done) if tr.ret_norm else reward
        tr.buffer.store_env(stored_reward, done)
        tm.stop("collect/info_retnorm")
        tr.obs, tr.masks = next_obs, next_masks
    tm.start()
    with torch.no_grad(), tr._autocast():
        last_value = tr.policy.get_value(
            tr._to_device(tr.obs), tr._to_device(tr.masks))
    tr.buffer.compute_gae(last_value.float(), tr.cfg.ppo.gamma, tr.cfg.ppo.gae_lambda)
    tm.stop("collect/gae")


def timed_update(tr: PPOTrainer, tm: Timers, minibatch_size: int) -> None:
    cfg = tr.cfg.ppo
    for _ in range(cfg.update_epochs):
        it = tr.buffer.minibatches(minibatch_size)
        while True:
            tm.start()
            try:
                mb = next(it)
            except StopIteration:
                break
            tm.stop("update/mb_fetch")
            with tr._update_autocast():
                new_logprob, entropy, new_value, _aux = tr.policy.evaluate_actions(
                    mb["obs"], mb["masks"], mb["actions"])
            new_logprob, entropy, new_value = (
                new_logprob.float(), entropy.float(), new_value.float())
            logratio = new_logprob - mb["logprobs"]
            ratio = logratio.exp()
            adv = mb["advantages"]
            if cfg.norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = torch.max(
                -adv * ratio,
                -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
            ).mean()
            v_loss = 0.5 * (new_value - mb["returns"]).pow(2).mean()
            loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
            tm.stop("update/forward")
            tr.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            tm.stop("update/backward")
            torch.nn.utils.clip_grad_norm_(tr.policy.parameters(), cfg.max_grad_norm)
            tr.optimizer.step()
            tm.stop("update/optim")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None, help="checkpoint COPY to load")
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--mb", type=int, default=None, help="override minibatch size")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--no-sync", action="store_true",
                   help="skip cuda sync at boundaries (end-to-end mode)")
    p.add_argument("--skip-update", action="store_true")
    p.add_argument("--torch-profile", action="store_true",
                   help="torch.profiler over 3 collect steps + 2 update mbs")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VAL",
                   help="config override, e.g. --set perf.bf16_update=true")
    args = p.parse_args()

    cfg = load_config(args.config)
    for ov in args.set:
        path, _, raw = ov.partition("=")
        import json as _json

        try:
            val = _json.loads(raw)
        except ValueError:
            val = raw
        obj = cfg
        *parts, last = path.split(".")
        for part in parts:
            obj = getattr(obj, part)
        setattr(obj, last, val)
    cfg.log.tensorboard = False
    cfg.log.wandb = False
    cfg.curriculum.enabled = True  # win_ante comes from the checkpoint
    if args.num_envs:
        cfg.env.num_envs = args.num_envs

    tr = PPOTrainer(cfg)
    # after init: perf.auto_minibatch may have shrunk cfg.ppo.minibatch_size
    mb = args.mb or cfg.ppo.minibatch_size
    if args.ckpt:
        tr.load_checkpoint(args.ckpt)
        print(f"loaded {args.ckpt} (win_ante {tr.win_ante})")
    n_params = tr.policy.num_params()
    batch = cfg.env.num_envs * cfg.ppo.rollout_len
    print(f"params {n_params:,}  N={cfg.env.num_envs} T={cfg.ppo.rollout_len} "
          f"batch={batch}  mb={mb}  bf16_rollout={cfg.bf16_rollout}")

    dev = tr.device
    torch.cuda.reset_peak_memory_stats(dev)

    # warmup
    for _ in range(args.warmup_iters):
        wtm = Timers(sync=False)
        timed_collect(tr, wtm)
        if not args.skip_update:
            timed_update(tr, wtm, mb)

    tm = Timers(sync=not args.no_sync)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        timed_collect(tr, tm)
        if not args.skip_update:
            timed_update(tr, tm, mb)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    steps = args.iters * batch
    if args.iters:
        print(f"\n== phase breakdown ({args.iters} iters, {steps} steps, "
              f"wall {wall:.2f}s, {steps / wall:,.0f} steps/s) ==")
        total = sum(tm.acc.values())
        for name, sec in sorted(tm.acc.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<28s} {sec / args.iters:8.3f} s/iter  "
                  f"{100 * sec / total:5.1f}%")
        print(f"  {'(sum)':<28s} {total / args.iters:8.3f} s/iter")
    print(f"peak VRAM (this process): "
          f"{torch.cuda.max_memory_allocated(dev) / 2**20:,.0f} MiB alloc / "
          f"{torch.cuda.max_memory_reserved(dev) / 2**20:,.0f} MiB reserved")

    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        print("\n== torch.profiler: 3 collect steps ==")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                obs_t = obs_to_torch(tr.obs, dev)
                masks_t = obs_to_torch(tr.masks, dev)
                with tr._autocast():
                    actions_t, *_ = tr.policy.act(obs_t, masks_t)
                actions_np = actions_to_numpy(actions_t)
                tr.obs, tr.masks, *_ = tr.env.step(actions_np)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15,
                                        max_name_column_width=60))
        if not args.skip_update:
            print("\n== torch.profiler: 2 update minibatches ==")
            it = tr.buffer.minibatches(mb)
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    b = next(it)
                    lp, ent, val, _ = tr.policy.evaluate_actions(
                        b["obs"], b["masks"], b["actions"])
                    loss = (lp - b["logprobs"]).pow(2).mean() + val.mean() + ent.mean()
                    tr.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    tr.optimizer.step()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15,
                                            max_name_column_width=60))


if __name__ == "__main__":
    main()
