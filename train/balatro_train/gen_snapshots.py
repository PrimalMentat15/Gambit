"""Generate a snapshot start-state pool (M3 backward curriculum).

Drives N parallel Rust sim envs with the built-in heuristic bot — and/or a
policy checkpoint — and harvests full env snapshots at trigger points:
ENTERING the shop, or blind select, at antes ``--antes`` (default 3-7).
Snapshots are deduplicated per trigger (run seed, ante, round, state) and
collected under stratified per-ante quotas (default: uniform over the ante
range, ``--total`` 50k overall).  Bot-only pools will skew toward antes 3-4
(the bot rarely passes ante 5): generation stops on quotas met OR budget
exhausted, reports the actual per-ante histogram, and writes whatever was
collected — per-ante shortfalls are accepted, not blocking.

Output: a packed pool file (:mod:`balatro_train.snapshot_pool` format,
loaded Rust-side via ``load_snapshot_pool``) plus a sidecar
``<out>.manifest.json`` with counts and generator provenance.

Usage (from train/):
  python -m balatro_train.gen_snapshots --out runs/pools/pool_v1.bin
  # mix in a policy checkpoint for deeper-ante coverage:
  python -m balatro_train.gen_snapshots --out runs/pools/pool_v2.bin \
      --ckpt runs/m1/ckpt_XXXX.pt --policy-frac 0.5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from balatro_train import snapshot_pool

HARVEST_STATES = ("SHOP", "BLIND_SELECT")


def _load_policy(ckpt_path: str, device):
    """Policy + its config dims from a ppo.py checkpoint."""
    import torch

    from balatro_train.config import TrainConfig, _build
    from balatro_train.policy import BalatroPolicy

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pcfg = _build(TrainConfig, ckpt["config"], "ckpt").policy
    policy = BalatroPolicy(pcfg).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, int(ckpt.get("global_step", -1))


def generate(
    out: str,
    total: int = 50_000,
    ante_lo: int = 3,
    ante_hi: int = 7,
    num_envs: int = 256,
    seed_base: int = 5_000_000,
    ckpt: str | None = None,
    policy_frac: float = 0.5,
    device: str = "cpu",
    max_steps: int = 200_000,
    max_seconds: float = 3600.0,
    log_every: int = 200,
) -> dict:
    """Run the harvest; returns the manifest dict (also written to disk)."""
    from balatro_train.sim_env import SimBalatroEnv

    if not 1 <= ante_lo <= ante_hi <= 8:
        raise ValueError(f"bad ante range {ante_lo}-{ante_hi}")
    antes = list(range(ante_lo, ante_hi + 1))
    quota = {a: total // len(antes) for a in antes}
    for a in antes[: total % len(antes)]:
        quota[a] += 1

    policy = None
    policy_step = -1
    n_policy_envs = 0
    if ckpt:
        import torch

        device_t = torch.device(device)
        policy, policy_step = _load_policy(ckpt, device_t)
        n_policy_envs = int(round(num_envs * policy_frac))
        print(f"[gen] policy {ckpt} (step {policy_step}) drives "
              f"{n_policy_envs}/{num_envs} envs; bot drives the rest")

    # win_ante 8: generator runs go as deep as the driver can take them.
    env = SimBalatroEnv(num_envs, win_ante=8)
    # Seed block disjoint from training (cfg.env.seed + [0, N)) and from
    # held-out eval seeds (1_000_000 + [0, N)).
    obs, masks = env.reset([seed_base + i for i in range(num_envs)])

    entries: list[tuple[int, bytes]] = []
    counts = {a: 0 for a in antes}
    state_counts = {s: 0 for s in HARVEST_STATES}
    seen: set[tuple] = set()
    prev_state = [env.run_info(i)["state"] for i in range(num_envs)]

    def quotas_met() -> bool:
        return all(counts[a] >= quota[a] for a in antes)

    t0 = time.time()
    steps = 0
    while steps < max_steps and time.time() - t0 < max_seconds and not quotas_met():
        if policy is not None and n_policy_envs > 0:
            import torch

            from balatro_train.policy import actions_to_numpy, obs_to_torch

            with torch.no_grad():
                pol_actions, *_ = policy.act(
                    obs_to_torch(obs, device_t), obs_to_torch(masks, device_t)
                )
            actions = actions_to_numpy(pol_actions)
            bot = env.bot_actions()
            for k in actions:
                actions[k][n_policy_envs:] = bot[k][n_policy_envs:]
        else:
            actions = env.bot_actions()

        obs, masks, _r, _done, _infos = env.step(actions)
        steps += 1

        for i in range(num_envs):
            info = env.run_info(i)
            state = info["state"]
            if state != prev_state[i] and state in HARVEST_STATES:
                ante = int(info["ante"])
                if ante_lo <= ante <= ante_hi and counts[ante] < quota[ante]:
                    # One snapshot per trigger point: shop entry and blind
                    # select of the same round are distinct starts, but a
                    # shop RE-entry (after a pack) is not.
                    key = (info["seed"], ante, int(info["round"]), state)
                    if key not in seen:
                        seen.add(key)
                        entries.append((ante, env.snapshot(i)))
                        counts[ante] += 1
                        state_counts[state] += 1
            prev_state[i] = state

        if steps % log_every == 0:
            hist = "  ".join(f"a{a}:{counts[a]}/{quota[a]}" for a in antes)
            print(f"[gen] step {steps}  {len(entries)}/{total}  {hist}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    if not entries:
        raise RuntimeError("harvested no snapshots — check the ante range/budget")
    snapshot_pool.write_pool(out, entries)

    manifest = {
        "pool_format_version": snapshot_pool.VERSION,
        "total": len(entries),
        "target_total": total,
        "counts_per_ante": {str(a): counts[a] for a in antes},
        "target_per_ante": {str(a): quota[a] for a in antes},
        "counts_per_state": state_counts,
        "generator": {
            "bot_envs": num_envs - n_policy_envs,
            "policy_envs": n_policy_envs,
            "ckpt": ckpt,
            "ckpt_global_step": policy_step,
            "num_envs": num_envs,
            "seed_base": seed_base,
            "win_ante": 8,
        },
        "steps": steps,
        "elapsed_seconds": round(elapsed, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    mpath = snapshot_pool.write_manifest(out, manifest)
    hist = "  ".join(f"a{a}:{counts[a]}/{quota[a]}" for a in antes)
    print(f"[gen] wrote {len(entries)} snapshots -> {out} (+ {mpath})\n"
          f"[gen] per-ante: {hist}  in {elapsed:.0f}s / {steps} steps", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="pool file to write")
    parser.add_argument("--total", type=int, default=50_000)
    parser.add_argument("--antes", default="3-7", help="inclusive ante range, e.g. 3-7")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--ckpt", default=None,
                        help="optional policy checkpoint to co-drive envs")
    parser.add_argument("--policy-frac", type=float, default=0.5,
                        help="fraction of envs the policy drives (with --ckpt)")
    parser.add_argument("--device", default="cpu", help="policy device")
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    lo, _, hi = args.antes.partition("-")
    ante_lo, ante_hi = int(lo), int(hi or lo)
    generate(
        out=args.out,
        total=args.total,
        ante_lo=ante_lo,
        ante_hi=ante_hi,
        num_envs=args.num_envs,
        seed_base=args.seed_base,
        ckpt=args.ckpt,
        policy_frac=args.policy_frac,
        device=args.device,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
