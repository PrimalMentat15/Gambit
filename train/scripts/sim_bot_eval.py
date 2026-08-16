"""Heuristic-bot baseline over N fresh seeds (default 1000): win rate and
mean ante reached. The bot's actions go through the SAME masked step path as
the learned policy (strict validation on).

    python scripts/sim_bot_eval.py --seeds 1000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from balatro_train.sim_env import SimBalatroEnv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=5000,
                   help="per-env step cap (an episode is far shorter)")
    args = p.parse_args()

    n = args.seeds
    env = SimBalatroEnv(num_envs=n, strict=True)
    env.reset([args.seed_base + i for i in range(n)])

    first_ep: dict[int, dict] = {}
    t0 = time.perf_counter()
    steps = 0
    while len(first_ep) < n and steps < args.max_steps:
        actions = env.bot_actions()
        _o, _m, _r, done, infos = env.step(actions)
        steps += 1
        for i in np.flatnonzero(done):
            i = int(i)
            if i not in first_ep:
                first_ep[i] = infos[i]["episode"]
    dt = time.perf_counter() - t0

    eps = list(first_ep.values())
    antes = np.array([e["ante"] for e in eps], dtype=np.float64)
    wins = np.array([e["won"] for e in eps], dtype=np.bool_)
    lens = np.array([e["l"] for e in eps], dtype=np.float64)
    rets = np.array([e["r"] for e in eps], dtype=np.float64)
    print(f"episodes={len(eps)}/{n} in {dt:.1f}s ({steps} vec-steps)")
    print(f"win_rate={wins.mean():.4f}  mean_ante={antes.mean():.3f}  "
          f"median_ante={np.median(antes):.1f}  max_ante={antes.max():.0f}")
    print(f"ante histogram: {np.bincount(antes.astype(int))[1:]}")
    print(f"mean_len={lens.mean():.1f}  mean_shaped_return={rets.mean():.3f}")


if __name__ == "__main__":
    main()
