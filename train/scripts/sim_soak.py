"""M0 soak + throughput gate for the Rust vec-env.

Random-masked-agent soak: N envs, strict validation ON, counts steps and
invalid-action errors (target: literal zero over >=5M env-steps), reports
episode stats and steps/s.

    python scripts/sim_soak.py --envs 256 --total-steps 5000000
    python scripts/sim_soak.py --bench          # 256/1024/2048 sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from balatro_train import encoding as E  # noqa: E402
from balatro_train.sim_env import SimBalatroEnv  # noqa: E402
from conftest import random_legal_actions  # noqa: E402


def vectorized_random_actions(rng, masks):
    """Fast vectorized random-legal-action sampler (same semantics as
    conftest.random_legal_actions, ~10x faster for big N)."""
    at_mask = masks["action_type_mask"]
    n = at_mask.shape[0]
    # Gumbel-max over legal types.
    g = rng.random(at_mask.shape)
    g[~at_mask] = -1.0
    action_type = g.argmax(axis=1)

    actions = {
        k: np.full((n, *shape), -1, dtype=dt) for k, (shape, dt) in E.ACTION_SPEC.items()
    }
    actions["action_type"] = action_type.astype(np.int64)
    actions["n_cards"] = np.zeros(n, dtype=np.int64)

    def pick(mask):
        g = rng.random(mask.shape)
        g[~mask] = -1.0
        return g.argmax(axis=1).astype(np.int64)

    for field, mask_key, needs in (
        ("joker_target", "joker_target_mask", (E.ActionType.SELL_JOKER,)),
        ("consumable_target", "consumable_target_mask",
         (E.ActionType.USE_CONSUMABLE, E.ActionType.SELL_CONSUMABLE)),
        ("shop_target", "shop_target_mask", (E.ActionType.BUY_SHOP,)),
        ("pack_target", "pack_target_mask", (E.ActionType.PICK_PACK,)),
    ):
        needed = np.isin(action_type, [int(t) for t in needs])
        if needed.any():
            actions[field][needed] = pick(masks[mask_key])[needed]

    card_types = np.isin(action_type, [int(t) for t in E.ACTION_NEEDS_CARDS])
    if card_types.any():
        cmask = masks["card_select_mask"]
        for i in np.flatnonzero(card_types):
            avail = np.flatnonzero(cmask[i])
            lo = E.CARD_PICK_MIN[E.ActionType(int(action_type[i]))]
            hi = min(E.MAX_CARD_PICKS, avail.size)
            k = int(rng.integers(lo, hi + 1)) if hi >= lo else 0
            if k:
                picks = rng.choice(avail, size=k, replace=False)
                actions["cards"][i, :k] = picks
            actions["n_cards"][i] = k
    return actions


def soak(num_envs: int, total_steps: int, seed: int, use_bot: bool) -> dict:
    rng = np.random.default_rng(seed)
    env = SimBalatroEnv(num_envs=num_envs, strict=True)
    obs, masks = env.reset([seed * 1_000_000 + i for i in range(num_envs)])
    steps = 0
    episodes = 0
    wins = 0
    antes: list[int] = []
    invalid = 0
    t0 = time.perf_counter()
    while steps < total_steps:
        if use_bot:
            actions = env.bot_actions()
        else:
            actions = vectorized_random_actions(rng, masks)
        try:
            obs, masks, _r, done, infos = env.step(actions)
        except ValueError as e:  # pragma: no cover - the gate is zero of these
            invalid += 1
            print(f"INVALID ACTION ERROR at step {steps}: {e}", file=sys.stderr)
            obs, masks = env.observe()
            continue
        steps += num_envs
        for i in np.flatnonzero(done):
            ep = infos[int(i)]["episode"]
            episodes += 1
            wins += bool(ep["won"])
            antes.append(ep["ante"])
    dt = time.perf_counter() - t0
    return {
        "num_envs": num_envs,
        "steps": steps,
        "seconds": dt,
        "sps": steps / dt,
        "episodes": episodes,
        "wins": wins,
        "mean_ante": float(np.mean(antes)) if antes else 0.0,
        "invalid": invalid,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bot", action="store_true", help="drive with the heuristic bot")
    p.add_argument("--bench", action="store_true",
                   help="short throughput sweep at N=256/1024/2048")
    args = p.parse_args()

    if args.bench:
        for n in (256, 1024, 2048):
            r = soak(n, max(n * 400, 200_000), args.seed, args.bot)
            print(f"N={n:5d}  {r['sps']:>10.0f} steps/s   "
                  f"({r['steps']} steps in {r['seconds']:.1f}s, "
                  f"{r['episodes']} episodes, mean ante {r['mean_ante']:.2f})")
        return

    r = soak(args.envs, args.total_steps, args.seed, args.bot)
    print(f"envs={r['num_envs']} steps={r['steps']} time={r['seconds']:.1f}s "
          f"sps={r['sps']:.0f}")
    print(f"episodes={r['episodes']} wins={r['wins']} mean_ante={r['mean_ante']:.3f}")
    print(f"invalid_action_errors={r['invalid']}")
    if r["invalid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
