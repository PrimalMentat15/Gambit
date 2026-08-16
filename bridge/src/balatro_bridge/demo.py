"""Live-game demo: the trained PPO policy plays real Balatro end-to-end.

The policy (argmax over the frozen RL env contract) picks every action from
the Rust sim's encoded state; each action is mirrored onto the live game via
balatrobot, and after every action the full game state is diffed against the
sim (the P5 cross-validation machinery), so the demo doubles as a fidelity
check. Divergences are reported but do not stop the show.

Prereqs:
    conda activate balatro    # see repo README
    balatro-bridge-launch          # game window + RPC (leave running)

Run:
    balatro-bridge-demo --ckpt ../train/runs/cloud_mirror/m5e2/ckpt_4000317440.pt

By default the demo pre-screens random seeds in the sim (a fraction of a
second each) until it finds one the policy wins, so the audience sees a full
ante-1-through-8 victory; pass --seed to play a specific seed instead, or
--no-screen to take the first random seed win-or-lose (the certified policy
wins ~71% of them).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from balatro_bridge.crossval import (
    GameDriver,
    SimDriver,
    make_ckpt_policy,
    make_seeds,
    run_one,
)


def screen_winning_seed(ckpt_path: str, device: str, tries: int = 40) -> str:
    """Play random seeds in the sim (argmax) until one wins ante 8."""
    policy = make_ckpt_policy(ckpt_path, device)
    rng = random.Random(0)
    for seed in make_seeds(tries):
        sim = SimDriver(seed)
        steps = 0
        while not sim.is_over() and steps < 800:
            if policy(sim, rng) is None:
                break
            steps += 1
        state = json.loads(sim.run.state_json())
        verdict = "WIN" if state["won"] else f"loss (ante {sim.ante()})"
        print(f"  screening {seed}: {verdict}", flush=True)
        if state["won"]:
            return seed
    print("no winning seed found in screening budget; using the last one")
    return seed


def narrating(policy):
    """Wrap a crossval policy to narrate each chosen action."""

    def wrapped(sim: SimDriver, rng: random.Random):
        state = sim.state()
        action = policy(sim, rng)
        if action is not None:
            desc = action["kind"]
            if "cards" in action:
                desc += f" {action['cards']}"
            print(
                f"    ante {state['ante_num']} ${state['money']:<3} "
                f"{state['state']:<17} -> {desc}",
                flush=True,
            )
        return action

    return wrapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="trained policy .pt")
    ap.add_argument("--seed", default=None, help="seed to play (default: screen for a win)")
    ap.add_argument("--no-screen", action="store_true",
                    help="with no --seed: play the first random seed unscreened")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12346)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    if args.seed:
        seed = args.seed.upper()
    elif args.no_screen:
        seed = make_seeds(1)[0]
    else:
        print("pre-screening seeds in the sim for a winning run...", flush=True)
        seed = screen_winning_seed(args.ckpt, args.device)

    game = GameDriver(args.host, args.port)
    game.client.health()

    repo = Path(__file__).resolve().parents[3]
    reports_dir = repo / "tools" / "crossval" / "reports"

    print(f"\n=== LIVE DEMO: seed {seed}, Red Deck / White Stake, full run ===\n",
          flush=True)
    policy_fn = narrating(make_ckpt_policy(args.ckpt, args.device))
    res = run_one(
        seed, game, "ckpt", max_antes=8, max_steps=args.max_steps,
        reports_dir=reports_dir, continue_on_divergence=True,
        policy_fn=policy_fn,
    )

    won = res.get("ante", 0) > 8
    print(f"\n=== {'VICTORY — ante 8 boss defeated!' if won else 'Run over.'} "
          f"status={res['status']} steps={res['steps']} ante={res.get('ante')} "
          f"divergences={res.get('divergences', 0)} ===", flush=True)
    return 1 if res["status"] == "game_error" else 0


if __name__ == "__main__":
    sys.exit(main())
