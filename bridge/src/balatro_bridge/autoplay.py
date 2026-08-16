"""Autoplay daemon: click BOT RUN in the game's main menu, the bot plays.

Pairs with the BotRun mod (bridge/mods/BotRun): the in-game button writes a
request marker into the game's save dir via love.filesystem; this daemon
polls for it, then starts a fresh RANDOM-seed Red Deck / White Stake run and
plays it with the trained policy (argmax), exactly like balatro-bridge-demo
(sim-lockstep decisions, every action mirrored to the live game, state
diffed each step, divergences reported but non-fatal). When the run ends the
daemon goes back to waiting, so the button works any number of times.

Run (game already up via balatro-bridge-launch):
    balatro-bridge-autoplay --ckpt ../train/runs/cloud_mirror/m5e2/ckpt_4000317440.pt
"""

from __future__ import annotations

import argparse
import random
import string
import sys
import time
import traceback
from pathlib import Path

from balatro_bridge.crossval import GameDriver, run_one
from balatro_bridge.demo import narrating
from balatro_bridge.paths import BOTRUN_REQUEST_FILE as DEFAULT_REQUEST_FILE


def random_seed() -> str:
    rng = random.SystemRandom()
    return "".join(rng.choices(string.digits + string.ascii_uppercase, k=8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="trained policy .pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12346)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--request-file", type=Path, default=DEFAULT_REQUEST_FILE,
                    help="marker file the BotRun mod writes")
    args = ap.parse_args()

    from balatro_bridge.crossval import make_ckpt_policy

    policy_fn = narrating(make_ckpt_policy(args.ckpt, args.device))
    game = GameDriver(args.host, args.port)
    game.client.health()

    repo = Path(__file__).resolve().parents[3]
    reports_dir = repo / "tools" / "crossval" / "reports"

    args.request_file.unlink(missing_ok=True)  # ignore clicks from before startup
    print(f"autoplay ready — click BOT RUN in the game's main menu "
          f"(watching {args.request_file})", flush=True)
    while True:
        if not args.request_file.exists():
            time.sleep(1.0)
            continue
        args.request_file.unlink(missing_ok=True)
        seed = random_seed()
        print(f"\n=== BOT RUN requested: seed {seed}, Red Deck / White Stake ===\n",
              flush=True)
        try:
            res = run_one(
                seed, game, "ckpt", max_antes=8, max_steps=args.max_steps,
                reports_dir=reports_dir, continue_on_divergence=True,
                policy_fn=policy_fn,
            )
            won = res.get("ante", 0) > 8
            print(f"\n=== {'VICTORY!' if won else 'Run over.'} status={res['status']} "
                  f"steps={res.get('steps')} ante={res.get('ante')} "
                  f"divergences={res.get('divergences', 0)} ===\n", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # A quit/intervention mid-run surfaces here (TimeoutError from the
            # lockstep waits, protocol errors, ...). The daemon must outlive
            # the run: the next start() re-menus the game itself.
            traceback.print_exc()
            print(f"\n=== Run aborted ({type(exc).__name__}: {exc}) — "
                  f"daemon still up ===\n", flush=True)
        finally:
            # Drop clicks made while the bot was playing; each run needs a
            # deliberate button press.
            args.request_file.unlink(missing_ok=True)
        print("autoplay ready — click BOT RUN in the game's main menu",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
