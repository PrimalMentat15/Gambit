"""Smoke test for the real-game bridge.

With Balatro already running with the balatrobot mod (see launch.py),
this script:

1. checks `health`,
2. returns to the menu if needed,
3. starts a seeded run,
4. selects the Small Blind and plays one hand,
5. prints the resulting gamestate.

Usage (game must be running):

    cd bridge
    balatro-bridge-smoke                # defaults: RED/WHITE, seed BRIDGE1
    balatro-bridge-smoke --seed AAA111 --deck BLUE --full
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import cast

import httpx

from balatro_bridge.client import APIError, BalatroBridgeClient
from balatro_bridge.types import Deck, GameState, Stake


def summarize(gs: GameState) -> str:
    round_info = gs.get("round", {})
    hand = gs.get("hand", {})
    cards = ", ".join(
        c.get("label", c.get("key", "?")) for c in hand.get("cards", [])
    )
    lines = [
        f"  state:        {gs.get('state')}",
        f"  deck/stake:   {gs.get('deck')}/{gs.get('stake')}  seed={gs.get('seed')}",
        f"  ante/round:   {gs.get('ante_num')}/{gs.get('round_num')}",
        f"  money:        ${gs.get('money')}",
        f"  chips scored: {round_info.get('chips')}",
        f"  hands left:   {round_info.get('hands_left')}"
        f"  discards left: {round_info.get('discards_left')}",
        f"  hand:         [{cards}]",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="balatro-bridge-smoke", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--deck", default="RED", help="deck enum, e.g. RED, BLUE")
    parser.add_argument("--stake", default="WHITE", help="stake enum, e.g. WHITE")
    parser.add_argument("--seed", default="BRIDGE1", help="run seed")
    parser.add_argument(
        "--wait", type=float, default=0.0,
        help="seconds to wait for the API to come up (0 = fail fast)",
    )
    parser.add_argument(
        "--full", action="store_true", help="also dump the full gamestate JSON"
    )
    args = parser.parse_args(argv)

    client = BalatroBridgeClient(host=args.host, port=args.port)

    try:
        if args.wait > 0:
            print(f"[1/5] waiting for API at {client.url} (up to {args.wait:.0f}s)...")
            client.wait_until_ready(timeout=args.wait)
        health = client.health()
        print(f"[1/5] health: {health}")

        gs = client.gamestate()
        state = gs.get("state")
        print(f"[2/5] current state: {state}")
        if state != "MENU":
            print("      not in MENU; calling menu()...")
            gs = client.menu()

        print(
            f"[3/5] starting seeded run: deck={args.deck} "
            f"stake={args.stake} seed={args.seed}"
        )
        gs = client.start(
            deck=cast(Deck, args.deck),
            stake=cast(Stake, args.stake),
            seed=args.seed,
        )
        print(f"      state: {gs.get('state')}")

        print("[4/5] selecting blind and playing first 5 cards...")
        gs = client.select()
        hand_cards = gs.get("hand", {}).get("cards", [])
        n = min(5, len(hand_cards))
        if n == 0:
            print("error: no cards in hand after selecting blind", file=sys.stderr)
            return 1
        gs = client.play(list(range(n)))

        print("[5/5] gamestate after one hand:")
        print(summarize(gs))
        if args.full:
            print(json.dumps(gs, indent=2))
        print("\nsmoke test PASSED")
        return 0

    except APIError as exc:
        print(f"\nAPI error: {exc.name} (code {exc.code}): {exc.message}",
              file=sys.stderr)
        return 1
    except (httpx.HTTPError, OSError, TimeoutError) as exc:
        print(
            f"\ncould not reach balatrobot at {client.url}: {exc}\n"
            "Is the game running? Start it with:\n"
            "  cd bridge && balatro-bridge-launch",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
