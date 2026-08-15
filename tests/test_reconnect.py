"""
Disconnect resilience, action decoding, and observation-size tests

Closing Balatro mid-run used to kill the trainer. These verify it now survives a
game restart and still fails cleanly when the game never returns, that card slots
decode to a legal selection, and that the observation keeps its declared length at
any hand size.

    venv/Scripts/python.exe tests/test_reconnect.py
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 12466
os.environ["BALATRO_RL_PORT"] = str(PORT)
os.environ["BALATRO_RL_VERBOSE_REWARD"] = "0"
os.environ["BALATRO_RL_RECONNECT_TIMEOUT"] = "12"

import numpy as np  # noqa: E402

from ai.environment.balatro_env import RECONNECT_TIMEOUT  # noqa: E402
from ai.utils.communication import BalatroSocketIO  # noqa: E402
from ai.utils.mappers import BalatroActionMapper  # noqa: E402


def game_state(chips=0):
    return {
        "state": 1, "game_over": 0, "game_win": 0,
        "round": {"hands_left": 4, "discards_left": 3},
        "blind_chips": 300, "chips": chips,
        "hand": {"cards": [], "size": 8, "highlighted_count": 0},
        "current_hand": {"chips": 0, "mult": 0, "score": 0, "handname": "None"},
        "seed": "SEED",
    }


def test_card_slot_decoding():
    """
    Card slots decode to a legal 1-5 card selection

    The action format is now one action-type value plus 5 card slots, each
    holding a 0-based hand position or STOP (8). This replaced 8 independent
    binary bits, which could express selections Balatro rejects; see
    tests/test_autoregressive.py for the policy-side guarantee that illegal
    selections are unrepresentable in the first place.
    """
    STOP = 8
    mapper = BalatroActionMapper(
        {"action_selection": slice(0, 1), "card_indices": slice(1, 6)}
    )

    cases = [
        ([0, STOP, STOP, STOP, STOP], [1], "immediate STOP falls back to one card"),
        ([0, 1, STOP, STOP, STOP], [1, 2], "two cards then STOP"),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], "a full five picks"),
        ([7, STOP, STOP, STOP, STOP], [8], "last hand position"),
        # Everything after the first STOP is ignored, so a stale value in a
        # later slot cannot smuggle in an extra card
        ([0, STOP, 3, 4, 5], [1], "values after STOP are ignored"),
    ]

    for slots, expected, label in cases:
        params = mapper._extract_select_hand_params(np.array([0] + slots))
        assert params == expected, f"{label}: got {params}, expected {expected}"
        assert 1 <= len(params) <= 5, f"{label} -> illegal count {params}"
        assert len(set(params)) == len(params), f"{label} -> duplicates {params}"

    # The mapper's clamp is now only a backstop; with the autoregressive policy
    # producing the slots it should never be needed
    assert mapper.clamped_count == 0, f"clamp fired unexpectedly: {mapper.clamped_count}"
    print(f"card-slot decoding OK across {len(cases)} cases, clamp backstop unused")


def test_observation_size_is_stable():
    """
    The observation stays its declared length at any hand size

    Hand size is not a constant: jokers and vouchers change it, and consumables
    like Cryptid insert copies straight into hand, which cardarea.lua does not
    cap. Previously only a hand of exactly 0 or 8 produced a correctly-sized
    observation; anything else silently violated the Box contract.
    """
    from ai.utils.mappers import BalatroStateMapper

    mapper = BalatroStateMapper(observation_size=216, max_actions=3, max_cards=8)

    def state(n):
        card = {"base": {"nominal": 5, "value": "5"}, "highlighted": False,
                "suit": "Hearts"}
        return {
            "game_state": {
                "state": 1, "game_over": 0, "game_win": 0,
                "round": {"hands_left": 4, "discards_left": 3},
                "blind_chips": 300, "chips": 120,
                "hand": {"cards": [dict(card) for _ in range(n)], "size": n,
                         "highlighted_count": 0},
                "current_hand": {"chips": 0, "mult": 0, "score": 0,
                                 "handname": "None"},
                "seed": "SEED",
            },
            "available_actions": [1],
        }

    # 100 covers the Cryptid-stockpile case, which is reachable in real play
    for n in [0, 1, 3, 5, 7, 8, 9, 10, 20, 80, 100]:
        obs = mapper.process_game_state(state(n))
        assert obs.shape == (216,), f"hand {n} produced shape {obs.shape}"
        expected_drop = max(0, n - 8)
        assert mapper.last_hand_truncated == expected_drop, (
            f"hand {n}: reported {mapper.last_hand_truncated} truncated, "
            f"expected {expected_drop}")

    # The safety net in process_game_state must never be what saves us
    assert mapper.resized_observations == 0, (
        "observation needed emergency resizing; the per-card padding is wrong")

    print(f"observation size stable across hands 0-100 "
          f"({mapper.truncated_hands} oversized hands truncated cleanly)")


def test_reconnect_after_drop():
    """A client that drops and returns is picked up again"""

    def first_client():
        # The constructor blocks on accept(), so the client must be waiting
        # before it runs, not started afterwards
        for _ in range(100):
            try:
                s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                break
            except OSError:
                time.sleep(0.05)
        else:
            return
        f = s.makefile("rw", encoding="utf-8", newline="\n")
        f.write(json.dumps({"game_state": game_state(10),
                            "available_actions": [1]}) + "\n")
        f.flush()
        time.sleep(0.3)
        s.close()  # simulate the game being closed mid-run

    threading.Thread(target=first_client, daemon=True).start()

    io = BalatroSocketIO(port=PORT)

    assert io.wait_for_request() is not None, "first request should arrive"
    assert io.wait_for_request() is None, "closed peer should read as disconnect"
    print("disconnect detected")

    # Relaunching the game a few seconds later
    def second_client():
        time.sleep(2.0)
        s = socket.create_connection(("127.0.0.1", PORT))
        f = s.makefile("rw", encoding="utf-8", newline="\n")
        f.write(json.dumps({"game_state": game_state(99),
                            "available_actions": [1]}) + "\n")
        f.flush()
        time.sleep(0.5)

    threading.Thread(target=second_client, daemon=True).start()

    start = time.perf_counter()
    again = io.wait_for_request(accept_timeout=10)
    waited = time.perf_counter() - start

    assert again is not None, "should have accepted the relaunched game"
    assert again["game_state"]["chips"] == 99, again
    print(f"reconnect OK: picked up the relaunched game after {waited:.1f}s")
    io.cleanup()


def test_accept_timeout_is_bounded():
    """A game that never returns times out rather than hanging forever"""
    io = BalatroSocketIO.__new__(BalatroSocketIO)  # skip the blocking constructor
    import logging
    io.host, io.port = "127.0.0.1", PORT + 1
    io.logger = logging.getLogger("test")
    io.server = io.client = io.reader = io.writer = None
    io.last_send_s = io.last_wait_s = 0.0
    io.round_trips = 0
    from ai.telemetry import Stopwatch
    io._wait_watch = Stopwatch()
    io.start_server()

    start = time.perf_counter()
    result = io.wait_for_request(accept_timeout=2)
    waited = time.perf_counter() - start

    assert result is None, "no client should mean no request"
    assert 1.5 < waited < 6, f"expected a ~2s bounded wait, took {waited:.1f}s"
    print(f"bounded wait OK: gave up after {waited:.1f}s instead of hanging")
    io.cleanup()


if __name__ == "__main__":
    print(f"reconnect timeout configured to {RECONNECT_TIMEOUT:.0f}s\n")
    test_card_slot_decoding()
    test_observation_size_is_stable()
    test_reconnect_after_drop()
    test_accept_timeout_is_bounded()
    print("\nALL RECONNECT TESTS PASSED")
