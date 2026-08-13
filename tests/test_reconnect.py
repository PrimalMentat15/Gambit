"""
Disconnect resilience and action-clamping tests

Closing Balatro mid-run used to kill the trainer. These verify it now survives a
game restart, still fails cleanly when the game never returns, and that illegal
card selections are projected into legal ones instead of being rejected.

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


def test_clamping():
    """Every selection the policy can emit becomes a legal 1-5 card selection"""
    mapper = BalatroActionMapper(
        {"action_selection": slice(0, 1), "card_indices": slice(1, 9)}
    )

    # Exactly the distribution seen in the real run: 0 and 6-8 were all rejected
    for bits, label in [
        ([0] * 8, "0 cards"),
        ([1, 0, 0, 0, 0, 0, 0, 0], "1 card"),
        ([1, 1, 1, 1, 1, 0, 0, 0], "5 cards"),
        ([1, 1, 1, 1, 1, 1, 0, 0], "6 cards"),
        ([1, 1, 1, 1, 1, 1, 1, 0], "7 cards"),
        ([1] * 8, "8 cards"),
    ]:
        params = mapper._extract_select_hand_params(np.array([0] + bits))
        assert 1 <= len(params) <= 5, f"{label} -> illegal selection {params}"
        assert len(set(params)) == len(params), f"{label} -> duplicate indices"
        assert all(1 <= p <= 8 for p in params), f"{label} -> out of bounds {params}"

    assert mapper.clamped_count == 3, mapper.clamped_count
    assert mapper.empty_count == 1, mapper.empty_count
    print(f"clamping OK: all selections legal, "
          f"{mapper.clamped_count} clamped, {mapper.empty_count} empty-filled")


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
    test_clamping()
    test_reconnect_after_drop()
    test_accept_timeout_is_bounded()
    print("\nALL RECONNECT TESTS PASSED")
