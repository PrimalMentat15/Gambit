"""Round-trip smoke test for BalatroSocketIO using a fake Balatro client."""
import json
import logging
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.utils.communication import BalatroSocketIO

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

PORT = 12399
results = {}


def fake_balatro():
    """Mimics the Lua side: connect, send request, read response."""
    time.sleep(0.3)
    for attempt in range(2):  # second pass tests reconnect after disconnect
        s = socket.create_connection(("127.0.0.1", PORT))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = s.makefile("rw", encoding="utf-8", newline="\n")
        req = {"game_state": {"state": 1, "pass": attempt}, "available_actions": [1, 2, 3]}
        f.write(json.dumps(req) + "\n")
        f.flush()
        resp = f.readline().strip()
        results.setdefault("responses", []).append(json.loads(resp))
        f.close()
        s.close()


t = threading.Thread(target=fake_balatro, daemon=True)
t.start()

io = BalatroSocketIO(port=PORT)

for expected_pass in (0, 1):
    req = io.wait_for_request()
    assert req is not None, "no request received"
    assert req["game_state"]["pass"] == expected_pass, req
    assert req["available_actions"] == [1, 2, 3], req
    assert io.send_response({"action": 2, "params": {"cards": [0, 1]}}), "send failed"

    # peer closes -> next read must report disconnect
    eof = io.wait_for_request()
    assert eof is None, f"expected None on disconnect, got {eof}"
    print(f"pass {expected_pass}: round-trip + disconnect detection OK")

t.join(timeout=5)
io.cleanup()
assert len(results["responses"]) == 2, results
assert results["responses"][0]["action"] == 2
print("ALL OK:", results["responses"])
