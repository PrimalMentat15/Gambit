"""
End-to-end Stage 1 check: fake Balatro client drives the real BalatroEnv over the
real socket, with a simulated 480ms game-side delay. Verifies telemetry captures
the latency breakdown and that instrumentation overhead is negligible.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["BALATRO_RL_PORT"] = "12455"
os.environ["BALATRO_RL_VERBOSE_REWARD"] = "0"

from ai.telemetry import EventType, RunSession, TelemetryEmitter, close_emitter, emit, set_emitter
from ai.environment.balatro_env import BalatroEnv

PORT = 12455
GAME_DELAY = 0.48   # simulate the measured 483ms game round-trip
N_STEPS = 25

tmp = tempfile.mkdtemp()
sess = RunSession.create(name="e2e", runs_dir=os.path.join(tmp, "runs"))
set_emitter(TelemetryEmitter(sess.events_path))


def game_state(step, chips, hands_left, game_over=0, game_win=0):
    return {
        "state": 1, "game_over": game_over, "game_win": game_win,
        "round": {"hands_left": hands_left, "discards_left": 3},
        "blind_chips": 300, "chips": chips,
        "hand": {"cards": [{"base": {"nominal": 5, "value": "5"}, "highlighted": False,
                            "suit": "Hearts"}] * 8, "size": 8, "highlighted_count": 0},
        "current_hand": {"chips": 30, "mult": 2, "score": 60, "handname": "Pair"},
        "seed": "TESTSEED",
    }


def fake_balatro():
    """Connect, then answer each response after a realistic game-side delay."""
    time.sleep(0.3)
    s = socket.create_connection(("127.0.0.1", PORT))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = s.makefile("rw", encoding="utf-8", newline="\n")
    frames = 0
    for i in range(N_STEPS + 2):
        over = 1 if i == N_STEPS else 0
        frames += 29
        req = {
            "game_state": game_state(i, i * 40, max(0, 4 - i // 5), game_over=over),
            "available_actions": [1, 2, 3],
            "timing": {
                "t_frame": time.time(), "elapsed": GAME_DELAY,
                "frames_since_last_action": 29, "blocked_frames": 24,
                "not_ready_frames": 26,
            },
        }
        f.write(json.dumps(req) + "\n")
        f.flush()
        if not f.readline():
            break
        time.sleep(GAME_DELAY)   # the game is busy animating
    f.close(); s.close()


threading.Thread(target=fake_balatro, daemon=True).start()

env = BalatroEnv()
obs, info = env.reset()
print(f"reset OK: obs shape {obs.shape}")

wall = time.perf_counter()
steps = 0
for i in range(N_STEPS):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    steps += 1
    if i == 0:
        assert info["chips"] is not None, info
        assert "hands_left" in info and "hand_type" in info, info
        print(f"info populated: action={info['action']} chips={info['chips']} "
              f"hand_type={info['hand_type']} components={len(info['reward_components'])}")
    if term:
        print(f"episode terminated at step {steps} (outcome={info.get('outcome')})")
        break
wall = time.perf_counter() - wall

env.cleanup()
close_emitter()

# --- verify telemetry ---
evts = [json.loads(l) for l in open(sess.events_path, encoding="utf-8") if l.strip()]
kinds = {}
for e in evts:
    kinds[e["type"]] = kinds.get(e["type"], 0) + 1
print(f"\nevents: {kinds}")

step_evts = [e for e in evts if e["type"] == EventType.STEP]
assert step_evts, "no step events"
t = step_evts[0]["data"]["timings"]
assert {"t_map", "t_send", "t_wait", "t_obs", "t_reward"} <= set(t), t
gt = step_evts[0]["data"]["game_timing"]
assert gt["blocked_frames"] == 24, gt
print(f"stage keys present: {sorted(t)}")
print(f"game_timing forwarded: {gt['frames_since_last_action']} frames, "
      f"{gt['blocked_frames']} blocked")

assert any(e["type"] == EventType.EPISODE_END for e in evts), "no episode_end"

# overhead: everything except waiting on the game
waits = [e["data"]["timings"]["t_wait"] for e in step_evts]
overhead = [sum(v for k, v in e["data"]["timings"].items() if k != "t_wait") for e in step_evts]
mean_wait = sum(waits) / len(waits)
mean_oh = sum(overhead) / len(overhead)
print(f"\nmean t_wait   : {mean_wait*1000:7.1f} ms  (simulated game delay {GAME_DELAY*1000:.0f} ms)")
print(f"mean overhead : {mean_oh*1000:7.3f} ms  ({mean_oh/(mean_wait+mean_oh)*100:.3f}% of step)")
assert 0.40 < mean_wait < 0.60, f"t_wait {mean_wait} should track the game delay"
assert mean_oh < 0.005, f"python overhead too high: {mean_oh*1000:.2f}ms"

print(f"\nwall clock: {wall:.2f}s for {steps} steps ({wall/steps*1000:.0f} ms/step)")
print("EVENTS PATH:", sess.events_path)
print("RUN DIR:", sess.run_dir)
print("\nE2E PASSED")
