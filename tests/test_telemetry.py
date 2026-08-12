"""Verify telemetry: correctness, drop-on-full, emit cost, and session dirs."""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.telemetry import (
    EventType, RunSession, Stopwatch, TelemetryEmitter, close_emitter,
    emit, find_latest_run, phase, set_emitter,
)

tmp = tempfile.mkdtemp()

# --- session directory ---
runs_dir = os.path.join(tmp, "runs")
sess = RunSession.create(name="smoke test!", runs_dir=runs_dir, config={"lr": 1e-4})
assert os.path.isdir(sess.tb_dir) and os.path.isdir(sess.checkpoints_dir)
meta = sess.read_meta()
assert meta["active"] is True and meta["trainer_pid"] == os.getpid()
assert "/" not in sess.run_id and " " not in sess.run_id, sess.run_id
print("run dir OK:", sess.run_id, "| git_sha:", (meta["git_sha"] or "none")[:8], "| dirty:", meta["git_dirty"])

# --- round trip ---
em = TelemetryEmitter(sess.events_path)
set_emitter(em)
emit(EventType.SESSION_START, run_id=sess.run_id)
for i in range(500):
    emit(EventType.STEP, step=i, t_wait=0.48, chips=i * 10)
emit(EventType.SESSION_END)
close_emitter()

lines = open(sess.events_path, encoding="utf-8").read().strip().split("\n")
assert len(lines) == 502, f"expected 502 events, got {len(lines)}"
evts = [json.loads(l) for l in lines]
assert [e["seq"] for e in evts] == list(range(1, 503)), "seq not monotonic"
assert evts[0]["type"] == EventType.SESSION_START
assert evts[250]["data"]["chips"] == 2490
assert all(e["v"] == 1 for e in evts)
print(f"round trip OK: {len(evts)} events, seq monotonic, schema v1")

# --- emit must never block: tiny queue, flood it ---
em2 = TelemetryEmitter(os.path.join(tmp, "drop.jsonl"), queue_size=8, flush_interval=10)
t0 = time.perf_counter()
for i in range(20000):
    em2.emit(EventType.STEP, i=i)
flood = time.perf_counter() - t0
per_emit_us = flood / 20000 * 1e6
assert em2.dropped > 0, "expected drops with an 8-slot queue"
em2.close()
print(f"drop-on-full OK: {em2.dropped} dropped of 20000, never blocked")
print(f"emit cost: {per_emit_us:.2f} us/event  (step budget is 483000 us)")
assert per_emit_us < 100, f"emit too expensive: {per_emit_us}us"

# --- unserializable payload must not kill the writer ---
class Bad:
    pass
em3 = TelemetryEmitter(os.path.join(tmp, "bad.jsonl"))
em3.emit(EventType.STEP, obj=Bad())
em3.emit(EventType.STEP, ok=1)
em3.close()
kept = [json.loads(l) for l in open(os.path.join(tmp, "bad.jsonl"), encoding="utf-8")]
assert any(e["data"].get("ok") == 1 for e in kept), "writer died on bad payload"
print(f"bad payload survived: {len(kept)} events written, writer alive")

# --- timing helpers ---
store = {}
with phase(store, "sleepy"):
    time.sleep(0.02)
assert 0.015 < store["sleepy"] < 0.1, store
sw = Stopwatch().start()
assert sw.running
time.sleep(0.01)
assert 0.005 < sw.stop() < 0.05 and not sw.running
assert Stopwatch().stop() == 0.0, "unstarted stopwatch must return 0"
print("timing helpers OK")

# --- run discovery ---
sess.mark_finished()
assert find_latest_run(runs_dir).run_id == sess.run_id
assert find_latest_run(runs_dir, active_only=True) is None, "finished run still active"
print("run discovery OK")

shutil.rmtree(tmp, ignore_errors=True)
print("\nALL TELEMETRY TESTS PASSED")
