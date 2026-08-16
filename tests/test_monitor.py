"""
Monitor tests

Runs headless: no game, no display. Verifies the tail reader's incremental and
restart behaviour, then renders the real window offscreen against a recorded run
and writes a screenshot so the result can actually be looked at.

    venv/Scripts/python.exe tests/test_monitor.py [run_dir] [-o out.png]
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The offscreen platform plugin loads no system fonts on Windows, so every label
# renders as a tofu box and the screenshot is useless for judging the design.
# Use the native platform instead and keep the window off-screen via
# WA_DontShowOnScreen, which lays out and paints without ever displaying it.
# BALATRO_RL_HEADLESS=1 forces offscreen for a machine with no display at all.
if os.environ.get("BALATRO_RL_HEADLESS") == "1":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from monitor import theme  # noqa: E402
from monitor.bus import TailReader  # noqa: E402
from monitor.config import MonitorConfig  # noqa: E402
from monitor.panels import discover  # noqa: E402


def test_tail_reader():
    """Incremental reads, partial lines, and restart detection"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "events.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"v": 1, "seq": 1, "type": "rollout", "data": {}}) + "\n")
        f.write(json.dumps({"v": 1, "seq": 2, "type": "rollout", "data": {}}) + "\n")

    reader = TailReader(path)
    events, restarted = reader.read()
    assert len(events) == 2 and not restarted, (events, restarted)

    # Nothing new
    assert reader.read() == ([], False)

    # A line written in two parts must not be emitted until it is complete
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"v":1,"seq":3,"type":"step",')
    events, _ = reader.read()
    assert events == [], f"partial line emitted: {events}"

    with open(path, "a", encoding="utf-8") as f:
        f.write('"data":{}}\n')
    events, _ = reader.read()
    assert len(events) == 1 and events[0]["seq"] == 3, events

    # Multi-byte character split across a read boundary
    with open(path, "ab") as f:
        f.write('{"v":1,"seq":4,"type":"log","data":{"m":"café – ok"}}'.encode("utf-8")[:30])
    reader.read()
    with open(path, "ab") as f:
        f.write('{"v":1,"seq":4,"type":"log","data":{"m":"café – ok"}}'.encode("utf-8")[30:] + b"\n")
    events, _ = reader.read()
    assert len(events) == 1 and events[0]["data"]["m"] == "café – ok", events

    # A shorter file means a new run
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"v": 1, "seq": 1, "type": "rollout", "data": {}}) + "\n")
    events, restarted = reader.read()
    assert restarted, "file shrink should report a restart"
    assert len(events) == 1, events

    print("tail reader OK: incremental, partial-line safe, utf-8 safe, restart detected")


def test_discovery():
    """Every panel file is found and declares the required attributes"""
    panels = discover()
    assert panels, "no panels discovered"
    for name, cls in sorted(panels.items()):
        assert cls.NAME == name
        assert cls.TITLE and cls.EVENT_TYPES is not None
    print(f"discovered {len(panels)} panels: {', '.join(sorted(panels))}")


def test_render(run_dir=None, out_path="monitor_preview.png"):
    """Build the real window, feed a recorded run, and screenshot it"""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication
    from monitor.app import MonitorWindow

    config = MonitorConfig()
    theme.configure()
    app = QApplication.instance() or QApplication([])

    families = QFontDatabase.families()
    if not families:
        print("WARNING: no fonts available; text will render as boxes")

    window = MonitorWindow(config)
    window.resize(1600, 1000)
    # Lays out and paints without putting a window on screen
    window.setAttribute(Qt.WA_DontShowOnScreen, True)
    window.show()

    # Feed panels directly rather than through the tail thread so the render is
    # deterministic instead of racing a poll interval
    events = []
    if run_dir:
        events_path = os.path.join(run_dir, "events.jsonl")
        if os.path.isfile(events_path):
            with open(events_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    if not events:
        events = _synthetic()

    # Silence the live tail and the run rescan so the render reflects exactly
    # the events fed below rather than racing a poll interval
    window.bus.stop()
    window.scan_timer.stop()
    window.redraw_timer.stop()

    # Point the picker at the run actually being rendered, so the screenshot is
    # not labelled with whichever run happens to be newest on disk
    if run_dir:
        wanted = os.path.join(run_dir, "events.jsonl")
        for i in range(window.run_picker.count()):
            if os.path.normpath(window.run_picker.itemData(i)) == os.path.normpath(wanted):
                window.run_picker.blockSignals(True)
                window.run_picker.setCurrentIndex(i)
                window.run_picker.blockSignals(False)
                break

    for event in events:
        for panel in window.panels:
            panel.handle(event)
    for panel in window.panels:
        panel.maybe_redraw(force=True)
    window._redraw()  # refresh the status bar counters too

    app.processEvents()
    window.grab().save(out_path)

    print(f"fed {len(events)} events into {len(window.panels)} panels")
    print(f"screenshot: {out_path}")

    # One screenshot per tab, so the non-Live views are actually looked at
    # rather than merely constructed without raising
    stem, ext = os.path.splitext(out_path)
    for index in range(1, window.tabs.count()):
        name = window.tabs.tabText(index).lower()
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        for _ in range(3):
            app.processEvents()
        path = f"{stem}_{name}{ext}"
        window.grab().save(path)
        print(f"screenshot: {path}")

    window.tabs.setCurrentIndex(0)
    return out_path


def _synthetic():
    """Fallback data when no recorded run is available"""
    import random

    LADDER = [1, 2, 3, 5, 8]
    N_ACTIONS = 14

    events = [{"v": 1, "seq": 0, "t": 0.0, "type": "session_start",
               "data": {"run_id": "synthetic", "total_timesteps": 400_000_000,
                        "device": "cuda", "num_envs": 2048, "rollout_len": 128,
                        "policy_params": 1_100_000, "win_ante": 1}}]

    seq = 0
    t = 0.0
    step = 0
    rung = 0
    beta = 1.0
    batch = 2048 * 128
    next_promo = 4_000_000
    next_milestone = 20_000_000

    def add(kind, data):
        nonlocal seq
        seq += 1
        events.append({"v": 1, "seq": seq, "t": t, "type": kind, "data": data})

    for iteration in range(1, 220):
        step += batch
        t += 26.0
        beta = max(0.1, beta - 0.004)
        goal = LADDER[rung]
        # Win rate climbs toward the promotion threshold, then resets a rung up
        progress = min(1.0, (step % 40_000_000) / 30_000_000)
        win = min(0.95, 0.05 + progress * 0.75 + random.uniform(-0.04, 0.04))

        counts = [0] * N_ACTIONS
        for action, weight in ((0, 34), (1, 18), (2, 9), (3, 3), (4, 9),
                               (5, 11), (6, 4), (7, 9), (8, 5), (9, 1),
                               (10, 1), (11, 4), (12, 2)):
            counts[action] = int(batch * weight / 110)

        add("rollout", {
            "iteration": iteration, "global_step": step,
            "sps": int(random.uniform(9500, 10500)),
            "loss": random.uniform(0.2, 0.6), "pg_loss": random.uniform(-0.05, 0.05),
            "v_loss": max(0.02, 1.2 - iteration * 0.004) + random.uniform(0, 0.05),
            "entropy": max(0.4, 2.4 - iteration * 0.008),
            "approx_kl": random.uniform(0.004, 0.022),
            "clipfrac": random.uniform(0.05, 0.22),
            "ep_return_mean": -4 + progress * 22 + random.uniform(-1.5, 1.5),
            "ep_length_mean": 60 + progress * 180,
            "ep_ante_mean": 1 + progress * (goal - 1) + random.uniform(-0.2, 0.2),
            "ep_win_rate": win,
            "win_ante": goal,
            "shaping_beta": beta,
            "snapshot_fraction": max(0.0, 0.3 - iteration * 0.0015),
            "ep_from_snapshot_share": max(0.0, 0.3 - iteration * 0.0015)
                                      + random.uniform(-0.02, 0.02),
            "action_counts": counts,
        })

        for _ in range(6):
            won = random.random() < win
            add("episode_end", {
                "step": step,
                "r": random.uniform(15, 40) if won else random.uniform(-8, 6),
                "l": random.randint(40, 260),
                "ante": goal if won else random.randint(1, max(1, goal)),
                "won": won, "from_snapshot": random.random() < 0.2,
            })

        if step >= next_promo:
            next_promo += 4_000_000
            rate = win + random.uniform(-0.03, 0.03)
            add("promotion_eval", {
                "step": step, "win_ante": goal, "win_rate": rate,
                "episodes": 256, "threshold": 0.7, "elapsed": 4.1,
            })
            if rate >= 0.7 and rung < len(LADDER) - 1:
                add("curriculum_promotion", {
                    "step": step, "from_ante": goal, "to_ante": LADDER[rung + 1],
                    "win_rate": rate, "threshold": 0.7,
                })
                rung += 1

        if step >= next_milestone:
            next_milestone += 20_000_000
            rate = min(0.72, progress * 0.5 * (rung + 1) / len(LADDER))
            add("milestone_eval", {
                "step": step, "win_ante": 8, "win_rate": rate,
                "ante_mean": 3 + rate * 5, "return_mean": 8 + rate * 20,
                "length_mean": 240, "episodes": 512, "elapsed": 31.0,
                "ante_hist": {str(a): random.randint(4, 90) for a in range(1, 9)},
            })

        if iteration % 40 == 0:
            add("checkpoint_saved", {
                "step": step, "iteration": iteration,
                "path": f"runs/synthetic/ckpt_{step}.pt", "win_ante": goal,
            })

    add("session_end", {"step": step, "iteration": 219, "status": "stopped",
                        "final_checkpoint": f"runs/synthetic/ckpt_{step}.pt",
                        "elapsed": t})
    return events


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    out = "monitor_preview.png"
    if "-o" in args:
        idx = args.index("-o")
        out = args[idx + 1]
        del args[idx:idx + 2]
    run = args[0] if args else None

    test_tail_reader()
    test_discovery()
    test_render(run, out)
    print("\nALL MONITOR TESTS PASSED")
