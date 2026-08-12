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
        f.write(json.dumps({"v": 1, "seq": 1, "type": "step", "data": {}}) + "\n")
        f.write(json.dumps({"v": 1, "seq": 2, "type": "step", "data": {}}) + "\n")

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
        f.write(json.dumps({"v": 1, "seq": 1, "type": "step", "data": {}}) + "\n")
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
    import math
    import random

    events = [{"v": 1, "seq": 0, "t": 0, "type": "session_start",
               "data": {"run_id": "synthetic", "total_timesteps": 250000,
                        "device": "cuda"}}]
    t = 0.0
    step = 0
    for episode in range(60):
        for i in range(17):
            step += 1
            action = 1 if i % 3 else (2 if i % 2 else 3)
            wait = 0.006 if action == 1 else (0.19 if action == 2 else 0.09)
            wait *= random.uniform(0.8, 1.2)
            t += wait
            events.append({"v": 1, "seq": step, "t": t, "type": "step", "data": {
                "episode": episode, "step": i, "total_step": step, "action": action,
                "reward": random.uniform(-1, 2), "chips": min(300, i * 22),
                "blind_chips": 300, "hands_left": max(0, 4 - i // 4),
                "discards_left": max(0, 3 - i // 5), "hand_type": "Pair",
                "retry_count": 1 if random.random() < 0.03 else 0,
                "timings": {"t_map": 2e-5, "t_send": 8e-5, "t_wait": wait,
                            "t_obs": 9e-5, "t_reward": 1e-5},
                "game_timing": {"gamespeed": 100, "drain_passes": 14},
            }})
        won = random.random() < 0.12
        events.append({"v": 1, "seq": step, "t": t, "type": "episode_end", "data": {
            "episode": episode, "outcome": "win" if won else "loss", "steps": 17,
            "reward": random.uniform(40, 95) if won else random.uniform(-25, -10),
            "chips": 320 if won else 180, "blind_chips": 300, "wall_time": 3.2,
        }})
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
