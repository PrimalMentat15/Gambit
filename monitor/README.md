# Monitor

Live dashboard over the telemetry event stream.

```bash
pip install -r ../requirements-monitor.txt
python -m monitor
```

## How it fits

The monitor is a **read-only consumer** of `train/runs/<id>/events.jsonl`. It
never talks to the trainer, so it can be started late, restarted, or crash
outright without affecting a run in progress. That one-way relationship is the
whole integration.

```
trainer ──appends──> train/runs/<id>/events.jsonl <──tails── monitor
```

Read-only is only half the isolation guarantee. The other half is **bounded**:
the two processes share host RAM, and the trainer's `pin_memory` staging
buffers are page-locked and cannot be swapped out. A monitor that allocates
without limit can therefore kill a run it never writes to — see Phase 7
Stage 0c in `DECISIONS.md` for the incident. Three rules keep it bounded:

* **Reads are capped.** `TailReader` never holds more than `MAX_CHUNK` (4 MB)
  of the file at once, however large the file is. `read()` sets `pending` when
  more remains; callers drain in a loop rather than asking for everything.
* **Bulk history is not replayed.** `episode_end` is ~99.95% of a mature run's
  stream and its only consumer holds a bounded ring buffer, so replaying it
  from the start of a multi-day run fills a ring that discards all but the last
  few thousand. Only the last `BACKFILL_BYTES` (16 MB) of it is parsed. The
  filter runs on raw bytes (`_peek_type`) **before** `json.loads`, because
  parsing is the expensive part — ~955 B of Python objects per event.
* **Rare events are never dropped.** Everything except the bulk types is
  parsed for the whole file, so promotion history, milestones and per-iteration
  curves are complete rather than truncated.

Attaching to a 1.01 GB stream costs ~55 MB peak and ~19 s, against ~7.5 GB and
a probable crash before this was in place.

**"Follow latest" follows the latest *active* run**, not the newest directory.
A short-lived debug or test run sorts newest by timestamp and would otherwise
pull the view off a live multi-day run — and switching back pays a full
re-scan. Tests must never write into the repo's real `train/runs/`; an autouse
fixture in `train/tests/conftest.py` fails the suite if one does.

It imports `balatro_train.telemetry` for the run-directory and event-schema
definitions rather than duplicating them. That package is stdlib-only, so the
monitor's own environment needs no trainer dependency; `monitor/__init__.py`
puts `train/` on `sys.path` so the import resolves.

## Adding a panel

Drop a file in `panels/`. The registry discovers it; the shell docks it.

```python
from .base import Panel, RingSeries
from .. import theme

class MyPanel(Panel):
    NAME = "my_panel"                       # unique id, used in monitor.json
    TITLE = "My Panel"                      # dock title
    EVENT_TYPES = frozenset({"rollout"})    # empty means every event
    SIZE = (420, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.series = RingSeries(config.history)

    def build(self):
        self.plot = theme.make_plot(y_label="entropy", x_label="env step")
        self.curve = self.plot.plot([], [], pen=theme.pen(theme.SERIES[0]))
        return self.plot

    def on_event(self, event):              # accumulate only, never paint
        self.series.append(event["data"]["global_step"], event["data"]["entropy"])

    def redraw(self):                       # paint, called at most redraw_hz
        self.curve.setData(*self.series.arrays(self.config.max_plot_points))

    def clear(self):                        # called when the run changes
        self.series.clear()
```

`on_event` and `redraw` are deliberately separate. Events arrive whenever the
trainer emits them; painting happens on one shared timer and only while the panel
is visible, so a hidden tab costs nothing.

## Configuration

Optional `monitor.json` in the repo root:

```json
{
  "runs_dir": "train/runs",
  "train_dir": "train",
  "config_path": "configs/m3_fast.yaml",
  "redraw_hz": 10,
  "history": 20000,
  "max_plot_points": 2000,
  "panels": ["milestone", "curriculum", "reward"],
  "follow_latest": true
}
```

An empty `panels` list means every discovered panel. Dock arrangement is saved
separately to `.monitor_layout.json` via the toolbar.

## Colour

`theme.py` holds a validated palette. The slot **ordering** is the
colour-vision-deficiency safety mechanism, not decoration — the three slots in use
pass the lightness band, chroma floor, CVD separation (worst adjacent ΔE 9.4),
normal-vision floor (20.9) and 3:1 surface contrast, in both the adjacent and
all-pairs pairlists. Re-validate before substituting colours.

Action hues are keyed to the *game phase* an action belongs to, not to the
action itself: `balatro_train.encoding.ActionType` has 14 members against 8
categorical slots, so per-action hues would either collide silently or run past
the validated palette. Phase is also the question worth asking of a chart of
action shares ("how much time in the shop?"); within a panel, position and
label carry the individual identity. The mapping is `ACTION_PHASES` /
`PHASE_COLORS` in `theme.py`, keyed by the frozen `ActionType` indices.
