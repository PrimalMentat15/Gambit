# Monitor

Live dashboard over the telemetry event stream.

```bash
pip install -r ../requirements-monitor.txt
python -m monitor
```

## How it fits

The monitor is a **read-only consumer** of `runs/<id>/events.jsonl`. It never
talks to the trainer or the game, so it can be started late, restarted, or
crash outright without affecting a run in progress. That one-way relationship is
the whole integration.

```
trainer ──appends──> runs/<id>/events.jsonl <──tails── monitor
```

## Adding a panel

Drop a file in `panels/`. The registry discovers it; the shell docks it.

```python
from .base import Panel, RingSeries
from .. import theme

class MyPanel(Panel):
    NAME = "my_panel"                       # unique id, used in monitor.json
    TITLE = "My Panel"                      # dock title
    EVENT_TYPES = frozenset({"step"})       # empty means every event
    SIZE = (420, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.series = RingSeries(config.history)

    def build(self):
        self.plot = theme.make_plot(y_label="chips", x_label="step")
        self.curve = self.plot.plot([], [], pen=theme.pen(theme.SERIES[0]))
        return self.plot

    def on_event(self, event):              # accumulate only, never paint
        self.series.append(event["data"]["total_step"], event["data"]["chips"])

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
  "runs_dir": "runs",
  "redraw_hz": 10,
  "history": 20000,
  "max_plot_points": 2000,
  "panels": ["throughput", "latency_action", "reward"],
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

Action hues are keyed to the action ids in `RLBridge/actions.lua`, so an action
keeps its colour across every panel rather than being assigned by current rank.
