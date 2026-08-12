"""
Latency over time panel

Per-step wait time as it evolves, with a rolling median alongside the raw values.

The raw series and the rolling median share one axis and one unit, which is what
makes them safe to overlay. The distribution here is strongly bimodal -- card
selection costs single-digit milliseconds while a scored hand costs hundreds -- so
the median line is what makes the trend legible against that spread.
"""

from collections import deque
from statistics import median
from typing import Any, Dict

from PySide6.QtWidgets import QWidget

from .. import theme
from .base import Panel, RingSeries


class LatencyTimePanel(Panel):
    """Raw and rolling-median wait time per step"""

    NAME = "latency_time"
    TITLE = "Latency over time"
    EVENT_TYPES = frozenset({"step"})
    SIZE = (480, 260)

    WINDOW = 50

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.raw = RingSeries(config.history)
        self.rolling = RingSeries(config.history)
        self.window: deque = deque(maxlen=self.WINDOW)
        self.step = 0

    def build(self) -> QWidget:
        self.plot = theme.make_plot(y_label="wait (ms)", x_label="step")

        # The legend must exist before the curves: pyqtgraph only auto-registers
        # named items added after addLegend, so building it later leaves it empty
        theme.legend(self.plot)

        # Raw points sit under the trend line, thin and translucent so they read
        # as a cloud rather than competing with it
        self.scatter = self.plot.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolSize=theme.MARKER_SIZE - 3,
            symbolBrush=theme.fill(theme.SERIES[0], alpha=70),
            symbolPen=None,
            name="per step",
        )
        self.median_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[1]), name=f"median ({self.WINDOW})"
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:.0f}   {y:.0f} ms"
        )
        return self.plot

    def on_event(self, event: Dict[str, Any]) -> None:
        wait = (event["data"].get("timings") or {}).get("t_wait")
        if wait is None:
            return

        self.step = event["data"].get("total_step", self.step + 1)
        ms = wait * 1000.0

        self.raw.append(self.step, ms)
        self.window.append(ms)
        self.rolling.append(self.step, median(self.window))

    def redraw(self) -> None:
        xs, ys = self.raw.arrays(self.config.max_plot_points)
        self.scatter.setData(xs, ys)

        mx, my = self.rolling.arrays(self.config.max_plot_points)
        self.median_curve.setData(mx, my)
        self.crosshair.set_series(mx, my)

    def clear(self) -> None:
        self.raw.clear()
        self.rolling.clear()
        self.window.clear()
        self.step = 0
