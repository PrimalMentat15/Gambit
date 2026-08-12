"""
Latency by action panel

Horizontal bars of mean wait time per action. This is the view that made the
scoring-animation cost obvious: a single blended average hides that one action
costs 20x another.

Bars are direct-labelled with their values, so the reading never depends on
telling two hues apart, and the numbers are available without hovering.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

from .. import theme
from .base import Panel


class LatencyActionPanel(Panel):
    """Mean game wait per action, ranked"""

    NAME = "latency_action"
    TITLE = "Latency by action"
    EVENT_TYPES = frozenset({"step"})
    SIZE = (420, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.totals: Dict[int, float] = {}
        self.counts: Dict[int, int] = {}
        self.bars = None
        self.labels: list = []

    def build(self) -> QWidget:
        self.plot = theme.make_plot(x_label="mean wait (ms)")
        item = self.plot.getPlotItem()
        item.showGrid(x=True, y=False, alpha=0.12)
        item.getAxis("left").setStyle(tickLength=0)
        return self.plot

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        action = data.get("action")
        wait = (data.get("timings") or {}).get("t_wait")
        if action is None or wait is None:
            return
        self.totals[action] = self.totals.get(action, 0.0) + wait
        self.counts[action] = self.counts.get(action, 0) + 1

    def redraw(self) -> None:
        item = self.plot.getPlotItem()

        if self.bars is not None:
            item.removeItem(self.bars)
            self.bars = None
        for label in self.labels:
            item.removeItem(label)
        self.labels.clear()

        if not self.counts:
            return

        # Rank by cost, but colour by action identity so a hue never migrates
        # between actions as the ranking shifts
        ranked = sorted(
            self.counts,
            key=lambda a: self.totals[a] / self.counts[a],
        )

        positions, values, colors, names = [], [], [], []
        for index, action in enumerate(ranked):
            mean_ms = self.totals[action] / self.counts[action] * 1000.0
            positions.append(index)
            values.append(mean_ms)
            colors.append(theme.ACTION_COLORS.get(action, theme.SERIES[7]))
            names.append(theme.ACTION_NAMES.get(action, str(action)))

        self.bars = pg.BarGraphItem(
            x0=0,
            y=positions,
            height=0.5,
            width=values,
            brushes=[pg.mkBrush(c) for c in colors],
            pen=pg.mkPen(None),
        )
        item.addItem(self.bars)

        widest = max(values) if values else 1.0
        for index, (value, name) in enumerate(zip(values, names)):
            # Label outside the bar end so a short bar never clips its own text
            label = pg.TextItem(
                f"{name}  {value:.0f} ms", color=theme.INK_2, anchor=(0, 0.5)
            )
            label.setPos(value + widest * 0.03, index)
            item.addItem(label)
            self.labels.append(label)

        item.getAxis("left").setTicks([[]])
        item.setYRange(-0.6, len(positions) - 0.4)
        # Headroom for the labels sitting past each bar end
        item.setXRange(0, widest * 1.45)

    def clear(self) -> None:
        self.totals.clear()
        self.counts.clear()
