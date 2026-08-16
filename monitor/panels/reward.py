"""
Reward panel

Episode return: the trainer's 100-episode mean as a line, with individual
episodes scattered behind it.

Both series are returns on one axis against one x (env step), which is what
makes the overlay honest. The scatter is the spread the mean is hiding; at high
throughput it is also the only way to see a bimodal run (wins and early deaths)
that a mean would render as a flat middle.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

from .. import theme
from .base import Panel, RingSeries


class RewardPanel(Panel):
    """Episode return, per episode and rolling mean"""

    NAME = "reward"
    TITLE = "Episode return"
    EVENT_TYPES = frozenset({"episode_end", "rollout"})
    SIZE = (480, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        # Episodes vastly outnumber iterations, so the scatter gets a shorter
        # ring: it is texture, and keeping a full run of it would evict the
        # mean's history for no gain.
        self.points = RingSeries(max(config.history // 2, 500))
        self.rolling = RingSeries(config.history)

    def build(self) -> QWidget:
        self.plot = theme.make_plot(y_label="return", x_label="env step")

        zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.AXIS, width=1))
        self.plot.addItem(zero, ignoreBounds=True)

        # Built before the curves: pyqtgraph only auto-registers named items
        # added after addLegend
        theme.legend(self.plot)

        self.scatter = self.plot.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolSize=theme.MARKER_SIZE - 3,
            symbolBrush=theme.fill(theme.SERIES[0], alpha=70),
            symbolPen=pg.mkPen(None),
            name="per episode",
        )
        self.mean_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[1]), name="mean (100 ep)"
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:+.2f}"
        )
        return self.plot

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]

        if event["type"] == "episode_end":
            step, value = data.get("step"), data.get("r")
            if step is not None and value is not None:
                self.points.append(step, value)
            return

        step, mean = data.get("global_step"), data.get("ep_return_mean")
        if step is not None and mean is not None:
            self.rolling.append(step, mean)

    def redraw(self) -> None:
        xs, ys = self.points.arrays(self.config.max_plot_points)
        self.scatter.setData(xs, ys)

        mx, my = self.rolling.arrays(self.config.max_plot_points)
        self.mean_curve.setData(mx, my)
        self.crosshair.set_series(mx, my)

    def clear(self) -> None:
        self.points.clear()
        self.rolling.clear()
