"""
Reward panel

Per-episode reward as a cloud of points with a rolling mean through it, plus a
zero rule so wins separate from losses at a glance.

Both series are rewards on one axis, which is what makes the overlay honest.
"""

from collections import deque
from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

from .. import theme
from .base import Panel, RingSeries


class RewardPanel(Panel):
    """Episode reward with a rolling mean"""

    NAME = "reward"
    TITLE = "Episode reward"
    EVENT_TYPES = frozenset({"episode_end"})
    SIZE = (480, 260)

    WINDOW = 25

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.points = RingSeries(config.history)
        self.rolling = RingSeries(config.history)
        self.window: deque = deque(maxlen=self.WINDOW)

    def build(self) -> QWidget:
        self.plot = theme.make_plot(y_label="reward", x_label="episode")

        zero = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(theme.AXIS, width=1)
        )
        self.plot.addItem(zero, ignoreBounds=True)

        # Built before the curves: pyqtgraph only auto-registers named items
        # added after addLegend
        theme.legend(self.plot)

        self.scatter = self.plot.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolSize=theme.MARKER_SIZE - 2,
            symbolBrush=theme.fill(theme.SERIES[0], alpha=90),
            # 2px surface ring keeps overlapping markers separable without
            # outlining every mark in a contrasting colour
            symbolPen=pg.mkPen(theme.SURFACE, width=2),
            name="per episode",
        )
        self.mean_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[1]), name=f"mean ({self.WINDOW})"
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"ep {x:.0f}   {y:+.1f}"
        )
        return self.plot

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        episode = data.get("episode")
        reward = data.get("reward")
        if episode is None or reward is None:
            return

        self.points.append(episode, reward)
        self.window.append(reward)
        self.rolling.append(episode, sum(self.window) / len(self.window))

    def redraw(self) -> None:
        xs, ys = self.points.arrays(self.config.max_plot_points)
        self.scatter.setData(xs, ys)

        mx, my = self.rolling.arrays(self.config.max_plot_points)
        self.mean_curve.setData(mx, my)
        self.crosshair.set_series(mx, my)

    def clear(self) -> None:
        self.points.clear()
        self.rolling.clear()
        self.window.clear()
