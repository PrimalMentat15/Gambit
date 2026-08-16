"""
Milestone eval panel

The full-game (ante 8, argmax, held-out seeds) evaluation that runs every
``milestone_every_steps`` -- the one number in the whole dashboard that means
the same thing on iteration 10 and iteration 10,000.

It gets its own panel because everything else on screen is measured against a
moving target. Training win rate is relative to the current curriculum goal,
returns are scaled by an annealing shaping term, and episode stats mix
snapshot-started episodes in with fresh ones. This series does none of that, so
it is the honest answer to "is this run going anywhere".

The ante histogram underneath is where the run actually dies. A win rate says
how often the agent finished; the histogram says whether the failures are
clustered at one ante -- which is a curriculum or joker-pool problem -- or spread
out, which is not.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class MilestonePanel(Panel):
    """Ante-8 argmax win rate over time, with the terminal-ante histogram"""

    NAME = "milestone"
    TITLE = "Milestone eval (ante 8)"
    EVENT_TYPES = frozenset({"milestone_eval"})
    SIZE = (480, 320)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.win_rate = RingSeries(config.history)
        self.ante_mean = RingSeries(config.history)
        self.hist: Dict[str, float] = {}
        self.latest: Dict[str, Any] = {}
        self.bars = None

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_win = StatTile("win rate")
        self.tile_ante = StatTile("mean ante")
        self.tile_episodes = StatTile("episodes")
        for tile in (self.tile_win, self.tile_ante, self.tile_episodes):
            row.addWidget(tile)

        self.plot = theme.make_plot(y_label="win rate (%)", x_label="env step")
        self.plot.getPlotItem().setYRange(0, 100)
        self.curve = self.plot.plot(
            [], [],
            pen=theme.pen(theme.SERIES[5]),
            symbol="s",
            symbolSize=theme.MARKER_SIZE - 3,
            symbolBrush=theme.SERIES[5],
            symbolPen=pg.mkPen(None),
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:.1f}%"
        )

        # Capped low: the win-rate trend is the panel's subject and the
        # histogram is context, so a docked panel dragged short shrinks the
        # histogram rather than squeezing the trend to a few pixels.
        self.hist_plot = theme.make_plot(y_label="share (%)", x_label="terminal ante")
        self.hist_plot.setMaximumHeight(80)

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.hist_plot)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        step = data.get("step")
        if step is None:
            return

        self.latest = data
        if data.get("win_rate") is not None:
            self.win_rate.append(step, data["win_rate"] * 100.0)
        if data.get("ante_mean") is not None:
            self.ante_mean.append(step, data["ante_mean"])
        if data.get("ante_hist"):
            self.hist = data["ante_hist"]

    def redraw(self) -> None:
        xs, ys = self.win_rate.arrays(self.config.max_plot_points)
        self.curve.setData(xs, ys)
        self.crosshair.set_series(xs, ys)

        rate = self.latest.get("win_rate")
        if rate is None:
            self.tile_win.set_value("--")
        else:
            percent = rate * 100.0
            # A judgement about the run: the certified reference sits at 71%.
            role = "good" if percent >= 50 else "warning" if percent >= 10 else "critical"
            self.tile_win.set_value(f"{percent:.1f}%", theme.STATUS[role])

        ante = self.latest.get("ante_mean")
        self.tile_ante.set_value(f"{ante:.2f}" if ante is not None else "--")

        episodes = self.latest.get("episodes")
        self.tile_episodes.set_value(f"{episodes:,}" if episodes else "--")

        self._draw_hist()

    def _draw_hist(self) -> None:
        item = self.hist_plot.getPlotItem()
        if self.bars is not None:
            item.removeItem(self.bars)
            self.bars = None
        if not self.hist:
            return

        total = sum(self.hist.values()) or 1
        antes = sorted(self.hist, key=lambda a: int(a))
        values = [self.hist[a] / total * 100.0 for a in antes]

        self.bars = pg.BarGraphItem(
            x=list(range(len(antes))),
            height=values,
            width=0.62,
            brush=pg.mkBrush(theme.SERIES[5]),
            pen=pg.mkPen(None),
        )
        item.addItem(self.bars)
        item.getAxis("bottom").setTicks(
            [[(pos, str(a)) for pos, a in enumerate(antes)]]
        )

    def clear(self) -> None:
        self.win_rate.clear()
        self.ante_mean.clear()
        self.hist = {}
        self.latest = {}
