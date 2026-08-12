"""
Win rate panel

Rolling win rate with the cumulative rate as a headline number.

A single series, so no legend box is needed -- the axis label names it. The
current value is direct-labelled via the stat tile rather than annotating points.
"""

from collections import deque
from typing import Any, Dict

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class WinRatePanel(Panel):
    """Rolling and cumulative win rate"""

    NAME = "winrate"
    TITLE = "Win rate"
    EVENT_TYPES = frozenset({"episode_end"})
    SIZE = (420, 230)

    WINDOW = 50

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.series = RingSeries(config.history)
        self.window: deque = deque(maxlen=self.WINDOW)
        self.wins = 0
        self.episodes = 0

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_rate = StatTile(f"rolling ({self.WINDOW})")
        self.tile_total = StatTile("wins / episodes")
        row.addWidget(self.tile_rate)
        row.addWidget(self.tile_total)

        self.plot = theme.make_plot(y_label="win rate (%)", x_label="episode")
        self.curve = self.plot.plot([], [], pen=theme.pen(theme.SERIES[0]))
        self.plot.getPlotItem().setYRange(0, 100)
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"ep {x:.0f}   {y:.1f}%"
        )

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        outcome = data.get("outcome")
        if outcome not in ("win", "loss"):
            return

        won = 1 if outcome == "win" else 0
        self.episodes += 1
        self.wins += won
        self.window.append(won)

        episode = data.get("episode", self.episodes)
        self.series.append(episode, sum(self.window) / len(self.window) * 100.0)

    def redraw(self) -> None:
        xs, ys = self.series.arrays(self.config.max_plot_points)
        self.curve.setData(xs, ys)
        self.crosshair.set_series(xs, ys)

        if not self.episodes:
            self.tile_rate.set_value("--")
            self.tile_total.set_value("--")
            return

        rolling = sum(self.window) / len(self.window) * 100.0
        # Status colour here means "is this run going anywhere", which is a state
        # judgement rather than series identity
        role = "good" if rolling >= 25 else "warning" if rolling >= 5 else "critical"
        self.tile_rate.set_value(f"{rolling:.1f}%", theme.STATUS[role])
        self.tile_total.set_value(f"{self.wins} / {self.episodes}")

    def clear(self) -> None:
        self.series.clear()
        self.window.clear()
        self.wins = 0
        self.episodes = 0
