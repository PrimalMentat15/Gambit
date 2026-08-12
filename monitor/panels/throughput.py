"""
Throughput panel

Steps per second over time, plus the headline numbers a long run is judged on:
current rate, total steps, and projected time remaining.

Deliberately separate from the latency panel: rate and latency have different
units and scales, and putting them on one plot with two y-axes would invent a
relationship the data does not contain.
"""

import time
from typing import Any, Dict

from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class ThroughputPanel(Panel):
    """Env steps/sec, with total and ETA"""

    NAME = "throughput"
    TITLE = "Throughput"
    EVENT_TYPES = frozenset({"step", "session_start"})
    SIZE = (480, 260)

    WINDOW = 40  # steps averaged for the instantaneous rate

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.series = RingSeries(config.history)
        self.recent: list = []
        self.total_steps = 0
        self.target_steps = 0
        self.first_seen = None
        self.rate = 0.0

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        grid = QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.tile_rate = StatTile("steps / sec")
        self.tile_total = StatTile("total steps")
        self.tile_eta = StatTile("est. remaining")
        for column, tile in enumerate((self.tile_rate, self.tile_total, self.tile_eta)):
            grid.addWidget(tile, 0, column)

        self.plot = theme.make_plot(y_label="steps/sec", x_label="step")
        self.curve = self.plot.plot([], [], pen=theme.pen(theme.SERIES[0]))
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:.0f}   {y:.2f}/s"
        )

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]

        if event["type"] == "session_start":
            self.target_steps = data.get("total_timesteps", 0) or 0
            return

        now = event.get("t") or time.time()
        if self.first_seen is None:
            self.first_seen = now

        self.total_steps = data.get("total_step", self.total_steps + 1)

        self.recent.append(now)
        if len(self.recent) > self.WINDOW:
            self.recent = self.recent[-self.WINDOW:]

        if len(self.recent) >= 2:
            span = self.recent[-1] - self.recent[0]
            if span > 0:
                self.rate = (len(self.recent) - 1) / span
                self.series.append(self.total_steps, self.rate)

    def redraw(self) -> None:
        xs, ys = self.series.arrays(self.config.max_plot_points)
        self.curve.setData(xs, ys)
        self.crosshair.set_series(xs, ys)

        self.tile_rate.set_value(f"{self.rate:.2f}" if self.rate else "--")
        self.tile_total.set_value(f"{self.total_steps:,}")

        remaining = self.target_steps - self.total_steps
        if self.rate > 0 and remaining > 0:
            self.tile_eta.set_value(self._duration(remaining / self.rate))
        else:
            self.tile_eta.set_value("--")

    @staticmethod
    def _duration(seconds: float) -> str:
        """Format a coarse human duration"""
        if seconds < 90:
            return f"{seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 90:
            return f"{minutes:.0f}m"
        hours = minutes / 60
        if hours < 48:
            return f"{hours:.1f}h"
        return f"{hours / 24:.1f}d"

    def clear(self) -> None:
        self.series.clear()
        self.recent.clear()
        self.total_steps = 0
        self.target_steps = 0
        self.first_seen = None
        self.rate = 0.0
