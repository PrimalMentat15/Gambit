"""
Curriculum panel

Where the ante ladder is, how it got there, and how hard the current rung is
proving.

The chart is promotion-eval win rate against training step, one series per rung,
because a single series would splice five different questions into one line and
make the drop at each promotion look like regression. Drawn per rung, the shape
you want is visible directly: each series climbing to the promotion threshold and
handing off to the next.

The threshold rule is drawn once, since it is the same for every rung and is the
only number on the plot that the trainer actually acts on.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class CurriculumPanel(Panel):
    """Promotion-eval win rate per ladder rung, with promotion history"""

    NAME = "curriculum"
    TITLE = "Curriculum ladder"
    EVENT_TYPES = frozenset(
        {"promotion_eval", "curriculum_promotion", "rollout", "session_start"}
    )
    SIZE = (480, 300)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.by_ante: Dict[int, RingSeries] = {}
        self.curves: Dict[int, Any] = {}
        self.promotions: list = []
        self.win_ante = None
        self.threshold = None
        self._threshold_line = None
        self._history_len = 0

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_goal = StatTile("current goal")
        self.tile_promotions = StatTile("promotions")
        row.addWidget(self.tile_goal)
        row.addWidget(self.tile_promotions)

        self.plot = theme.make_plot(y_label="promo win rate (%)", x_label="env step")
        self.plot.getPlotItem().setYRange(0, 100)
        theme.legend(self.plot)
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:.1f}%"
        )

        self.history = QLabel("no promotions yet")
        self.history.setWordWrap(True)
        self.history.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.history)
        return container

    def _series_for(self, ante: int) -> RingSeries:
        """One series per rung, created on first sight of that rung"""
        if ante not in self.by_ante:
            self.by_ante[ante] = RingSeries(self.config.history)
            color = theme.SERIES[len(self.curves) % len(theme.SERIES)]
            self.curves[ante] = self.plot.plot(
                [], [],
                pen=theme.pen(color),
                symbol="o",
                symbolSize=theme.MARKER_SIZE - 4,
                symbolBrush=color,
                symbolPen=pg.mkPen(None),
                name=f"ante {ante}",
            )
        return self.by_ante[ante]

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        kind = event["type"]

        if kind == "session_start":
            self.win_ante = data.get("win_ante")

        elif kind == "rollout":
            if data.get("win_ante") is not None:
                self.win_ante = data["win_ante"]

        elif kind == "promotion_eval":
            ante, step, rate = (
                data.get("win_ante"), data.get("step"), data.get("win_rate")
            )
            if data.get("threshold") is not None:
                self.threshold = data["threshold"]
            if ante is not None and step is not None and rate is not None:
                self._series_for(int(ante)).append(step, rate * 100.0)

        elif kind == "curriculum_promotion":
            self.promotions.append(data)

    def redraw(self) -> None:
        for ante, series in self.by_ante.items():
            xs, ys = series.arrays(self.config.max_plot_points)
            self.curves[ante].setData(xs, ys)
            self.crosshair.set_series(xs, ys)

        if self.threshold is not None and self._threshold_line is None:
            self._threshold_line = pg.InfiniteLine(
                pos=self.threshold * 100.0,
                angle=0,
                pen=pg.mkPen(theme.STATUS["good"], width=1, style=Qt.DashLine),
                label=f"promote at {self.threshold * 100:.0f}%",
                labelOpts={"color": theme.MUTED, "position": 0.45},
            )
            self.plot.addItem(self._threshold_line, ignoreBounds=True)

        self.tile_goal.set_value(
            f"ante {self.win_ante}" if self.win_ante is not None else "--"
        )
        self.tile_promotions.set_value(str(len(self.promotions)))

        if len(self.promotions) != self._history_len:
            self._history_len = len(self.promotions)
            lines = [
                f"{p.get('from_ante')}→{p.get('to_ante')} at "
                f"{p.get('step', 0):,} ({p.get('win_rate', 0) * 100:.0f}%)"
                for p in self.promotions[-6:]
            ]
            self.history.setText("   ".join(lines) or "no promotions yet")

    def clear(self) -> None:
        for series in self.by_ante.values():
            series.clear()
        for curve in self.curves.values():
            self.plot.removeItem(curve)
        self.by_ante.clear()
        self.curves.clear()
        self.promotions.clear()
        self.win_ante = None
        self._history_len = 0
        if self._threshold_line is not None:
            self.plot.removeItem(self._threshold_line)
            self._threshold_line = None
