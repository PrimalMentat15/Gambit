"""
Action distribution panel

How often the policy picks each action, plus how often actions had to be retried.

Retries are the interesting signal: a rising retry count means the agent is
choosing actions the game then rejects, which is a masking problem rather than a
learning one.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, StatTile


class ActionDistPanel(Panel):
    """Action counts with a retry tally"""

    NAME = "actions"
    TITLE = "Action distribution"
    EVENT_TYPES = frozenset({"step"})
    SIZE = (420, 230)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.counts: Dict[int, int] = {}
        self.retries = 0
        self.steps = 0
        self.bars = None
        self.labels: list = []

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_steps = StatTile("steps")
        self.tile_retries = StatTile("retries")
        row.addWidget(self.tile_steps)
        row.addWidget(self.tile_retries)

        self.plot = theme.make_plot(y_label="count")
        item = self.plot.getPlotItem()
        item.showGrid(x=False, y=True, alpha=0.12)

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        action = data.get("action")
        if action is not None:
            self.counts[action] = self.counts.get(action, 0) + 1
        self.steps += 1
        if data.get("retry_count"):
            self.retries += 1

    def redraw(self) -> None:
        self.tile_steps.set_value(f"{self.steps:,}")
        self.tile_retries.set_value(
            f"{self.retries:,}",
            theme.STATUS["warning"] if self.retries else None,
        )

        item = self.plot.getPlotItem()
        if self.bars is not None:
            item.removeItem(self.bars)
            self.bars = None
        for label in self.labels:
            item.removeItem(label)
        self.labels.clear()

        if not self.counts:
            return

        actions = sorted(self.counts)
        positions = list(range(len(actions)))
        values = [self.counts[a] for a in actions]
        colors = [theme.ACTION_COLORS.get(a, theme.SERIES[7]) for a in actions]

        self.bars = pg.BarGraphItem(
            x=positions,
            height=values,
            # Leaves a gap between adjacent bars instead of drawing a border
            # around each one to separate them
            width=0.62,
            brushes=[pg.mkBrush(c) for c in colors],
            pen=pg.mkPen(None),
        )
        item.addItem(self.bars)

        ticks = [
            (pos, theme.ACTION_NAMES.get(a, str(a)).replace("_HAND", ""))
            for pos, a in zip(positions, actions)
        ]
        item.getAxis("bottom").setTicks([ticks])

        tallest = max(values)
        for pos, value in zip(positions, values):
            label = pg.TextItem(f"{value:,}", color=theme.INK_2, anchor=(0.5, 1))
            label.setPos(pos, value + tallest * 0.04)
            item.addItem(label)
            self.labels.append(label)

        item.setYRange(0, tallest * 1.2)

    def clear(self) -> None:
        self.counts.clear()
        self.retries = 0
        self.steps = 0
