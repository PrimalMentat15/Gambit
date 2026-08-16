"""
Action distribution panel

How the policy spends its actions across the game's 14 action types, as a share
of the most recent rollout.

A share rather than a running total: totals only ever grow, so after an hour
every bar is dominated by whatever the policy did early on and the chart stops
responding to what it is doing now. The share answers the question actually
being asked -- is it stuck rerolling, is it skipping every pack -- and the step
count that produced it is on a tile so the share is never read without its n.

Bars are coloured by game phase, not per action: 14 types against 8 categorical
slots would collide, and phase is the grouping worth seeing anyway.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, StatTile


class ActionDistPanel(Panel):
    """Per-rollout action-type shares"""

    NAME = "actions"
    TITLE = "Action distribution"
    EVENT_TYPES = frozenset({"rollout"})
    SIZE = (560, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.counts: list = []
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
        self.tile_steps = StatTile("actions this rollout")
        self.tile_types = StatTile("types used")
        row.addWidget(self.tile_steps)
        row.addWidget(self.tile_types)

        self.plot = theme.make_plot(y_label="share of rollout (%)")
        item = self.plot.getPlotItem()
        item.showGrid(x=False, y=True, alpha=0.12)

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        counts = event["data"].get("action_counts")
        if not counts:
            return
        self.counts = counts
        self.steps = sum(counts)

    def redraw(self) -> None:
        item = self.plot.getPlotItem()
        if self.bars is not None:
            item.removeItem(self.bars)
            self.bars = None
        for label in self.labels:
            item.removeItem(label)
        self.labels.clear()

        if not self.steps:
            self.tile_steps.set_value("--")
            self.tile_types.set_value("--")
            return

        # Reserved slots the policy never selects would be permanent empty
        # bars, so only types that actually occurred get an axis position.
        used = [(a, c) for a, c in enumerate(self.counts) if c]
        self.tile_steps.set_value(f"{self.steps:,}")
        self.tile_types.set_value(f"{len(used)}")

        positions = list(range(len(used)))
        shares = [c / self.steps * 100.0 for _a, c in used]
        colors = [
            theme.ACTION_COLORS.get(a, theme.SERIES[7]) for a, _c in used
        ]

        self.bars = pg.BarGraphItem(
            x=positions,
            height=shares,
            # Leaves a gap between adjacent bars instead of drawing a border
            # around each one to separate them
            width=0.62,
            brushes=[pg.mkBrush(c) for c in colors],
            pen=pg.mkPen(None),
        )
        item.addItem(self.bars)

        # Short labels: 13 full action names side by side overlap into an
        # unreadable band at any width the panel actually gets.
        ticks = [
            (pos, theme.ACTION_SHORT.get(a, str(a)))
            for pos, (a, _c) in zip(positions, used)
        ]
        item.getAxis("bottom").setTicks([ticks])

        tallest = max(shares)
        for pos, share in zip(positions, shares):
            # Every bar here has a nonzero count, so a label must never read
            # "0": a rare action rounds to zero percent and would look like it
            # never fired. Sub-1% shares keep a decimal instead.
            text = f"{share:.1f}" if share < 1 else f"{share:.0f}"
            label = pg.TextItem(text, color=theme.INK_2, anchor=(0.5, 1))
            label.setPos(pos, share + tallest * 0.04)
            item.addItem(label)
            self.labels.append(label)

        # Headroom for the value labels, which sit above their bars
        item.setYRange(0, tallest * 1.32)

    def clear(self) -> None:
        self.counts = []
        self.steps = 0
