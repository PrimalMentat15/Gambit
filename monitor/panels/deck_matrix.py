"""Deck-matrix eval panel (P7).

The per-deck ante-8 argmax win rate from the `deck_matrix_eval` event, which
the trainer emits at the milestone cadence once the deck curriculum is live.
One bar per *active* deck, coloured by win rate, so the question this panel
answers at a glance is "which deck is the mixture lagging on" -- the input the
prioritized sampler already leans on, made visible.

Deck-only for now (the P7 grid trains stakes as a later run). When a stake
dimension lands the event gains per-(deck, stake) cells; this panel reads
`win_rate_by_deck` and ignores anything else, so it keeps working until a
proper 2-D heatmap replaces it.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, StatTile

# Short deck labels in encoding.Deck order (index == bar position). Hardcoded
# rather than imported so the panel renders even if the trainer package is not
# importable in the monitor's environment -- the ordering is a frozen wire
# format (config.rs) and will not change.
DECK_LABELS = [
    "Red", "Blue", "Yellow", "Green", "Black", "Magic", "Nebula", "Ghost",
    "Aband", "Check", "Zodiac", "Paint", "Anagl", "Plasma", "Errat",
]


class DeckMatrixPanel(Panel):
    """Per-deck win rate as a coloured bar chart, latest eval."""

    NAME = "deck_matrix"
    TITLE = "Deck matrix (ante 8)"
    EVENT_TYPES = frozenset({"deck_matrix_eval"})
    SIZE = (520, 300)
    PAGE = "progress"

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.by_deck: Dict[int, float] = {}
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
        self.tile_live = StatTile("decks live")
        self.tile_mean = StatTile("mean win")
        self.tile_worst = StatTile("worst deck")
        for t in (self.tile_live, self.tile_mean, self.tile_worst):
            row.addWidget(t)

        self.plot = theme.make_plot(y_label="win rate (%)", x_label="deck")
        self.plot.getPlotItem().setYRange(0, 100)
        # The ante-8 target line, so a bar's colour and its height agree.
        self.target = pg.InfiniteLine(
            pos=70, angle=0, pen=pg.mkPen(theme.INK_2, style=Qt.DashLine)
        )
        self.plot.getPlotItem().addItem(self.target)

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        wbd = data.get("win_rate_by_deck")
        if not wbd:
            return
        # Keys are deck-index strings.
        self.by_deck = {int(k): float(v) for k, v in wbd.items()}
        self.latest = data

    def redraw(self) -> None:
        item = self.plot.getPlotItem()
        if self.bars is not None:
            item.removeItem(self.bars)
            self.bars = None
        if not self.by_deck:
            self.tile_live.set_value("--")
            self.tile_mean.set_value("--")
            self.tile_worst.set_value("--")
            return

        decks = sorted(self.by_deck)
        values = [self.by_deck[d] * 100.0 for d in decks]
        brushes = [pg.mkBrush(self._colour(v)) for v in values]

        self.bars = pg.BarGraphItem(
            x=list(range(len(decks))),
            height=values,
            width=0.7,
            brushes=brushes,
            pen=pg.mkPen(None),
        )
        item.addItem(self.bars)
        item.getAxis("bottom").setTicks(
            [[(pos, DECK_LABELS[d] if d < len(DECK_LABELS) else str(d))
              for pos, d in enumerate(decks)]]
        )

        self.tile_live.set_value(str(len(decks)))
        mean = sum(values) / len(values)
        self.tile_mean.set_value(f"{mean:.1f}%",
                                 theme.STATUS[self._role(mean)])
        worst_d = min(decks, key=lambda d: self.by_deck[d])
        worst_v = self.by_deck[worst_d] * 100.0
        label = DECK_LABELS[worst_d] if worst_d < len(DECK_LABELS) else str(worst_d)
        self.tile_worst.set_value(f"{label} {worst_v:.0f}%",
                                  theme.STATUS[self._role(worst_v)])

    @staticmethod
    def _role(percent: float) -> str:
        return ("good" if percent >= 60
                else "warning" if percent >= 40
                else "serious" if percent >= 20
                else "critical")

    def _colour(self, percent: float) -> str:
        return theme.STATUS[self._role(percent)]

    def clear(self) -> None:
        self.by_deck = {}
        self.latest = {}
