"""
Game state panel

What the agent is looking at right now: chips against the blind requirement,
resources left, and the last action taken.

All scalars, so this is stat tiles and a progress meter rather than charts -- a
plot of a single current value would be the wrong form.
"""

from typing import Any, Dict

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, StatTile


class GameStatePanel(Panel):
    """Live snapshot of the current hand"""

    NAME = "gamestate"
    TITLE = "Game state"
    EVENT_TYPES = frozenset({"step"})
    SIZE = (420, 230)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.state: Dict[str, Any] = {}

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.progress_caption = QLabel("CHIPS VS BLIND")
        self.progress_caption.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(22)
        self._style_progress(theme.SERIES[0])

        tiles = QWidget()
        grid = QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.tile_hands = StatTile("hands left")
        self.tile_discards = StatTile("discards left")
        self.tile_hand_type = StatTile("hand type")
        self.tile_action = StatTile("last action")

        grid.addWidget(self.tile_hands, 0, 0)
        grid.addWidget(self.tile_discards, 0, 1)
        grid.addWidget(self.tile_hand_type, 1, 0)
        grid.addWidget(self.tile_action, 1, 1)

        layout.addWidget(self.progress_caption)
        layout.addWidget(self.progress)
        layout.addWidget(tiles, 1)
        return container

    def _style_progress(self, color: str) -> None:
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {theme.PAGE}; border: none; "
            f"border-radius: 4px; color: {theme.INK}; font-size: 11px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )

    def on_event(self, event: Dict[str, Any]) -> None:
        self.state = event["data"]

    def redraw(self) -> None:
        if not self.state:
            return

        chips = self.state.get("chips", 0) or 0
        blind = self.state.get("blind_chips", 0) or 0

        percent = min(100, int(chips / blind * 100)) if blind else 0
        self.progress.setValue(percent)
        self.progress.setFormat(f"{chips:,} / {blind:,}  ({percent}%)")
        # Sequential single hue for magnitude; status green only once the blind
        # is actually beaten, where it means a state rather than a quantity
        self._style_progress(theme.STATUS["good"] if percent >= 100 else theme.SERIES[0])

        self.tile_hands.set_value(str(self.state.get("hands_left", "--")))
        self.tile_discards.set_value(str(self.state.get("discards_left", "--")))

        hand_type = self.state.get("hand_type") or "None"
        self.tile_hand_type.set_value(hand_type if hand_type != "None" else "--")

        action = self.state.get("action")
        name = theme.ACTION_NAMES.get(action, str(action) if action else "--")
        self.tile_action.set_value(
            name.replace("_", " ").title(),
            theme.ACTION_COLORS.get(action),
        )

    def clear(self) -> None:
        self.state = {}
