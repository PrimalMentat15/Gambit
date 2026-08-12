"""
Replays tab

Browse the saved winning games and step through the actions that produced them.

Reuses ReplaySystem from ai/utils/replay.py rather than re-reading the file
format, so the two stay in agreement.
"""

import os
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import theme
from ..panels.base import StatTile

from ai.utils.replay import ReplaySystem


class ReplaysTab(QWidget):
    """Top saved games and their action sequences"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.replays: List[Dict[str, Any]] = []
        self.system = ReplaySystem()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.refresh)
        reload_button.setStyleSheet(
            f"QPushButton {{ background: {theme.SURFACE}; color: {theme.INK}; "
            f"border: 1px solid {theme.AXIS}; border-radius: 3px; padding: 5px 12px; }}"
        )
        self.count_tile = StatTile("saved replays")
        self.best_tile = StatTile("best chips")
        header.addWidget(self.count_tile)
        header.addWidget(self.best_tile)
        header.addWidget(reload_button)
        header.addStretch(1)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.list_table = self._make_table(["#", "Chips", "Score", "Actions", "When"])
        self.list_table.currentCellChanged.connect(self._on_selected)
        self.action_table = self._make_table(["Step", "Action", "Params"])
        splitter.addWidget(self._wrap("Replays", self.list_table))
        splitter.addWidget(self._wrap("Actions", self.action_table))
        splitter.setSizes([700, 700])
        layout.addWidget(splitter, 1)

        self.empty = QLabel("")
        self.empty.setStyleSheet(f"color: {theme.MUTED};")
        layout.addWidget(self.empty)

        self.refresh()

    def _make_table(self, headers) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setStyleSheet(
            f"QTableWidget {{ background: {theme.SURFACE}; color: {theme.INK_2}; "
            f"border: 1px solid {theme.AXIS}; gridline-color: {theme.GRID}; }}"
            f"QTableWidget::item:selected {{ background: {theme.AXIS}; color: {theme.INK}; }}"
            f"QHeaderView::section {{ background: {theme.PAGE}; color: {theme.MUTED}; "
            f"border: none; padding: 4px; }}"
        )
        return table

    @staticmethod
    def _wrap(title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        label = QLabel(title.upper())
        label.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")
        column.addWidget(label)
        column.addWidget(widget, 1)
        return box

    def refresh(self) -> None:
        """Reload replays.json"""
        path = self.system.REPLAY_FILE_PATH
        self.replays = self.system.get_top_replays(path)

        self.count_tile.set_value(f"{len(self.replays)}")
        best = max((r.get("chips", 0) for r in self.replays), default=0)
        self.best_tile.set_value(f"{best:,}")

        if not self.replays:
            self.empty.setText(
                f"No replays in {os.path.abspath(path)} yet - they are written "
                f"when a run beats the blind."
            )
        else:
            self.empty.setText("")

        self.list_table.setRowCount(len(self.replays))
        for row, replay in enumerate(self.replays):
            stamp = str(replay.get("timestamp", ""))[:19].replace("T", " ")
            values = [
                str(row + 1),
                f"{replay.get('chips', 0):,}",
                f"{replay.get('score', 0):.1f}",
                str(len(replay.get("actions") or [])),
                stamp,
            ]
            for column, value in enumerate(values):
                self.list_table.setItem(row, column, QTableWidgetItem(value))

        if self.replays:
            self.list_table.setCurrentCell(0, 0)

    def _on_selected(self, row: int, _c: int, _pr: int, _pc: int) -> None:
        if row < 0 or row >= len(self.replays):
            self.action_table.setRowCount(0)
            return

        actions = self.replays[row].get("actions") or []
        self.action_table.setRowCount(len(actions))
        for index, action in enumerate(actions):
            action_id = action.get("action")
            name = theme.ACTION_NAMES.get(action_id, str(action_id))
            params = action.get("params")
            self.action_table.setItem(index, 0, QTableWidgetItem(str(index + 1)))

            item = QTableWidgetItem(name)
            color = theme.ACTION_COLORS.get(action_id)
            if color:
                from PySide6.QtGui import QColor
                item.setForeground(QColor(color))
            self.action_table.setItem(index, 1, item)
            self.action_table.setItem(index, 2, QTableWidgetItem(str(params or "")))
