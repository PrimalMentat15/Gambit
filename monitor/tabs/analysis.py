"""
Analysis tab

Post-run evaluation: pick runs, compare their learning curves, and see which
reward components actually fired.

Reads TensorBoard event files directly, so there is no separate server to start
and no second window to switch to.
"""

import os
from typing import Dict, List

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .. import analysis, theme
from ..panels.base import StatTile

# Curves worth overlaying when comparing runs
COMPARE_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/entropy_loss",
    "train/explained_variance",
    "time/fps",
]


class AnalysisTab(QWidget):
    """Run comparison and reward diagnostics"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.summaries: Dict[str, dict] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1200])
        layout.addWidget(splitter, 1)

        self.refresh_runs()

    # --- Construction ---

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        reload_button = QPushButton("Reload runs")
        reload_button.clicked.connect(self.refresh_runs)
        reload_button.setStyleSheet(self._button_style())

        row.addWidget(QLabel("Metric"))
        self.metric = QComboBox()
        self.metric.addItems(COMPARE_TAGS)
        self.metric.currentTextChanged.connect(self._redraw_compare)
        row.addWidget(self.metric)
        row.addWidget(reload_button)
        row.addStretch(1)
        return bar

    def _build_left(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        label = QLabel("RUNS  (tick to compare)")
        label.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")

        self.run_list = QListWidget()
        self.run_list.setStyleSheet(
            f"QListWidget {{ background: {theme.SURFACE}; color: {theme.INK_2}; "
            f"border: 1px solid {theme.AXIS}; border-radius: 3px; }}"
            f"QListWidget::item:selected {{ background: {theme.AXIS}; color: {theme.INK}; }}"
        )
        self.run_list.itemChanged.connect(self._redraw_compare)
        self.run_list.currentItemChanged.connect(self._on_run_selected)

        column.addWidget(label)
        column.addWidget(self.run_list, 1)
        return box

    def _build_right(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        tiles = QWidget()
        grid = QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.tiles = {}
        for index, (key, caption) in enumerate([
            ("episodes", "episodes"), ("win_rate", "win rate"),
            ("mean_reward", "mean reward"), ("best_reward", "best reward"),
            ("best_chips", "best chips"), ("steps_per_sec", "steps / sec"),
            ("wall_time", "wall time"), ("steps", "steps"),
        ]):
            tile = StatTile(caption)
            self.tiles[key] = tile
            grid.addWidget(tile, index // 4, index % 4)

        self.compare_plot = theme.make_plot(y_label="value", x_label="timestep")
        theme.legend(self.compare_plot)

        self.tables = QSplitter(Qt.Horizontal)
        self.components = self._make_table(["Reward component", "Times fired"])
        self.hands = self._make_table(["Hand type", "Played", "Mean chips"])
        self.tables.addWidget(self._wrap("Reward components", self.components))
        self.tables.addWidget(self._wrap("Hand types", self.hands))

        column.addWidget(tiles)
        column.addWidget(self.compare_plot, 2)
        column.addWidget(self.tables, 1)
        return box

    def _make_table(self, headers: List[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setStyleSheet(
            f"QTableWidget {{ background: {theme.SURFACE}; color: {theme.INK_2}; "
            f"border: 1px solid {theme.AXIS}; gridline-color: {theme.GRID}; }}"
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

    @staticmethod
    def _button_style() -> str:
        return (
            f"QPushButton {{ background: {theme.SURFACE}; color: {theme.INK}; "
            f"border: 1px solid {theme.AXIS}; border-radius: 3px; padding: 5px 12px; }}"
        )

    # --- Data ---

    def refresh_runs(self) -> None:
        """Rescan the runs directory"""
        from ai.telemetry import list_runs

        blocked = self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in list_runs(self.config.runs_dir):
            item = QListWidgetItem(run.run_id)
            item.setData(Qt.UserRole, run.run_dir)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.run_list.addItem(item)
        self.run_list.blockSignals(blocked)

        if self.run_list.count():
            self.run_list.setCurrentRow(0)
            self.run_list.item(0).setCheckState(Qt.Checked)

    def _summary_for(self, run_dir: str) -> dict:
        """Summaries are cached: a long run's event file is expensive to re-read"""
        if run_dir not in self.summaries:
            self.summaries[run_dir] = analysis.summarize_run(run_dir)
        return self.summaries[run_dir]

    def _on_run_selected(self, current, _previous) -> None:
        if current is None:
            return
        run_dir = current.data(Qt.UserRole)
        summary = self._summary_for(run_dir)

        for key, value in analysis.headline(summary).items():
            if key in self.tiles:
                self.tiles[key].set_value(value)

        rows = sorted(summary["components"].items(), key=lambda kv: -kv[1])
        self.components.setRowCount(len(rows))
        for row, (label, count) in enumerate(rows):
            self.components.setItem(row, 0, QTableWidgetItem(label))
            self.components.setItem(row, 1, QTableWidgetItem(f"{count:,}"))

        hands = sorted(summary["hand_types"].items(), key=lambda kv: -kv[1]["count"])
        self.hands.setRowCount(len(hands))
        for row, (name, entry) in enumerate(hands):
            mean = entry["chips"] / entry["count"] if entry["count"] else 0
            self.hands.setItem(row, 0, QTableWidgetItem(name))
            self.hands.setItem(row, 1, QTableWidgetItem(f"{entry['count']:,}"))
            self.hands.setItem(row, 2, QTableWidgetItem(f"{mean:.0f}"))

    def _checked_runs(self) -> List[tuple]:
        out = []
        for row in range(self.run_list.count()):
            item = self.run_list.item(row)
            if item.checkState() == Qt.Checked:
                out.append((item.text(), item.data(Qt.UserRole)))
        return out

    def _redraw_compare(self, *_args) -> None:
        """Overlay the chosen metric across every ticked run"""
        item = self.compare_plot.getPlotItem()
        item.clear()
        if item.legend is not None:
            item.legend.clear()

        tag = self.metric.currentText()
        # Colour by position in the ticked list, capped at the palette length --
        # a ninth run folds onto the last slot rather than inventing a hue
        for index, (name, run_dir) in enumerate(self._checked_runs()):
            scalars = analysis.load_scalars(run_dir)
            points = scalars.get(tag)
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = theme.SERIES[min(index, len(theme.SERIES) - 1)]
            item.plot(xs, ys, pen=theme.pen(color), name=name)
