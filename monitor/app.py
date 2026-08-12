"""
Monitor application shell

Hosts the panels in a dock area, tails the selected run, and repaints on a single
shared timer.

Three properties keep the monitor from competing with training for resources:

- One timer drives every repaint, at 10 Hz by default rather than 60. The trainer
  emits a few events per second, so this is already oversampled.
- Panels that are not visible skip their repaint entirely and stay dirty until
  shown, so a hidden tab costs nothing.
- Series are bounded ring buffers and are decimated before drawing.
"""

import os
import sys
from typing import List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QComboBox, QLabel, QMainWindow, QMessageBox, QStatusBar,
    QTabWidget, QToolBar,
)

from pyqtgraph.dockarea import Dock, DockArea

from . import layout as layout_store
from . import theme
from .bus import EventBus
from .config import MonitorConfig
from .panels import build_panels

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.telemetry import list_runs  # noqa: E402


class MonitorWindow(QMainWindow):
    """Main window: toolbar, tabbed dock areas, status bar"""

    def __init__(self, config: MonitorConfig):
        super().__init__()
        self.config = config
        self.panels = build_panels(config)

        self.setWindowTitle("Gambit Monitor")
        self.resize(1500, 950)
        self.setStyleSheet(
            f"QMainWindow {{ background: {theme.PAGE}; }}"
            f"QToolBar {{ background: {theme.PAGE}; border: none; spacing: 8px; padding: 6px; }}"
            f"QTabWidget::pane {{ border: none; }}"
            f"QTabBar::tab {{ background: {theme.PAGE}; color: {theme.MUTED}; "
            f"padding: 6px 14px; border: none; }}"
            f"QTabBar::tab:selected {{ color: {theme.INK}; "
            f"border-bottom: 2px solid {theme.SERIES[0]}; }}"
            f"QComboBox {{ background: {theme.SURFACE}; color: {theme.INK}; "
            f"border: 1px solid {theme.AXIS}; border-radius: 3px; padding: 4px 8px; }}"
            f"QLabel {{ color: {theme.INK_2}; }}"
            f"QStatusBar {{ background: {theme.PAGE}; color: {theme.MUTED}; }}"
        )

        self.bus = EventBus(poll_ms=config.poll_ms, parent=self)
        self.bus.batch.connect(self._on_batch)
        self.bus.restarted.connect(self._on_restart)

        self._build_toolbar()
        self._build_tabs()
        self._build_status()

        # One timer for every panel
        self.redraw_timer = QTimer(self)
        self.redraw_timer.timeout.connect(self._redraw)
        self.redraw_timer.start(config.redraw_ms)

        # Rescan for new runs periodically so a session started after the
        # monitor is already open gets picked up
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(lambda: self._scan_runs(auto=True))
        self.scan_timer.start(3000)

        self._scan_runs(auto=True)

    # --- Construction ---

    def _build_toolbar(self) -> None:
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)

        bar.addWidget(QLabel("Run "))
        self.run_picker = QComboBox()
        self.run_picker.setMinimumWidth(320)
        self.run_picker.currentIndexChanged.connect(self._on_run_picked)
        bar.addWidget(self.run_picker)

        bar.addSeparator()

        self.follow_action = QAction("Follow latest", self)
        self.follow_action.setCheckable(True)
        self.follow_action.setChecked(self.config.follow_latest)
        self.follow_action.toggled.connect(self._on_follow_toggled)
        bar.addAction(self.follow_action)

        bar.addSeparator()

        save_layout = QAction("Save layout", self)
        save_layout.triggered.connect(self._save_layout)
        bar.addAction(save_layout)

        reset_layout = QAction("Reset layout", self)
        reset_layout.triggered.connect(self._reset_layout)
        bar.addAction(reset_layout)

    def _build_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        self.live_area = DockArea()
        self.tabs.addTab(self.live_area, "Live")

        # Two-column grid, built in two passes. Adding a dock 'bottom' of one
        # that already has a neighbour splits only that neighbour's cell, which
        # nests splitters instead of making a row -- so stack the full-width row
        # anchors first, then place the second column beside each of them.
        def make_dock(panel) -> Dock:
            width, height = panel.SIZE
            # autoOrientation would rotate the title to vertical text down the
            # side of a tall dock, which wastes width and reads poorly
            dock = Dock(panel.TITLE, size=(width, height), closable=False,
                        autoOrientation=False)
            dock.setOrientation("horizontal", force=True)
            dock.addWidget(panel.ensure_widget())
            return dock

        anchors: List[Dock] = []
        for index in range(0, len(self.panels), 2):
            dock = make_dock(self.panels[index])
            if anchors:
                self.live_area.addDock(dock, "bottom", anchors[-1])
            else:
                self.live_area.addDock(dock)
            anchors.append(dock)

        for row, index in enumerate(range(1, len(self.panels), 2)):
            self.live_area.addDock(make_dock(self.panels[index]), "right", anchors[row])

        layout_store.load(self.live_area)

    def _build_status(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("No run selected")
        self.status.addWidget(self.status_label)

    # --- Run selection ---

    def _scan_runs(self, auto: bool = False) -> None:
        """Refresh the run list, optionally following the newest run"""
        runs = list_runs(self.config.runs_dir)
        existing = [self.run_picker.itemData(i) for i in range(self.run_picker.count())]
        paths = [run.events_path for run in runs]

        if paths != existing:
            blocked = self.run_picker.blockSignals(True)
            current = self.run_picker.currentData()
            self.run_picker.clear()
            for run in runs:
                meta = run.read_meta()
                marker = " (live)" if meta.get("active") else ""
                self.run_picker.addItem(run.run_id + marker, run.events_path)
            self.run_picker.blockSignals(blocked)

            if current in paths:
                self.run_picker.setCurrentIndex(paths.index(current))

        if auto and self.follow_action.isChecked() and paths:
            if self.run_picker.currentIndex() != 0:
                self.run_picker.setCurrentIndex(0)
            self.bus.set_source(paths[0])
        elif self.bus.path is None and paths:
            self.bus.set_source(self.run_picker.currentData())

    def _on_run_picked(self, index: int) -> None:
        if index < 0:
            return
        path = self.run_picker.itemData(index)
        if path:
            self.bus.set_source(path)

    def _on_follow_toggled(self, checked: bool) -> None:
        self.config.follow_latest = checked
        if checked:
            self._scan_runs(auto=True)

    # --- Event flow ---

    def _on_batch(self, events: List[dict]) -> None:
        for event in events:
            for panel in self.panels:
                panel.handle(event)

    def _on_restart(self) -> None:
        for panel in self.panels:
            panel.reset()

    def _redraw(self) -> None:
        for panel in self.panels:
            panel.maybe_redraw()

        run = os.path.basename(os.path.dirname(self.bus.path)) if self.bus.path else "none"
        self.status_label.setText(
            f"run: {run}    events: {self.bus.total_events:,}    "
            f"{self.bus.events_per_sec:.1f}/s    redraw: {self.config.redraw_hz:g} Hz"
        )

    # --- Layout ---

    def _save_layout(self) -> None:
        if layout_store.save(self.live_area):
            self.status.showMessage("Layout saved", 2000)
        else:
            self.status.showMessage("Could not save layout", 3000)

    def _reset_layout(self) -> None:
        layout_store.clear()
        QMessageBox.information(
            self, "Layout reset",
            "The saved layout has been cleared. Restart the monitor to see the "
            "default arrangement."
        )

    def closeEvent(self, event) -> None:
        self.bus.stop()
        super().closeEvent(event)


def main(argv=None) -> int:
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Gambit training monitor")
    parser.add_argument("--runs-dir", default=None, help="Directory holding run directories")
    parser.add_argument("--redraw-hz", type=float, default=None, help="Panel repaint rate")
    parser.add_argument("--config", default=None, help="Path to monitor.json")
    args = parser.parse_args(argv)

    config = MonitorConfig.load(args.config)
    if args.runs_dir:
        config.runs_dir = args.runs_dir
    if args.redraw_hz:
        config.redraw_hz = args.redraw_hz

    theme.configure()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Gambit Monitor")

    window = MonitorWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
