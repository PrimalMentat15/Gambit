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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QLabel, QMainWindow, QMessageBox, QStatusBar,
    QTabWidget, QToolBar,
)

from pyqtgraph.dockarea import Dock, DockArea

from . import layout as layout_store
from . import resources
from . import theme
from .bus import EventBus
from .config import MonitorConfig
from .panels import build_panels
from .supervisor import Supervisor
from .tabs import AnalysisTab, ControlTab

from balatro_train.telemetry import list_runs


class MonitorWindow(QMainWindow):
    """Main window: toolbar, tabbed dock areas, status bar"""

    def __init__(self, config: MonitorConfig):
        super().__init__()
        self.config = config
        self.panels = build_panels(config)

        self.setWindowTitle("Gambit Monitor")
        self.setWindowIcon(resources.app_icon())
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

        self.supervisor = Supervisor(config, parent=self)
        self.supervisor.attach_latest()
        self.supervisor.changed.connect(self._on_supervisor_changed)

        self._build_toolbar()
        self._build_tabs()
        self._build_status()
        self._on_supervisor_changed()

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

        # Stop and Kill live in the toolbar rather than inside a tab: a kill
        # switch you have to go looking for is not a kill switch. Both act on
        # the same targets and differ only in whether the trainer is given the
        # chance to checkpoint.
        self.stop_action = QAction("Stop", self)
        self.stop_action.setShortcut(QKeySequence("Ctrl+."))
        self.stop_action.setToolTip(
            "Checkpoint and exit at the next iteration boundary (Ctrl+.)"
        )
        self.stop_action.triggered.connect(self._on_stop)
        bar.addAction(self.stop_action)

        self.kill_action = QAction("Kill", self)
        self.kill_action.setShortcut(QKeySequence("Ctrl+Shift+."))
        self.kill_action.setToolTip(
            "Terminate the trainer immediately (Ctrl+Shift+.)\n"
            "Hold Shift while clicking to skip the confirmation"
        )
        self.kill_action.triggered.connect(self._on_kill)
        bar.addAction(self.kill_action)

        bar.addSeparator()

        save_layout = QAction("Save layout", self)
        save_layout.triggered.connect(self._save_layout)
        bar.addAction(save_layout)

        reset_layout = QAction("Reset layout", self)
        reset_layout.triggered.connect(self._reset_layout)
        bar.addAction(reset_layout)

    # Live sub-tabs, in display order: (page key, tab label)
    PAGES = (("progress", "Progress"), ("diagnostics", "Diagnostics"))

    def _build_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # One dock area per page. Nine panels in a single area gave every plot
        # ~200px of height on a 1080p screen; split by the question each answers
        # so both pages stay readable, and each keeps its own saved arrangement.
        self.live_areas: dict = {}
        for key, title in self.PAGES:
            area = DockArea()
            panels = [p for p in self.panels if getattr(p, "PAGE", "progress") == key]
            self._fill_area(area, panels)
            layout_store.load(area, layout_store.path_for(key))
            self.live_areas[key] = area
            self.tabs.addTab(area, title)

        # Kept as the primary area: Save/Reset layout and any caller that only
        # knows about "the" dock area act on whichever page is in front.
        self.live_area = self.live_areas[self.PAGES[0][0]]

        self.control_tab = ControlTab(self.supervisor, self.config)
        self.analysis_tab = AnalysisTab(self.config)
        self.tabs.addTab(self.control_tab, "Control")
        self.tabs.addTab(self.analysis_tab, "Analysis")

    @staticmethod
    def _fill_area(area: DockArea, panels: List) -> None:
        """
        Lay panels into a two-column grid

        Built in two passes: adding a dock 'bottom' of one that already has a
        neighbour splits only that neighbour's cell, which nests splitters
        instead of making a row -- so stack the full-width row anchors first,
        then place the second column beside each of them.
        """
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
        for index in range(0, len(panels), 2):
            dock = make_dock(panels[index])
            if anchors:
                area.addDock(dock, "bottom", anchors[-1])
            else:
                area.addDock(dock)
            anchors.append(dock)

        for row, index in enumerate(range(1, len(panels), 2)):
            area.addDock(make_dock(panels[index]), "right", anchors[row])

    def _current_area(self):
        """The dock area of the Live page in front, or None on another tab"""
        widget = self.tabs.currentWidget()
        return widget if widget in self.live_areas.values() else None

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
            # "Follow latest" means the run that is actually TRAINING, not
            # whichever directory was written to most recently. A short-lived
            # debug or test run sorts newest and would otherwise yank the view
            # off a live multi-day run -- and switching back costs a full
            # re-read of its events.jsonl. Fall back to newest overall only
            # when nothing is active.
            target = next((r.events_path for r in runs if r.read_meta().get("active")),
                          paths[0])
            index = paths.index(target)
            if self.run_picker.currentIndex() != index:
                self.run_picker.setCurrentIndex(index)
            self.bus.set_source(target)
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

    # --- Stopping ---

    def _on_supervisor_changed(self) -> None:
        """Enable Stop/Kill only while something is actually running"""
        running = self.supervisor.running
        self.stop_action.setEnabled(running)
        self.kill_action.setEnabled(running)

    def _on_stop(self) -> None:
        if not self.supervisor.stop():
            self.status.showMessage("No run to stop", 3000)
            return
        self.status.showMessage(
            "Stop requested - trainer will checkpoint and exit. "
            "Use Kill if it does not stop.", 8000)

    def _on_kill(self) -> None:
        # Shift skips the prompt: in a genuine emergency, a dialog is friction
        skip_prompt = QApplication.keyboardModifiers() & Qt.ShiftModifier
        if not skip_prompt:
            answer = QMessageBox.warning(
                self, "Kill run",
                "Terminate the trainer and Balatro immediately?\n\n"
                "The in-flight rollout is lost. The last checkpoint and every "
                "event already written are kept.",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        if self.supervisor.kill():
            self.status.showMessage("Run killed", 5000)
        else:
            self.status.showMessage("Some processes could not be killed", 8000)

    # --- Layout ---

    def _save_layout(self) -> None:
        """Save the arrangement of whichever Live page is in front"""
        saved = [
            title for key, title in self.PAGES
            if self.live_areas[key] is self._current_area()
            and layout_store.save(self.live_areas[key], layout_store.path_for(key))
        ]
        if saved:
            self.status.showMessage(f"{saved[0]} layout saved", 2000)
        elif self._current_area() is None:
            self.status.showMessage(
                "Switch to a Live page first — layouts are saved per page", 3000)
        else:
            self.status.showMessage("Could not save layout", 3000)

    def _reset_layout(self) -> None:
        for key, _title in self.PAGES:
            layout_store.clear(layout_store.path_for(key))
        QMessageBox.information(
            self, "Layout reset",
            "Saved layouts for both Live pages have been cleared. Restart the "
            "monitor to see the default arrangement."
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
    app.setWindowIcon(resources.app_icon())

    # Without a distinct AppUserModelID, Windows groups the window under the
    # generic python.exe taskbar entry and shows its icon instead of ours
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("gambit.monitor")
        except Exception:
            pass

    window = MonitorWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
