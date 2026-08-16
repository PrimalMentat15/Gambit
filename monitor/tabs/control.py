"""
Control tab

Launch a run, watch process health, and read the trainer's output.

The trainer's stdout is tailed from a file rather than piped, so this view keeps
working across a monitor restart and shows the whole session rather than only
what arrived since the panel opened.
"""

import os
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import theme


class ControlTab(QWidget):
    """Run launcher and process status"""

    def __init__(self, supervisor, config, parent=None):
        super().__init__(parent)
        self.supervisor = supervisor
        self.config = config
        self._stdout_path: Optional[str] = None
        self._stdout_offset = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(self._build_left(), 0)
        layout.addWidget(self._build_right(), 1)

        supervisor.changed.connect(self.refresh)
        supervisor.message.connect(self._append_message)

        self.tail_timer = QTimer(self)
        self.tail_timer.timeout.connect(self._tail_stdout)
        self.tail_timer.start(500)

        self.refresh()

    def _build_left(self) -> QWidget:
        box = QWidget()
        box.setFixedWidth(340)
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)

        launch = QGroupBox("Launch")
        launch.setStyleSheet(self._group_style())
        form = QFormLayout(launch)

        self.run_name = QLineEdit("run")
        self.config_path = QLineEdit(self.config.config_path)

        # 0 means "use whatever the config says", which is the normal case --
        # total_timesteps is part of the run's recipe, not a monitor setting.
        self.timesteps = QSpinBox()
        self.timesteps.setRange(0, 2_000_000_000)
        self.timesteps.setSingleStep(1_000_000)
        self.timesteps.setValue(0)
        self.timesteps.setSpecialValueText("from config")
        self.timesteps.setGroupSeparatorShown(True)

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])

        self.resume = QLineEdit()
        self.resume.setPlaceholderText("path to ckpt_*.pt (optional)")

        form.addRow("Run name", self.run_name)
        form.addRow("Config", self.config_path)
        form.addRow("Timesteps", self.timesteps)
        form.addRow("Device", self.device)
        form.addRow("Resume", self.resume)

        self.start_button = QPushButton("Start run")
        self.start_button.setStyleSheet(self._button_style(theme.SERIES[0]))
        self.start_button.clicked.connect(self._on_start)
        form.addRow("", self.start_button)

        status = QGroupBox("Status")
        status.setStyleSheet(self._group_style())
        status_layout = QVBoxLayout(status)
        self.status_label = QLabel("no run")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.INK_2};")
        status_layout.addWidget(self.status_label)

        note = QLabel(
            "Closing this window leaves training running.\n"
            "Stop checkpoints and exits at the next iteration;\n"
            "Kill is immediate and loses the current iteration."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")
        status_layout.addWidget(note)

        column.addWidget(launch)
        column.addWidget(status)
        column.addStretch(1)
        return box

    def _build_right(self) -> QWidget:
        box = QGroupBox("Trainer output")
        box.setStyleSheet(self._group_style())
        layout = QVBoxLayout(box)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(3000)
        self.output.setStyleSheet(
            f"QPlainTextEdit {{ background: {theme.SURFACE}; color: {theme.INK_2}; "
            f"border: none; font-family: Consolas, monospace; font-size: 11px; }}"
        )
        layout.addWidget(self.output)
        return box

    @staticmethod
    def _group_style() -> str:
        return (
            f"QGroupBox {{ color: {theme.MUTED}; border: 1px solid {theme.AXIS}; "
            f"border-radius: 4px; margin-top: 10px; padding-top: 10px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}"
            f"QLineEdit, QSpinBox, QComboBox {{ background: {theme.SURFACE}; "
            f"color: {theme.INK}; border: 1px solid {theme.AXIS}; "
            f"border-radius: 3px; padding: 4px; }}"
            f"QCheckBox {{ color: {theme.INK_2}; }}"
        )

    @staticmethod
    def _button_style(color: str) -> str:
        return (
            f"QPushButton {{ background: {color}; color: #ffffff; border: none; "
            f"border-radius: 3px; padding: 7px 14px; font-weight: 600; }}"
            f"QPushButton:disabled {{ background: {theme.AXIS}; color: {theme.MUTED}; }}"
        )

    def _on_start(self) -> None:
        device = self.device.currentText()
        self.supervisor.start(
            run_name=self.run_name.text().strip() or "run",
            timesteps=self.timesteps.value() or None,
            device=None if device == "auto" else device,
            config_path=self.config_path.text().strip() or None,
            resume=self.resume.text().strip() or None,
        )

    def _append_message(self, text: str) -> None:
        self.output.appendPlainText(f"[monitor] {text}")

    def refresh(self) -> None:
        """Update status text and button availability"""
        self.status_label.setText(self.supervisor.status_text())
        self.start_button.setEnabled(not self.supervisor.running)

        session = self.supervisor.session
        path = os.path.join(session.run_dir, "trainer.out") if session else None
        if path != self._stdout_path:
            self._stdout_path = path
            self._stdout_offset = 0

    def _tail_stdout(self) -> None:
        """Append whatever the trainer has written since the last check"""
        if not self._stdout_path or not os.path.isfile(self._stdout_path):
            return

        try:
            size = os.path.getsize(self._stdout_path)
            if size < self._stdout_offset:
                self._stdout_offset = 0
            if size == self._stdout_offset:
                return
            with open(self._stdout_path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._stdout_offset)
                chunk = handle.read()
                self._stdout_offset = handle.tell()
        except OSError:
            return

        text = chunk.rstrip("\n")
        if text:
            self.output.appendPlainText(text)
