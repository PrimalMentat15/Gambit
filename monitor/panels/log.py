"""
Event log panel

A readable tail of the notable events: iteration summaries, evaluations,
promotions, checkpoints and session boundaries. Episode events are excluded --
hundreds finish per second, and they would bury everything worth reading, which
is the same reason the pre-sim version excluded per-step events.

This is also the table-view escape hatch for the dashboard: every value here is
text, so nothing depends on colour or on hovering a mark.
"""

import time
from typing import Any, Dict

from PySide6.QtWidgets import QPlainTextEdit, QWidget

from .. import theme


from .base import Panel


class LogPanel(Panel):
    """Human-readable stream of notable events"""

    NAME = "log"
    TITLE = "Event log"
    EVENT_TYPES = frozenset({"session_start", "session_end", "rollout",
                             "promotion_eval", "curriculum_promotion",
                             "milestone_eval", "checkpoint_saved", "log"})
    SIZE = (560, 220)

    MAX_LINES = 500

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.pending: list = []

    def build(self) -> QWidget:
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(self.MAX_LINES)
        self.view.setStyleSheet(
            f"QPlainTextEdit {{ background: {theme.SURFACE}; color: {theme.INK_2}; "
            f"border: none; font-family: Consolas, monospace; font-size: 11px; }}"
        )
        return self.view

    def on_event(self, event: Dict[str, Any]) -> None:
        line = self._format(event)
        if line:
            stamp = time.strftime("%H:%M:%S", time.localtime(event.get("t", 0)))
            self.pending.append(f"{stamp}  {line}")

    def _format(self, event: Dict[str, Any]) -> str:
        """Render one event as a log line, or empty to skip it"""
        kind = event.get("type")
        data = event.get("data", {})

        if kind == "session_start":
            parts = ["SESSION START"]
            if data.get("run_id"):
                parts.append(f"run={data['run_id']}")
            if data.get("device"):
                parts.append(f"device={data['device']}")
            if data.get("total_timesteps"):
                parts.append(f"target={data['total_timesteps']:,}")
            return "  ".join(parts)

        if kind == "session_end":
            return (f"SESSION END  status={data.get('status')}  "
                    f"step={data.get('step', 0):,}")

        if kind == "rollout":
            parts = [f"iter {data.get('iteration')}",
                     f"step={data.get('global_step', 0):,}",
                     f"sps={data.get('sps', 0):,}"]
            for label, key, fmt in (
                ("ep_ret", "ep_return_mean", "{:.2f}"),
                ("ante", "ep_ante_mean", "{:.2f}"),
                ("win", "ep_win_rate", "{:.2f}"),
                ("loss", "loss", "{:.4f}"),
                ("kl", "approx_kl", "{:.5f}"),
            ):
                if data.get(key) is not None:
                    parts.append(f"{label}={fmt.format(data[key])}")
            if data.get("win_ante") is not None:
                parts.append(f"goal={data['win_ante']}")
            return "  ".join(parts)

        if kind == "promotion_eval":
            return (f"promo eval  ante {data.get('win_ante')}  "
                    f"win={data.get('win_rate', 0) * 100:.1f}% "
                    f"over {data.get('episodes')} eps "
                    f"(need {data.get('threshold', 0) * 100:.0f}%)")

        if kind == "curriculum_promotion":
            return (f"PROMOTION  {data.get('from_ante')} -> {data.get('to_ante')}  "
                    f"win={data.get('win_rate', 0) * 100:.1f}%")

        if kind == "milestone_eval":
            return (f"MILESTONE  ante-8 win={data.get('win_rate', 0) * 100:.1f}%  "
                    f"ante_mean={data.get('ante_mean', 0):.2f}  "
                    f"return={data.get('return_mean', 0):.2f}  "
                    f"({data.get('episodes')} eps)")

        if kind == "checkpoint_saved":
            return f"checkpoint  step={data.get('step', 0):,}  {data.get('path')}"

        if kind == "log":
            return str(data.get("message", ""))

        return ""

    def redraw(self) -> None:
        if not self.pending:
            return
        # One appendPlainText per batch rather than per line keeps the widget
        # from relaying out repeatedly during a burst
        self.view.appendPlainText("\n".join(self.pending))
        self.pending.clear()

    def clear(self) -> None:
        self.pending.clear()
        if self.widget is not None:
            self.view.clear()
