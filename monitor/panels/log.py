"""
Event log panel

A readable tail of the notable events: episode outcomes, wins, connection
changes and session boundaries. Step events are excluded -- at 15 steps/sec they
would bury everything worth reading.

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
    EVENT_TYPES = frozenset({"episode_end", "game", "comm", "session_start",
                             "session_end", "rollout", "log"})
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
            return f"SESSION END  status={data.get('status')}"

        if kind == "episode_end":
            outcome = str(data.get("outcome", "?")).upper()
            return (f"episode {data.get('episode')} {outcome}  "
                    f"steps={data.get('steps')} "
                    f"reward={data.get('reward'):+.1f} "
                    f"chips={data.get('chips')}/{data.get('blind_chips')}")

        if kind == "game":
            if data.get("event") == "episode_win":
                return (f"WIN  {data.get('final_chips')} chips in "
                        f"{data.get('hands_used')} hands")
            if data.get("event") == "win_rate":
                return (f"win rate {data.get('win_rate')}%  "
                        f"({data.get('wins')}/{data.get('episode')})")
            return f"game: {data.get('event')}"

        if kind == "comm":
            return f"comm: {data.get('event')}"

        if kind == "rollout":
            parts = [f"rollout {data.get('iteration')}",
                     f"steps={data.get('timesteps'):,}"]
            if data.get("ep_rew_mean") is not None:
                parts.append(f"ep_rew_mean={data['ep_rew_mean']:.2f}")
            if data.get("train/loss") is not None:
                parts.append(f"loss={data['train/loss']:.4f}")
            return "  ".join(parts)

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
