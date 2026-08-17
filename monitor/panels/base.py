"""
Panel framework

A panel is a self-contained view over the event stream. Adding a chart to the
dashboard means adding one file in this package -- the registry discovers it, the
shell docks it, and the config can enable or disable it by name.

Two rules make the dashboard cheap to run:

- ``on_event`` accumulates only; it never paints. Events arrive whenever the
  trainer emits them.
- ``redraw`` paints, and the shell calls it at a capped rate and only while the
  panel is actually visible. Background tabs cost nothing.
"""

from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import theme


class RingSeries:
    """
    Bounded x/y series

    A multi-hour run at 15 steps/sec produces well over a million points, so
    every series is capped rather than grown without limit.
    """

    def __init__(self, maxlen: int):
        self.x: deque = deque(maxlen=maxlen)
        self.y: deque = deque(maxlen=maxlen)

    def append(self, x: float, y: float) -> None:
        self.x.append(x)
        self.y.append(y)

    def clear(self) -> None:
        self.x.clear()
        self.y.clear()

    def __len__(self) -> int:
        return len(self.x)

    def arrays(self, limit: int = 0) -> Tuple[List[float], List[float]]:
        """Return the series as lists, decimated to at most ``limit`` points"""
        xs, ys = list(self.x), list(self.y)
        if limit:
            return theme.decimate(xs, ys, limit)
        return xs, ys


class StatTile(QWidget):
    """
    A single labelled number

    Used where the story is one value: a bar chart with one bar, or a plot of a
    scalar, would both be the wrong form.
    """

    def __init__(self, caption: str, value: str = "--", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        # Qt style sheets support neither text-transform nor letter-spacing, so
        # the casing is applied here rather than silently ignored in CSS
        self.caption = QLabel(caption.upper())
        self.caption.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")

        self.value = QLabel(value)
        # Proportional figures deliberately: tabular-nums makes large standalone
        # numbers look loose. Tabular is for columns that align vertically.
        self.value.setStyleSheet(
            f"color: {theme.INK}; font-size: 21px; font-weight: 600;"
        )

        layout.addWidget(self.caption)
        layout.addWidget(self.value)
        layout.addStretch(1)

        self.setStyleSheet(
            f"background: {theme.SURFACE}; border-radius: 4px;"
        )

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        """Update the number, optionally tinting it with a status colour"""
        self.value.setText(text)
        self.value.setStyleSheet(
            f"color: {color or theme.INK}; font-size: 21px; font-weight: 600;"
        )


class Panel(QObject):
    """
    Base class for all panels

    Subclasses set the class attributes and implement build/on_event/redraw.
    """

    NAME: str = "panel"
    TITLE: str = "Panel"
    EVENT_TYPES: FrozenSet[str] = frozenset()  # empty means every type
    SIZE: Tuple[int, int] = (420, 260)
    # Which Live sub-tab this panel belongs to. Nine panels on one screen means
    # every plot is a few hundred pixels tall; splitting by the question being
    # asked ("is it learning" vs "is the machinery healthy") keeps each page
    # readable. "progress" and "diagnostics" are the two built pages.
    PAGE: str = "progress"

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.widget: Optional[QWidget] = None
        self._dirty = True

    # --- Subclass interface ---

    def build(self) -> QWidget:
        """Create and return this panel's widget"""
        raise NotImplementedError

    def on_event(self, event: Dict[str, Any]) -> None:
        """Accumulate state from one event. Must not paint."""

    def redraw(self) -> None:
        """Paint accumulated state. Called at a capped rate while visible."""

    def clear(self) -> None:
        """Drop all accumulated state, e.g. when the run changes"""

    # --- Framework ---

    def ensure_widget(self) -> QWidget:
        """Build the widget on first use"""
        if self.widget is None:
            self.widget = self.build()
        return self.widget

    def handle(self, event: Dict[str, Any]) -> None:
        """Filter by event type and accumulate"""
        if self.EVENT_TYPES and event.get("type") not in self.EVENT_TYPES:
            return
        self.on_event(event)
        self._dirty = True

    def maybe_redraw(self, force: bool = False) -> bool:
        """
        Repaint if there is anything new and the panel can be seen

        Returns:
            True if a repaint happened
        """
        if not self._dirty and not force:
            return False
        if not force and (self.widget is None or not self.widget.isVisible()):
            # Stay dirty so the panel catches up when its tab is selected
            return False

        self.redraw()
        self._dirty = False
        return True

    def reset(self) -> None:
        """Clear state and mark for repaint"""
        self.clear()
        self._dirty = True
