"""
Dock layout persistence

Panels live in a pyqtgraph DockArea, so they can be dragged, resized, stacked and
torn off at runtime. Saving the arrangement is what makes that customisation
stick between sessions.
"""

import json
import os
from typing import Any, Optional

LAYOUT_FILENAME = ".monitor_layout.json"


def path_for(page: str) -> str:
    """
    Layout file for one Live page

    Each page owns its arrangement: a state saved from one page lists only that
    page's docks, so restoring it onto another would drop every panel the file
    does not mention.
    """
    return f".monitor_layout_{page}.json"


def save(area, path: str = LAYOUT_FILENAME) -> bool:
    """
    Write the dock arrangement to disk

    Returns:
        True if the layout was written
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(area.saveState(), handle, indent=2)
        return True
    except (OSError, TypeError, ValueError):
        return False


def load(area, path: str = LAYOUT_FILENAME) -> bool:
    """
    Restore a previously saved arrangement

    A layout saved before a panel was added or removed will not apply cleanly;
    in that case the default arrangement is kept rather than raising.

    Returns:
        True if the layout was applied
    """
    if not os.path.isfile(path):
        return False

    try:
        with open(path, "r", encoding="utf-8") as handle:
            state: Optional[Any] = json.load(handle)
        if not state:
            return False
        area.restoreState(state)
        return True
    except Exception:
        return False


def clear(path: str = LAYOUT_FILENAME) -> None:
    """Delete the saved layout so the next start uses defaults"""
    try:
        os.remove(path)
    except OSError:
        pass
