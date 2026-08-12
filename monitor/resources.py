"""
Bundled resources

Paths are resolved relative to the repo root so the icon is found however the
monitor is launched (``python -m monitor`` from anywhere, or the web server from
its own package).
"""

import os
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")

ICON_PNG = os.path.join(ASSETS_DIR, "icon.png")
ICON_ICO = os.path.join(ASSETS_DIR, "icon.ico")


def icon_path() -> Optional[str]:
    """
    Best available application icon file

    Prefers the .ico on Windows, which embeds several sizes so the taskbar and
    title bar each pick a crisp one instead of downscaling a single large image.

    Returns:
        Path to an icon file, or None if neither exists
    """
    import sys

    candidates = [ICON_ICO, ICON_PNG] if sys.platform == "win32" else [ICON_PNG, ICON_ICO]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def app_icon():
    """
    Load the application icon as a QIcon

    Returns:
        QIcon, empty if no icon file is present
    """
    from PySide6.QtGui import QIcon

    path = icon_path()
    return QIcon(path) if path else QIcon()
