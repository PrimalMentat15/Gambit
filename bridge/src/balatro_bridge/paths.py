"""Host-side paths into the game's love.filesystem save directory.

The BotRun mod and the bridge tools use this directory as a tiny file-based
mailbox (see mods/BotRun/botrun.lua), so the bridge has to resolve the same
directory the game writes to. Where that is depends on how the game runs:

* **Windows (native)** -- ``%APPDATA%\\Balatro``.
* **macOS** -- ``~/Library/Application Support/Balatro``.
* **Linux under Proton** (Steam appid 2379780) -- inside the Wine prefix, which
  is why this is not simply an XDG path.

``BALATRO_SAVE_DIR`` overrides all of it, for installs that sit somewhere the
defaults do not predict.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

STEAM_APPID = "2379780"


def _default_save_dir() -> Path:
    """Where love.filesystem writes for this platform"""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData/Roaming"
        return base / "Balatro"

    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Balatro"

    # Linux: the game is a Windows build under Proton, so its "AppData" lives
    # in the Steam compatdata Wine prefix rather than anywhere XDG-shaped.
    return (
        Path.home()
        / f".local/share/Steam/steamapps/compatdata/{STEAM_APPID}/pfx/drive_c/users"
        / "steamuser/AppData/Roaming/Balatro"
    )


BALATRO_SAVE_DIR = Path(
    os.environ.get("BALATRO_SAVE_DIR") or _default_save_dir()
)

# BOT RUN button click -> autoplay daemon plays a run.
BOTRUN_REQUEST_FILE = BALATRO_SAVE_DIR / "botrun_request.txt"

# Intentional quit (QUIT button / window close) -> supervising launcher
# stops instead of relaunching.
BOTRUN_USER_QUIT_FILE = BALATRO_SAVE_DIR / "botrun_user_quit.txt"
