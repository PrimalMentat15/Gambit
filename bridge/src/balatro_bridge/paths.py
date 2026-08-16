"""Host-side paths into the game's love.filesystem save directory.

Balatro under Proton (Steam appid 2379780) writes love.filesystem files to
this Wine-prefix AppData dir; the BotRun mod and the bridge tools use it as
a tiny file-based mailbox (see mods/BotRun/botrun.lua).
"""

from __future__ import annotations

from pathlib import Path

BALATRO_SAVE_DIR = (
    Path.home()
    / ".local/share/Steam/steamapps/compatdata/2379780/pfx/drive_c/users"
    / "steamuser/AppData/Roaming/Balatro"
)

# BOT RUN button click -> autoplay daemon plays a run.
BOTRUN_REQUEST_FILE = BALATRO_SAVE_DIR / "botrun_request.txt"

# Intentional quit (QUIT button / window close) -> supervising launcher
# stops instead of relaunching.
BOTRUN_USER_QUIT_FILE = BALATRO_SAVE_DIR / "botrun_user_quit.txt"
