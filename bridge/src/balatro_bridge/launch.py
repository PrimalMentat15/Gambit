"""Launch Balatro (Steam/Proton) with the balatrobot API, vanilla mods only.

This is a thin wrapper around the upstream `balatrobot serve` CLI (from the
`balatrobot` PyPI package), which handles the Proton launch itself:
it finds Steam, a Proton build, and Balatro, sets
WINEDLLOVERRIDES="version=n,b" so lovely's version.dll is loaded, sets
STEAM_COMPAT_* vars, and runs `proton run Balatro.exe` with all
BALATROBOT_* settings passed through the environment.

What this wrapper adds:

1. A non-destructive "vanilla" mod profile. By default it builds
   bridge/profiles/vanilla/Mods containing symlinks to ONLY Steamodded,
   balatrobot and this repo's bridge/mods/ (BotRun button), then sets
   LOVELY_MOD_DIR to that directory (as a Z:-mapped Wine path) so lovely --
   and therefore Steamodded, which takes its mod dir from lovely -- ignores
   every other installed mod (Cryptid, Talisman, ...) without touching them.

2. Sane defaults. balatrobot defaults audio OFF (it zeroes every volume and
   mutes); for a windowed, watchable session that's wrong, so `--audio` is
   added automatically unless the caller passes --audio/--no-audio themselves
   or runs --headless (where balatrobot's silent default stands).

3. Supervision. `balatrobot serve` does not notice when the game process
   dies (observed live: a user intervening in a bot-driven run mid-request
   hard-crashes Balatro, and serve keeps printing "running"). This wrapper
   polls the API and restarts serve when the game stops answering, so the
   BOT RUN button always has a live game behind it. Intentional quits are
   respected: the BotRun mod marks them from love.quit (QUIT button, window
   close) and the supervisor stops instead of relaunching. Ctrl+C also
   stops everything.

Examples:

    # First smoke test: windowed, watchable
    balatro-bridge-launch

    # Training/validation: headless and fast
    balatro-bridge-launch -- --headless --fast

    # Render frames only when an API call arrives (cheaper than full render)
    balatro-bridge-launch -- --render-on-api --fast

    # Use the full user Mods dir (content mods included -- NOT for validation)
    balatro-bridge-launch --all-mods

    # Show what would run, create the profile, but do not launch
    balatro-bridge-launch --dry-run

Anything after `--` is forwarded verbatim to `balatrobot serve`
(see `balatrobot serve --help` or
https://coder.github.io/balatrobot/latest/cli/ for all flags:
--port, --headless, --render-on-api, --fast, --gamespeed, ...).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The real, user-wide Mods directory (also reachable by the game via
# compatdata .../AppData/Roaming/Balatro/Mods -> ~/.config/Balatro/Mods ->
# ~/.local/share/Balatro/Mods symlinks).
USER_MODS_DIR = Path.home() / ".local/share/Balatro/Mods"

# Directories from USER_MODS_DIR that make up the "vanilla + bot" profile.
# smods dir name is matched by glob so an smods upgrade doesn't break this.
BALATROBOT_DIR_NAME = "balatrobot"
SMODS_GLOB = "smods*"

# Default profile location: bridge/profiles/vanilla/Mods
BRIDGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_MODS = BRIDGE_ROOT / "profiles" / "vanilla" / "Mods"

# Repo-local mods linked into the profile alongside Steamodded + balatrobot.
# BotRun = the main-menu BOT RUN button (see mods/BotRun/botrun.lua).
REPO_MODS_DIR = BRIDGE_ROOT / "mods"
REPO_MOD_NAMES = ("BotRun",)


def unix_to_wine_path(path: Path) -> str:
    """Map an absolute host path to the Wine/Proton view of it.

    Proton maps the host filesystem root to the Z: drive, so
    /home/teq/x is Z:/home/teq/x for the Windows-built lovely injector.
    """
    return "Z:" + str(path)


def build_vanilla_profile(profile_mods: Path) -> list[str]:
    """(Re)build a Mods directory containing only Steamodded + balatrobot.

    Only symlinks are created; nothing in the user's Mods dir is modified.
    Returns the list of mod directory names linked in.
    """
    smods_dirs = sorted(USER_MODS_DIR.glob(SMODS_GLOB))
    if not smods_dirs:
        raise RuntimeError(f"No Steamodded ({SMODS_GLOB}) dir found in {USER_MODS_DIR}")
    if len(smods_dirs) > 1:
        raise RuntimeError(
            f"Multiple Steamodded dirs found in {USER_MODS_DIR}: "
            f"{[d.name for d in smods_dirs]}; remove or rename the stale ones"
        )
    balatrobot_dir = USER_MODS_DIR / BALATROBOT_DIR_NAME
    if not balatrobot_dir.is_dir():
        raise RuntimeError(f"balatrobot mod not found at {balatrobot_dir}")

    wanted = {d.name: d for d in (smods_dirs[0], balatrobot_dir)}
    for name in REPO_MOD_NAMES:
        repo_mod = REPO_MODS_DIR / name
        if repo_mod.is_dir():
            wanted[name] = repo_mod

    profile_mods.mkdir(parents=True, exist_ok=True)
    # Drop stale entries (old smods versions, renamed mods) -- symlinks only.
    for entry in profile_mods.iterdir():
        if entry.is_symlink() and entry.name not in wanted:
            entry.unlink()
    for name, target in wanted.items():
        link = profile_mods / name
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"{link} exists and is not a symlink; refusing to touch")
        link.symlink_to(target)
    return sorted(wanted)


def _port_from_serve_args(serve_args: list[str]) -> int:
    """balatrobot serve's --port if given, else its default (12346)."""
    for i, a in enumerate(serve_args):
        if a == "--port" and i + 1 < len(serve_args):
            return int(serve_args[i + 1])
        if a.startswith("--port="):
            return int(a.split("=", 1)[1])
    return 12346


def _api_up(port: int) -> bool:
    from balatro_bridge.client import BalatroBridgeClient

    try:
        BalatroBridgeClient(host="127.0.0.1", port=port, timeout=3.0).health()
        return True
    except Exception:  # noqa: BLE001 — any failure means "not up"
        return False


def _stop(proc: subprocess.Popen) -> None:
    """SIGINT serve (it does its own Proton/wineserver cleanup), then kill."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def supervise(cmd: list[str], env: dict[str, str], port: int) -> int:
    """Run `balatrobot serve`, restarting it whenever the game stops answering.

    serve keeps running obliviously when the game process dies, so liveness
    is judged by the API health endpoint, not by the child's exit status.
    """
    from balatro_bridge.paths import BOTRUN_USER_QUIT_FILE

    failed_boots = 0
    while True:
        # A stale quit marker must not stop a fresh session.
        BOTRUN_USER_QUIT_FILE.unlink(missing_ok=True)
        proc = subprocess.Popen(cmd, env=env)
        try:
            # Boot: Proton startup is slow; wait generously for first health.
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None or _api_up(port):
                    break
                time.sleep(3)
            if proc.poll() is not None or not _api_up(port):
                _stop(proc)
                failed_boots += 1
                if failed_boots >= 3:
                    print("[bridge] game failed to come up 3 times; giving up",
                          file=sys.stderr)
                    return 1
                print("[bridge] game did not come up; retrying...", flush=True)
                time.sleep(5)
                continue
            failed_boots = 0
            print(f"[bridge] game healthy on port {port} (supervised; "
                  f"Ctrl+C to stop)", flush=True)
            # Steady state: restart on 3 consecutive failed health checks
            # (~15s) so one slow poll during heavy animation doesn't bounce
            # a live game.
            misses = 0
            while misses < 3:
                time.sleep(5)
                if proc.poll() is not None:
                    break
                misses = misses + 1 if not _api_up(port) else 0
            if BOTRUN_USER_QUIT_FILE.exists():
                # The BotRun mod writes this from love.quit: the user quit
                # on purpose (QUIT button / closed the window) — respect it.
                BOTRUN_USER_QUIT_FILE.unlink(missing_ok=True)
                print("[bridge] user quit the game — stopping (not a crash)",
                      flush=True)
                _stop(proc)
                return 0
            print("[bridge] game stopped answering — relaunching...", flush=True)
            _stop(proc)
            time.sleep(3)
        except KeyboardInterrupt:
            print("[bridge] stopping", flush=True)
            _stop(proc)
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="balatro-bridge-launch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all-mods",
        action="store_true",
        help="do not isolate mods; use the normal user Mods dir "
        "(content mods alter game behavior -- never use for validation)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_MODS,
        help=f"vanilla profile Mods dir to build/use (default: {DEFAULT_PROFILE_MODS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the profile and print the command + env, but do not launch",
    )
    parser.add_argument(
        "serve_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are forwarded to `balatrobot serve`",
    )
    args = parser.parse_args(argv)

    serve_args = args.serve_args
    if serve_args and serve_args[0] == "--":
        serve_args = serve_args[1:]
    # Audio on by default for watchable sessions (see module docstring).
    if not {"--audio", "--no-audio", "--headless"} & set(serve_args):
        serve_args = [*serve_args, "--audio"]

    env = os.environ.copy()
    if not args.all_mods:
        profile_mods: Path = args.profile_dir.resolve()
        linked = build_vanilla_profile(profile_mods)
        env["LOVELY_MOD_DIR"] = unix_to_wine_path(profile_mods)
        print(f"[bridge] vanilla profile: {profile_mods}")
        print(f"[bridge] mods enabled: {', '.join(linked)}")
        print(f"[bridge] LOVELY_MOD_DIR={env['LOVELY_MOD_DIR']}")
    else:
        print(f"[bridge] using full user Mods dir: {USER_MODS_DIR} (--all-mods)")

    balatrobot_cli = shutil.which("balatrobot")
    if balatrobot_cli is None:
        print(
            "error: `balatrobot` CLI not found on PATH.\n"
            "Run through the bridge project venv, e.g.:\n"
            "  conda activate balatro && balatro-bridge-launch",
            file=sys.stderr,
        )
        return 1

    cmd = [balatrobot_cli, "serve", *serve_args]
    print(f"[bridge] exec: {' '.join(cmd)}")

    if args.dry_run:
        print("[bridge] dry run; not launching. Verify after a real launch that the")
        print("[bridge] lovely log (profile Mods/lovely/log/) says:")
        print('[bridge]   Using mod directory at "Z:/.../profiles/vanilla/Mods"')
        return 0

    # Supervised run: balatrobot serve handles Proton, logging, and cleanup
    # (wineserver -k) on SIGINT; we watch API health and restart it when the
    # game dies (serve itself never notices).
    return supervise(cmd, env, _port_from_serve_args(serve_args))


if __name__ == "__main__":
    raise SystemExit(main())
