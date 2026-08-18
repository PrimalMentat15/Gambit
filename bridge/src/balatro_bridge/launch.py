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

IS_WINDOWS = sys.platform == "win32"


def _user_mods_dir() -> Path:
    """
    The real, user-wide Mods directory for this platform

    On Linux the game runs as a Windows build under Proton, and its Mods dir is
    reachable through a chain of symlinks from the XDG location. On Windows it
    is simply the save directory's Mods subfolder -- the same place Steamodded
    and Lovely look. ``BALATRO_MODS_DIR`` overrides both.
    """
    override = os.environ.get("BALATRO_MODS_DIR")
    if override:
        return Path(override)
    if IS_WINDOWS or sys.platform == "darwin":
        from balatro_bridge.paths import BALATRO_SAVE_DIR

        return BALATRO_SAVE_DIR / "Mods"
    return Path.home() / ".local/share/Balatro/Mods"


USER_MODS_DIR = _user_mods_dir()

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

    A native Windows run has no Wine layer, so the path is already what lovely
    expects and is returned unchanged.
    """
    if IS_WINDOWS:
        return str(path)
    return "Z:" + str(path)


def _resolve_mod_root(path: Path) -> Path:
    """
    Descend into a release-zip wrapper directory, if there is one

    Extracting a GitHub release lands the mod one level deeper than the loader
    expects -- ``Mods/smods/smods-1.0.0-beta-1814a/`` rather than
    ``Mods/smods/``. Lovely scans ``Mods/<mod>/lovely/*.toml`` and Steamodded
    reads ``Mods/<mod>/manifest.json``, so with the extra level both find
    nothing, load nothing, and start a completely unmodded game with no error:
    the launcher comes up, the API never answers, and the only symptom is a
    health-check timeout.

    A directory holding a single subdirectory and none of the mod markers is
    such a wrapper, so follow it.
    """
    markers = ("manifest.json", "lovely")
    for _ in range(3):  # bounded: nobody nests a release more than this
        if any((path / m).exists() for m in markers):
            return path
        if any(path.glob("*.lua")):
            return path
        children = [c for c in path.iterdir() if c.is_dir()]
        if len(children) != 1:
            return path
        path = children[0]
    return path


def _is_link(path: Path) -> bool:
    """True for a symlink or (on Windows) a directory junction"""
    return path.is_symlink() or os.path.isjunction(path)


def _unlink(path: Path) -> None:
    """Remove a link without touching what it points at"""
    if os.path.isjunction(path) or (IS_WINDOWS and path.is_dir()):
        os.rmdir(path)  # removes the reparse point, not the target
    else:
        path.unlink()


def _link_dir(link: Path, target: Path) -> None:
    """
    Point ``link`` at ``target`` using whatever this platform allows

    Creating a directory *symlink* on Windows needs Developer Mode or an
    elevated shell; a *junction* needs neither and behaves the same for this
    purpose. Try the symlink first so the result is identical across platforms
    when permissions allow, and fall back rather than demanding elevation.
    """
    if not IS_WINDOWS:
        link.symlink_to(target)
        return
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True, capture_output=True,
        )


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

    # Link the directory the loaders actually read, which is one level in when
    # the mod was installed by extracting a release zip.
    wanted = {
        d.name: _resolve_mod_root(d) for d in (smods_dirs[0], balatrobot_dir)
    }
    for name in REPO_MOD_NAMES:
        repo_mod = REPO_MODS_DIR / name
        if repo_mod.is_dir():
            wanted[name] = _resolve_mod_root(repo_mod)

    profile_mods.mkdir(parents=True, exist_ok=True)
    # Drop stale entries (old smods versions, renamed mods) -- links only.
    for entry in profile_mods.iterdir():
        if _is_link(entry) and entry.name not in wanted:
            _unlink(entry)
    for name, target in wanted.items():
        link = profile_mods / name
        if _is_link(link):
            if link.resolve() == target.resolve():
                continue
            _unlink(link)
        elif link.exists():
            raise RuntimeError(f"{link} exists and is not a link; refusing to touch")
        _link_dir(link, target)
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
    """Ask serve to shut down (it cleans up Proton/wineserver), then kill.

    Windows has no SIGINT delivery to another process the way POSIX does --
    send_signal(SIGINT) there either raises or fells this process instead of the
    child -- so terminate() is the portable request. There is no wineserver to
    wind down on Windows anyway.
    """
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        proc.terminate()
    else:
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


def _add_platform_args(serve_args: list[str]) -> list[str]:
    """
    Tell `balatrobot serve` where things are when it cannot work it out itself

    On Linux it locates Steam, a Proton build and the game on its own. A native
    Windows install has none of that to go on, so the executable and the lovely
    library have to be named explicitly. Anything the caller already passed is
    left alone.

    Env:
        BALATRO_EXE  -- path to Balatro.exe
        LOVELY_DLL   -- path to version.dll (defaults to beside the exe)
    """
    if not IS_WINDOWS:
        return serve_args

    args = list(serve_args)
    if not any(a == "--platform" or a.startswith("--platform=") for a in args):
        args += ["--platform", "windows"]

    # --love-path, NOT --balatro-path: balatrobot's Windows launcher validates
    # and launches config.love_path (platforms/windows.py), and silently falls
    # back to a hardcoded Steam location when it is None -- so passing only
    # --balatro-path fails with "not found: C:\\Program Files (x86)\\Steam\\..."
    # however correct the path you supplied was.
    exe = os.environ.get("BALATRO_EXE")
    has_path = any(a == "--love-path" or a.startswith("--love-path=")
                   for a in args)
    if exe and not has_path:
        args += ["--love-path", exe]

    lovely = os.environ.get("LOVELY_DLL")
    if not lovely and exe:
        # Lovely injects by sitting next to the executable as version.dll
        beside = Path(exe).parent / "version.dll"
        lovely = str(beside) if beside.is_file() else None
    has_lovely = any(a == "--lovely-path" or a.startswith("--lovely-path=")
                     for a in args)
    if lovely and not has_lovely:
        args += ["--lovely-path", lovely]

    return args


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

    serve_args = _add_platform_args(serve_args)

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
