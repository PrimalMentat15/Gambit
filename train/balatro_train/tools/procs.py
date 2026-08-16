"""
Process control shared by the monitor's kill switch and the killrun CLI

Both paths must behave identically, so the logic lives here rather than being
implemented twice. Nothing in this module imports Qt or torch, so the CLI works
without the training dependencies installed.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional

IS_WINDOWS = sys.platform == "win32"


def is_alive(pid: Optional[int]) -> bool:
    """
    Whether a process is currently running

    Args:
        pid: Process id, or None

    Returns:
        True if the pid exists
    """
    if not pid:
        return False

    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in out.stdout
        except Exception:
            return False

    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def terminate_tree(pid: Optional[int], timeout: int = 10) -> bool:
    """
    Forcibly end a process and everything it spawned

    Terminate rather than signal, deliberately: the whole point of the kill
    switch is to work when the target is wedged and no longer cooperating.

    Args:
        pid: Process id to kill
        timeout: Seconds to allow the kill command itself

    Returns:
        True if the process is gone afterwards
    """
    if not pid:
        return True
    if not is_alive(pid):
        return True

    try:
        if IS_WINDOWS:
            # /T includes the process tree, /F forces it
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=timeout,
            )
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass

    return not is_alive(pid)


def run_pids(meta: Dict) -> Dict[str, Optional[int]]:
    """
    Extract the process ids recorded for a run

    Training runs entirely in-process against the Rust sim, so the trainer is
    the only process there is to act on.

    Args:
        meta: Parsed meta.json contents

    Returns:
        Mapping of role to pid
    """
    return {"trainer": meta.get("trainer_pid")}


def describe(meta: Dict) -> List[str]:
    """Human-readable status lines for a run's processes"""
    lines = []
    for role, pid in run_pids(meta).items():
        if not pid:
            lines.append(f"  {role:<9} (no pid recorded)")
            continue
        lines.append(
            f"  {role:<9} pid {pid}  {'running' if is_alive(pid) else 'not running'}"
        )
    return lines
