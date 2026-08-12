"""
Run session management

Each training session owns a directory holding everything needed to reconstruct
and compare it later:

    runs/2026-08-12_1543_<name>/
      meta.json      # git sha, hyperparams, device, PIDs, active flag
      events.jsonl   # durable telemetry stream (source of truth)
      tb/            # per-run tensorboard_log
      monitor.csv    # SB3 Monitor output
      checkpoints/

The ``active`` flag and recorded PIDs in meta.json are what let the monitor
re-attach to a session it did not start, and what let ``killrun`` stop a run
without the GUI being alive.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_RUNS_DIR = "runs"
META_FILENAME = "meta.json"
EVENTS_FILENAME = "events.jsonl"
MONITOR_FILENAME = "monitor.csv"


def _git_sha() -> Optional[str]:
    """Current commit SHA, or None outside a git checkout"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes"""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


class RunSession:
    """
    Owns one run directory and its metadata

    Attributes:
        run_dir: Path to this run's directory
        run_id: Directory basename, sortable by creation time
    """

    def __init__(self, run_dir: str, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id

    @classmethod
    def create(
        cls,
        name: str = "run",
        runs_dir: str = DEFAULT_RUNS_DIR,
        config: Optional[Dict[str, Any]] = None,
    ) -> "RunSession":
        """
        Create a new run directory and write its initial meta.json

        Args:
            name: Human-readable suffix for the directory name
            runs_dir: Parent directory holding all runs
            config: Hyperparameters and settings to snapshot

        Returns:
            The new RunSession
        """
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        run_id = f"{stamp}_{safe_name}"
        run_dir = os.path.join(runs_dir, run_id)

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        session = cls(run_dir, run_id)
        session.write_meta(
            {
                "run_id": run_id,
                "created": time.time(),
                "created_iso": datetime.now().isoformat(),
                "active": True,
                "git_sha": _git_sha(),
                "git_dirty": _git_dirty(),
                "python": sys.version.split()[0],
                "trainer_pid": os.getpid(),
                "balatro_pid": None,
                "config": config or {},
            }
        )
        return session

    @property
    def meta_path(self) -> str:
        return os.path.join(self.run_dir, META_FILENAME)

    @property
    def events_path(self) -> str:
        return os.path.join(self.run_dir, EVENTS_FILENAME)

    @property
    def tb_dir(self) -> str:
        return os.path.join(self.run_dir, "tb")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def monitor_path(self) -> str:
        return os.path.join(self.run_dir, MONITOR_FILENAME)

    def read_meta(self) -> Dict[str, Any]:
        """Load meta.json, or an empty dict if unreadable"""
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        """Write meta.json atomically so a reader never sees a partial file"""
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, self.meta_path)

    def update_meta(self, **fields: Any) -> Dict[str, Any]:
        """Merge fields into meta.json"""
        meta = self.read_meta()
        meta.update(fields)
        self.write_meta(meta)
        return meta

    def mark_finished(self, status: str = "completed") -> None:
        """Clear the active flag so tooling stops treating this run as live"""
        self.update_meta(active=False, status=status, finished=time.time())


def list_runs(runs_dir: str = DEFAULT_RUNS_DIR) -> List[RunSession]:
    """
    All runs, newest first

    Run ids are timestamp-prefixed, so a reverse name sort is chronological.
    """
    if not os.path.isdir(runs_dir):
        return []

    entries = [
        d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))
    ]
    entries.sort(reverse=True)
    return [RunSession(os.path.join(runs_dir, d), d) for d in entries]


def find_latest_run(
    runs_dir: str = DEFAULT_RUNS_DIR, active_only: bool = False
) -> Optional[RunSession]:
    """
    Most recent run, optionally restricted to ones still marked active

    Args:
        runs_dir: Parent directory holding all runs
        active_only: Only return a run whose meta.json still has active=True

    Returns:
        The matching RunSession, or None
    """
    for session in list_runs(runs_dir):
        if not active_only:
            return session
        if session.read_meta().get("active"):
            return session
    return None
