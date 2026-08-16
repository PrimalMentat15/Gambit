"""
Run session management

Each training session owns a directory holding everything needed to reconstruct
and compare it later:

    runs/<run_name>/
      meta.json      # git sha, hyperparams, device, PIDs, active flag
      events.jsonl   # durable telemetry stream (source of truth)
      ckpt_*.pt      # checkpoints
      events.out.*   # TensorBoard

The ``active`` flag and recorded PIDs in meta.json are what let the monitor
re-attach to a session it did not start, and what let ``killrun`` stop a run
without the GUI being alive.

Two ways to get a session directory:

``for_run`` uses ``runs/<run_name>/`` directly, which is what the trainer does.
The run name is the stable handle the whole workflow is built on -- configs name
it, ``--resume runs/<name>/ckpt_*.pt`` points into it, supervisors restart into
it, and TensorBoard curves stay continuous across restarts because the event
files keep accumulating in one place. A timestamped directory per launch would
break all of that.

``create`` makes a fresh timestamped directory, for one-off runs that should
never collide with an existing one.
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
STOP_FILENAME = "STOP"


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
        run_id: Directory basename
    """

    def __init__(self, run_dir: str, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id

    @classmethod
    def for_run(
        cls,
        name: str = "run",
        runs_dir: str = DEFAULT_RUNS_DIR,
        config: Optional[Dict[str, Any]] = None,
    ) -> "RunSession":
        """
        Open ``runs_dir/name``, creating it if it does not exist

        Reopening an existing directory refreshes the live fields (pid, git sha,
        config, active) and preserves ``created``, so a resumed run keeps one
        identity across restarts. Any stale stop request is cleared, since a
        leftover STOP file would otherwise halt the new run immediately.

        Args:
            name: Run name; becomes the directory name
            runs_dir: Parent directory holding all runs
            config: Hyperparameters and settings to snapshot

        Returns:
            The RunSession for that directory
        """
        run_dir = os.path.join(runs_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        session = cls(run_dir, name)
        session.clear_stop()

        existing = session.read_meta()
        session.write_meta(
            {
                "run_id": name,
                "created": existing.get("created", time.time()),
                "created_iso": existing.get(
                    "created_iso", datetime.now().isoformat()
                ),
                "resumed": time.time() if existing else None,
                "active": True,
                "git_sha": _git_sha(),
                "git_dirty": _git_dirty(),
                "python": sys.version.split()[0],
                "trainer_pid": os.getpid(),
                "config": config or {},
            }
        )
        return session

    @classmethod
    def create(
        cls,
        name: str = "run",
        runs_dir: str = DEFAULT_RUNS_DIR,
        config: Optional[Dict[str, Any]] = None,
    ) -> "RunSession":
        """
        Create a new timestamped run directory and write its initial meta.json

        Args:
            name: Human-readable suffix for the directory name
            runs_dir: Parent directory holding all runs
            config: Hyperparameters and settings to snapshot

        Returns:
            The new RunSession
        """
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return cls.for_run(f"{stamp}_{safe_name}", runs_dir, config)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.run_dir, META_FILENAME)

    @property
    def events_path(self) -> str:
        return os.path.join(self.run_dir, EVENTS_FILENAME)

    @property
    def stop_path(self) -> str:
        return os.path.join(self.run_dir, STOP_FILENAME)

    def request_stop(self, reason: str = "requested") -> None:
        """
        Ask the trainer to finish cleanly at the next iteration boundary

        A file rather than a signal: signal delivery into a process blocked in
        native code is unreliable on Windows, and a file works identically
        whether the trainer was started by the monitor, a shell, or a
        supervisor -- none of which need to share a process tree with whoever
        asks it to stop.
        """
        with open(self.stop_path, "w", encoding="utf-8") as handle:
            handle.write(reason)

    def stop_requested(self) -> bool:
        """True if a graceful stop has been asked for"""
        return os.path.exists(self.stop_path)

    def clear_stop(self) -> None:
        """Remove any stale stop request"""
        try:
            os.remove(self.stop_path)
        except OSError:
            pass

    @classmethod
    def attach(cls, run_dir: str) -> "RunSession":
        """Wrap an existing run directory without touching its metadata"""
        return cls(run_dir, os.path.basename(os.path.normpath(run_dir)))

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

    def _sort_key(self) -> float:
        """
        When this run was last written to, newest-first orderings

        Run names are no longer timestamp-prefixed, so ordering comes from
        meta.json's ``resumed``/``created`` -- falling back to directory mtime
        for a directory that has no readable metadata yet.
        """
        meta = self.read_meta()
        stamp = meta.get("resumed") or meta.get("created")
        if isinstance(stamp, (int, float)):
            return float(stamp)
        try:
            return os.path.getmtime(self.run_dir)
        except OSError:
            return 0.0


def list_runs(runs_dir: str = DEFAULT_RUNS_DIR) -> List[RunSession]:
    """All runs, newest first"""
    if not os.path.isdir(runs_dir):
        return []

    sessions = [
        RunSession(os.path.join(runs_dir, d), d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    sessions.sort(key=RunSession._sort_key, reverse=True)
    return sessions


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
