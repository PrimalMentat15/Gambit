"""
Process supervision

Launches Balatro and the trainer, and provides the two ways to end a run.

Two properties are deliberate and independent:

- **A monitor crash must not cascade.** Children are spawned so they outlive the
  monitor, and their output goes to files rather than pipes -- a pipe whose
  reader dies can wedge the writer. Restarting the monitor re-attaches by
  reading the run directory.
- **The user must always be able to stop everything.** Pids are persisted to
  meta.json, so Stop and Kill work even for a run this monitor did not start,
  and ``python -m ai.tools.killrun`` does the same without any GUI.
"""

import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.telemetry import RunSession, find_latest_run  # noqa: E402
from ai.tools.procs import is_alive, run_pids, terminate_tree  # noqa: E402

# Windows creation flags
CREATE_NEW_PROCESS_GROUP = 0x00000200


class Supervisor(QObject):
    """Owns the trainer and Balatro processes for one run"""

    changed = Signal()
    message = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.session: Optional[RunSession] = None

        self.poll = QTimer(self)
        self.poll.timeout.connect(self._check)
        self.poll.start(2000)

    # --- State ---

    def attach_latest(self) -> None:
        """Adopt the newest active run so Stop/Kill work after a restart"""
        if self.session is not None:
            return
        session = find_latest_run(self.config.runs_dir, active_only=True)
        if session is not None:
            self.session = session
            self.changed.emit()

    @property
    def pids(self) -> dict:
        if self.session is None:
            return {"trainer": None, "balatro": None}
        return run_pids(self.session.read_meta())

    @property
    def running(self) -> bool:
        """True while the trainer process is alive"""
        return is_alive(self.pids["trainer"])

    def status_text(self) -> str:
        """One-line summary for the toolbar and control tab"""
        if self.session is None:
            return "no run"

        pids = self.pids
        trainer = "running" if is_alive(pids["trainer"]) else "stopped"
        balatro = "running" if is_alive(pids["balatro"]) else "stopped"
        return f"{self.session.run_id}   trainer: {trainer}   balatro: {balatro}"

    # --- Launch ---

    def start(self, run_name: str = "run", timesteps: int = 250000,
              device: Optional[str] = None, launch_game: bool = True,
              resume: bool = True) -> Optional[RunSession]:
        """
        Create a run directory and start the trainer, then Balatro

        The trainer goes first because it owns the listening socket; the mod
        retries its connection, but starting in this order avoids the retries
        entirely.

        Returns:
            The new RunSession, or None if the trainer failed to start
        """
        if self.running:
            self.message.emit("A run is already in progress")
            return None

        session = RunSession.create(
            name=run_name,
            runs_dir=self.config.runs_dir,
            config={"total_timesteps": timesteps, "device": device or "auto",
                    "launched_by": "supervisor"},
        )
        session.clear_stop()
        self.session = session

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stdout_path = os.path.join(session.run_dir, "trainer.out")

        cmd = [
            sys.executable, "-u", "-m", "ai.train_balatro",
            "--no-prompt",
            "--run-dir", session.run_dir,
            "--timesteps", str(timesteps),
        ]
        if device:
            cmd += ["--device", device]
        if not resume:
            cmd.append("--no-resume")

        try:
            # Output to a file, not a pipe: if the monitor dies, a pipe with no
            # reader can block the trainer. The log panel tails this file.
            handle = open(stdout_path, "w", encoding="utf-8", buffering=1)
            trainer = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                start_new_session=sys.platform != "win32",
            )
        except Exception as exc:
            self.message.emit(f"Could not start trainer: {exc}")
            session.mark_finished("failed")
            return None

        session.update_meta(trainer_pid=trainer.pid, stdout=stdout_path)
        self.message.emit(f"Trainer started (pid {trainer.pid})")

        if launch_game:
            self._start_balatro(session)

        self.changed.emit()
        return session

    def _start_balatro(self, session: RunSession) -> None:
        """Launch the game with autostart enabled for this process only"""
        exe = self.config.balatro_exe
        if not os.path.isfile(exe):
            self.message.emit(f"Balatro not found at {exe} - start it manually")
            return

        # Injected into the child environment only. Exported globally it would
        # make every manual launch start training.
        env = dict(os.environ)
        env["BALATRO_RL_AUTOSTART"] = "1"

        try:
            game = subprocess.Popen(
                [exe],
                cwd=os.path.dirname(exe),
                env=env,
                creationflags=CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
        except Exception as exc:
            self.message.emit(f"Could not start Balatro: {exc}")
            return

        session.update_meta(balatro_pid=game.pid)
        self.message.emit(f"Balatro started (pid {game.pid})")

    # --- Stopping ---

    def stop(self) -> bool:
        """
        Ask the trainer to checkpoint and exit at the next step boundary

        Cooperative, so it cannot help when the trainer is wedged waiting on the
        game. That case is what kill() is for.
        """
        if self.session is None:
            return False

        self.session.request_stop("monitor")
        self.message.emit("Stop requested - the trainer will checkpoint and exit")
        self.changed.emit()
        return True

    def kill(self) -> bool:
        """
        Terminate the trainer and Balatro immediately

        Also kills the game: leaving half a session alive holding the socket
        port is exactly what blocks the next run from starting.
        """
        if self.session is None:
            return False

        pids = self.pids
        ok = True
        for role in ("trainer", "balatro"):
            pid = pids[role]
            if pid and not terminate_tree(pid):
                self.message.emit(f"Could not kill {role} (pid {pid})")
                ok = False
            elif pid:
                self.message.emit(f"Killed {role} (pid {pid})")

        self.session.mark_finished("killed")
        self.session.clear_stop()
        self.changed.emit()
        return ok

    # --- Polling ---

    def _check(self) -> None:
        """Emit a change when the trainer's liveness flips"""
        state = self.running
        if state != getattr(self, "_was_running", None):
            self._was_running = state
            self.changed.emit()
