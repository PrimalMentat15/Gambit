"""
Process supervision

Launches the trainer and provides the two ways to end a run.

Two properties are deliberate and independent:

- **A monitor crash must not cascade.** The trainer is spawned so it outlives
  the monitor, and its output goes to a file rather than a pipe -- a pipe whose
  reader dies can wedge the writer. Restarting the monitor re-attaches by
  reading the run directory.
- **The user must always be able to stop everything.** The trainer's pid is
  persisted to meta.json, so Stop and Kill work even for a run this monitor did
  not start, and ``python -m balatro_train.tools.killrun`` does the same without
  any GUI.

There is no game process here: training runs entirely in-process against the
Rust sim. Live evaluation against the real game goes through ``bridge/``, which
supervises its own launch.
"""

import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from balatro_train.telemetry import RunSession, find_latest_run
from balatro_train.tools.procs import is_alive, run_pids, terminate_tree

# Windows creation flags
CREATE_NEW_PROCESS_GROUP = 0x00000200


class Supervisor(QObject):
    """Owns the trainer process for one run"""

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
            return {"trainer": None}
        return run_pids(self.session.read_meta())

    @property
    def running(self) -> bool:
        """True while the trainer process is alive"""
        return is_alive(self.pids["trainer"])

    def status_text(self) -> str:
        """One-line summary for the toolbar and control tab"""
        if self.session is None:
            return "no run"

        trainer = "running" if self.running else "stopped"
        return f"{self.session.run_id}   trainer: {trainer}"

    # --- Launch ---

    def start(self, run_name: str = "run", timesteps: Optional[int] = None,
              device: Optional[str] = None, config_path: Optional[str] = None,
              resume: Optional[str] = None) -> Optional[RunSession]:
        """
        Start the trainer and attach to the run directory it will write

        The trainer owns its run directory (it has to: a resumed run reuses the
        same one), so this predicts the path rather than creating it, and
        attaches once the process is up.

        Returns:
            The RunSession, or None if the trainer failed to start
        """
        if self.running:
            self.message.emit("A run is already in progress")
            return None

        train_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.config.train_dir,
            )
        )
        run_dir = os.path.join(self.config.runs_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        session = RunSession.attach(run_dir)
        # A leftover STOP from a previous run would otherwise halt this one on
        # its first iteration.
        session.clear_stop()
        self.session = session

        stdout_path = os.path.join(run_dir, "trainer.out")

        # sys.executable: the monitor and the trainer share one environment, so
        # the interpreter running this window is the one that should run the
        # trainer. Spelling it out beats "python" on PATH, which can resolve to
        # a different install than the one the monitor was launched from.
        cmd = [
            sys.executable, "-u", "-m", "balatro_train.ppo",
            "--config", config_path or self.config.config_path,
            "--run-name", run_name,
        ]
        if timesteps:
            cmd += ["--total-timesteps", str(timesteps)]
        if device:
            cmd += ["--device", device]
        if resume:
            cmd += ["--resume", resume]

        try:
            # Output to a file, not a pipe: if the monitor dies, a pipe with no
            # reader can block the trainer. The log panel tails this file.
            handle = open(stdout_path, "w", encoding="utf-8", buffering=1)
            trainer = subprocess.Popen(
                cmd,
                cwd=train_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                start_new_session=sys.platform != "win32",
            )
        except Exception as exc:
            self.message.emit(f"Could not start trainer: {exc}")
            return None

        # The trainer rewrites meta.json on startup with its own pid; recording
        # it here means Stop and Kill work during the seconds before it does.
        session.update_meta(trainer_pid=trainer.pid, stdout=stdout_path,
                            active=True, launched_by="monitor")
        self.message.emit(f"Trainer started (pid {trainer.pid})")

        self.changed.emit()
        return session

    # --- Stopping ---

    def stop(self) -> bool:
        """
        Ask the trainer to checkpoint and exit at the next iteration boundary

        Cooperative, and normally takes effect within seconds -- an iteration is
        one rollout against the sim. A promotion or milestone eval can stretch
        that out; kill() is the answer when waiting is not acceptable.
        """
        if self.session is None:
            return False

        self.session.request_stop("monitor")
        self.message.emit("Stop requested - the trainer will checkpoint and exit")
        self.changed.emit()
        return True

    def kill(self) -> bool:
        """Terminate the trainer immediately, losing the current iteration"""
        if self.session is None:
            return False

        ok = True
        for role, pid in self.pids.items():
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
