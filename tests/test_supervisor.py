"""
Kill switch and analysis tests

The kill switch is verified against a real spawned process that ignores
cooperative requests, because the case it exists for is precisely the one where
the target is no longer cooperating.

    venv/Scripts/python.exe tests/test_supervisor.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.telemetry import RunSession, find_latest_run
from ai.tools.procs import is_alive, run_pids, terminate_tree


def test_stop_file():
    """A graceful stop is visible to the trainer and clearable"""
    tmp = tempfile.mkdtemp()
    session = RunSession.create(name="stopfile", runs_dir=os.path.join(tmp, "runs"))

    assert not session.stop_requested()
    session.request_stop("test")
    assert session.stop_requested()
    assert os.path.isfile(session.stop_path)

    session.clear_stop()
    assert not session.stop_requested()
    session.clear_stop()  # must be idempotent
    print("stop file OK: request, detect, clear, idempotent")


def test_kill_unresponsive_process():
    """Kill terminates a process that ignores cooperative shutdown"""
    tmp = tempfile.mkdtemp()
    session = RunSession.create(name="killme", runs_dir=os.path.join(tmp, "runs"))

    # A process that traps SIGINT and never exits on its own: this is the
    # 'trainer wedged waiting on Balatro' case the Kill button exists for
    script = (
        "import signal, time, sys\n"
        "try:\n"
        "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "except Exception: pass\n"
        "sys.stdout.write('up'); sys.stdout.flush()\n"
        "while True: time.sleep(0.5)\n"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", script],
                            stdout=subprocess.PIPE)
    try:
        assert proc.stdout.read(2) == b"up", "child never started"
        session.update_meta(trainer_pid=proc.pid)

        assert is_alive(proc.pid), "child should be alive"

        # A graceful stop cannot help here: nothing reads the file
        session.request_stop("test")
        time.sleep(0.5)
        assert is_alive(proc.pid), "unresponsive process should survive a stop request"
        print(f"stop request ignored by unresponsive pid {proc.pid} (as expected)")

        pids = run_pids(session.read_meta())
        assert pids["trainer"] == proc.pid, pids

        assert terminate_tree(proc.pid), "terminate_tree reported failure"
        time.sleep(0.5)
        assert not is_alive(proc.pid), "process survived the kill"
        print(f"kill terminated pid {proc.pid}")
    finally:
        if proc.poll() is None:
            proc.kill()

    session.mark_finished("killed")
    assert session.read_meta()["status"] == "killed"
    assert session.read_meta()["active"] is False
    print("kill switch OK: survives stop, dies to kill, run marked killed")


def test_kill_missing_pid():
    """Killing something already gone is a no-op, not an error"""
    assert terminate_tree(None)
    assert terminate_tree(999_999_999)
    assert not is_alive(999_999_999)
    print("kill of absent process OK")


def test_attach_and_discovery():
    """A run created by one process is fully usable from another"""
    tmp = tempfile.mkdtemp()
    runs = os.path.join(tmp, "runs")
    created = RunSession.create(name="attachme", runs_dir=runs)
    created.update_meta(trainer_pid=4242, balatro_pid=4243)

    # Simulates the monitor restarting and adopting a run it did not start
    found = find_latest_run(runs, active_only=True)
    assert found is not None and found.run_id == created.run_id

    attached = RunSession.attach(found.run_dir)
    pids = run_pids(attached.read_meta())
    assert pids == {"trainer": 4242, "balatro": 4243}, pids
    print(f"re-attach OK: recovered pids {pids} from meta.json")


def test_analysis():
    """Summaries aggregate a real run without Qt"""
    from monitor import analysis

    runs = [d for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d))] \
        if os.path.isdir("runs") else []
    if not runs:
        print("analysis: no runs on disk, skipped")
        return

    run_dir = os.path.join("runs", sorted(runs)[-1])
    summary = analysis.summarize_run(run_dir)
    head = analysis.headline(summary)

    assert summary["steps"] >= 0
    assert set(head) >= {"episodes", "win_rate", "mean_reward", "steps_per_sec"}
    print(f"analysis OK on {os.path.basename(run_dir)}: "
          f"{head['episodes']} episodes, {head['steps']} steps, "
          f"win rate {head['win_rate']}, {head['steps_per_sec']} steps/s")

    scalars = analysis.load_scalars(run_dir)
    print(f"tensorboard scalars: {len(scalars)} tags "
          f"({', '.join(sorted(scalars)[:3])}{'...' if len(scalars) > 3 else ''})")

    rows = analysis.load_monitor_csv(run_dir)
    print(f"monitor.csv rows: {len(rows)}")


if __name__ == "__main__":
    test_stop_file()
    test_kill_unresponsive_process()
    test_kill_missing_pid()
    test_attach_and_discovery()
    test_analysis()
    print("\nALL SUPERVISOR TESTS PASSED")
