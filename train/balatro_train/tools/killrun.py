"""
Stop a training run from the terminal

The kill switch must not depend on the GUI being alive, so this does exactly
what the monitor's Stop and Kill buttons do, reading the same meta.json.

    python -m balatro_train.tools.killrun            # status only
    python -m balatro_train.tools.killrun --stop     # graceful: checkpoint, exit
    python -m balatro_train.tools.killrun --kill     # force: terminate trainer
"""

import argparse
import time

from ..telemetry import RunSession, find_latest_run
from .procs import describe, is_alive, run_pids, terminate_tree


def resolve(target=None, runs_dir="runs"):
    """Find the run to act on: an explicit directory, else the newest active one"""
    if target:
        return RunSession.attach(target)

    session = find_latest_run(runs_dir, active_only=True)
    if session is None:
        session = find_latest_run(runs_dir)
    return session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stop a Balatro training run")
    parser.add_argument("run", nargs="?", default=None,
                        help="Run directory (default: newest active run)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--stop", action="store_true",
                        help="Graceful: finish the iteration, checkpoint, exit")
    parser.add_argument("--kill", action="store_true",
                        help="Force: terminate the trainer immediately")
    parser.add_argument("--wait", type=float, default=120.0,
                        help="Seconds to wait for a graceful stop before reporting")
    args = parser.parse_args(argv)

    session = resolve(args.run, args.runs_dir)
    if session is None:
        print(f"No runs found in {args.runs_dir}/")
        return 1

    meta = session.read_meta()
    print(f"Run: {session.run_id}")
    print(f"  active   {meta.get('active')}   status {meta.get('status', '-')}")
    for line in describe(meta):
        print(line)

    if not (args.stop or args.kill):
        print("\nNothing to do. Pass --stop (graceful) or --kill (force).")
        return 0

    pids = run_pids(meta)

    if args.stop:
        session.request_stop("killrun")
        print("\nStop requested. The trainer checkpoints and exits at the next")
        print("iteration boundary.")

        deadline = time.time() + args.wait
        while time.time() < deadline:
            if not is_alive(pids["trainer"]):
                print("Trainer exited cleanly.")
                return 0
            time.sleep(1)

        print(f"Trainer still running after {args.wait:.0f}s.")
        print("An iteration plus a promotion/milestone eval can take a while;")
        print("wait longer, or use --kill to force it.")
        return 2

    if args.kill:
        print("\nForcing termination...")
        ok = True
        for role, pid in pids.items():
            if not pid:
                print(f"  {role:<9} no pid recorded, skipping")
                continue
            if terminate_tree(pid):
                print(f"  {role:<9} pid {pid} terminated")
            else:
                print(f"  {role:<9} pid {pid} COULD NOT BE KILLED")
                ok = False

        session.mark_finished("killed")
        session.clear_stop()
        print("\nRun marked as killed." if ok else "\nSome processes survived.")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
