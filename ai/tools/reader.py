"""
Event stream reader

Shared by the CLI tools and, later, the monitor UI. Reads newline-delimited JSON
and tolerates a partially-written trailing line, which is normal when reading a
file the trainer is still appending to.
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional


def read_events(path: str, types: Optional[set] = None) -> Iterator[Dict[str, Any]]:
    """
    Yield events from an events.jsonl file

    Args:
        path: Path to events.jsonl
        types: Optional set of event types to keep

    Yields:
        Event dictionaries
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Trailing partial line from a live writer
                continue
            if types is None or event.get("type") in types:
                yield event


def load_events(path: str, types: Optional[set] = None) -> List[Dict[str, Any]]:
    """Read all matching events into a list"""
    return list(read_events(path, types))


def resolve_run(target: Optional[str], runs_dir: str = "runs") -> str:
    """
    Turn a CLI argument into an events.jsonl path

    Accepts a run directory, a direct path to an events file, or None to mean
    the most recent run.

    Args:
        target: Run directory, events.jsonl path, or None
        runs_dir: Parent directory holding all runs

    Returns:
        Path to an events.jsonl file

    Raises:
        SystemExit: If no matching run exists
    """
    if target:
        if os.path.isfile(target):
            return target
        candidate = os.path.join(target, "events.jsonl")
        if os.path.isfile(candidate):
            return candidate
        raise SystemExit(f"No events.jsonl found at: {target}")

    from ..telemetry import find_latest_run

    session = find_latest_run(runs_dir)
    if session is None:
        raise SystemExit(f"No runs found in {runs_dir}/ - run a training session first")
    if not os.path.isfile(session.events_path):
        raise SystemExit(f"Run {session.run_id} has no events.jsonl")
    return session.events_path
