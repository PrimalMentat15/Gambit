"""
Post-run analysis

Reads a finished run from three sources and returns plain Python structures, so
this module stays testable and has no Qt dependency.

TensorBoard event files are read directly with EventAccumulator rather than by
launching the TensorBoard server, which is what lets post-run evaluation live in
the same window as the live view.
"""

import csv
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ai.tools.reader import read_events


def load_scalars(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read TensorBoard scalars for a run

    Args:
        run_dir: Run directory containing a tb/ subdirectory

    Returns:
        Mapping of tag to [(step, value), ...]. Empty if TensorBoard is not
        installed or the run has no event files.
    """
    tb_dir = os.path.join(run_dir, "tb")
    if not os.path.isdir(tb_dir):
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    # SB3 nests each learn() call in its own subdirectory
    candidates = [tb_dir] + [
        os.path.join(tb_dir, name) for name in os.listdir(tb_dir)
        if os.path.isdir(os.path.join(tb_dir, name))
    ]

    scalars: Dict[str, List[Tuple[int, float]]] = {}
    for path in candidates:
        try:
            acc = EventAccumulator(path)
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            points = [(s.step, s.value) for s in acc.Scalars(tag)]
            if points:
                scalars.setdefault(tag, []).extend(points)

    for tag in scalars:
        scalars[tag].sort(key=lambda p: p[0])
    return scalars


def load_monitor_csv(run_dir: str) -> List[Dict[str, float]]:
    """
    Read the SB3 Monitor CSV

    Returns:
        List of {"r": reward, "l": length, "t": wall_time}
    """
    path = os.path.join(run_dir, "monitor.csv")
    if not os.path.isfile(path):
        return []

    rows = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            # First line is a JSON comment written by Monitor
            first = handle.readline()
            if not first.startswith("#"):
                handle.seek(0)
            for row in csv.DictReader(handle):
                try:
                    rows.append({
                        "r": float(row["r"]),
                        "l": float(row["l"]),
                        "t": float(row["t"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    return rows


def summarize_run(run_dir: str) -> Dict[str, Any]:
    """
    Aggregate one run's event stream

    Returns:
        Dict with episode outcomes, reward components, hand types, action costs
        and headline counters
    """
    events_path = os.path.join(run_dir, "events.jsonl")
    summary: Dict[str, Any] = {
        "episodes": 0,
        "wins": 0,
        "steps": 0,
        "rewards": [],
        "components": Counter(),
        "hand_types": defaultdict(lambda: {"count": 0, "chips": 0}),
        "action_wait": defaultdict(lambda: {"count": 0, "total": 0.0}),
        "wall_time": 0.0,
        "best_chips": 0,
    }

    if not os.path.isfile(events_path):
        return summary

    first_t = last_t = None
    for event in read_events(events_path):
        kind = event.get("type")
        data = event.get("data", {})
        stamp = event.get("t")
        if stamp:
            first_t = first_t if first_t is not None else stamp
            last_t = stamp

        if kind == "step":
            summary["steps"] += 1

            action = data.get("action")
            wait = (data.get("timings") or {}).get("t_wait")
            if action is not None and wait is not None:
                bucket = summary["action_wait"][action]
                bucket["count"] += 1
                bucket["total"] += wait

            for line in data.get("reward_components") or []:
                # Components are formatted strings; the label before the colon
                # is the stable part worth counting
                label = line.split(":")[0].strip()
                if label:
                    summary["components"][label] += 1

        elif kind == "episode_end":
            summary["episodes"] += 1
            if data.get("outcome") == "win":
                summary["wins"] += 1
            if data.get("reward") is not None:
                summary["rewards"].append(data["reward"])
            summary["best_chips"] = max(summary["best_chips"], data.get("chips") or 0)

            for hand in data.get("hands_played") or []:
                entry = summary["hand_types"][hand.get("hand_type", "Unknown")]
                entry["count"] += 1
                entry["chips"] += hand.get("chips", 0) or 0

    if first_t and last_t:
        summary["wall_time"] = last_t - first_t

    summary["hand_types"] = dict(summary["hand_types"])
    summary["action_wait"] = dict(summary["action_wait"])
    return summary


def headline(summary: Dict[str, Any]) -> Dict[str, str]:
    """Format a summary into display strings"""
    episodes = summary["episodes"] or 1
    rewards = summary["rewards"]

    return {
        "episodes": f"{summary['episodes']:,}",
        "steps": f"{summary['steps']:,}",
        "win_rate": f"{summary['wins'] / episodes * 100:.1f}%",
        "mean_reward": f"{sum(rewards) / len(rewards):.2f}" if rewards else "--",
        "best_reward": f"{max(rewards):.1f}" if rewards else "--",
        "best_chips": f"{summary['best_chips']:,}",
        "wall_time": _duration(summary["wall_time"]),
        "steps_per_sec": (f"{summary['steps'] / summary['wall_time']:.2f}"
                          if summary["wall_time"] else "--"),
    }


def _duration(seconds: float) -> str:
    """Coarse human duration"""
    if not seconds:
        return "--"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"
