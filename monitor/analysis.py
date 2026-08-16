"""
Post-run analysis

Reads a finished run's event stream and TensorBoard scalars, returning plain
Python structures, so this module stays testable and has no Qt dependency.

TensorBoard event files are read directly with EventAccumulator rather than by
launching the TensorBoard server, which is what lets post-run evaluation live in
the same window as the live view.
"""

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from balatro_train.tools.reader import read_events


def load_scalars(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read TensorBoard scalars for a run

    Args:
        run_dir: Run directory holding the TensorBoard event files

    Returns:
        Mapping of tag to [(step, value), ...]. Empty if TensorBoard is not
        installed or the run has no event files.
    """
    if not os.path.isdir(run_dir):
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    # The trainer writes event files into the run directory itself; a nested
    # tb/ is still accepted so runs written by other tooling load too.
    candidates = [run_dir] + [
        os.path.join(run_dir, name) for name in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, name))
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


def summarize_run(run_dir: str) -> Dict[str, Any]:
    """
    Aggregate one run's event stream

    Returns:
        Dict with episode outcomes, terminal antes, action-type totals,
        curriculum history and headline counters
    """
    events_path = os.path.join(run_dir, "events.jsonl")
    summary: Dict[str, Any] = {
        "episodes": 0,
        "wins": 0,
        "steps": 0,
        "iterations": 0,
        "returns": [],
        "antes": Counter(),
        "action_counts": Counter(),
        "promotions": [],
        "milestones": [],
        "win_ante": None,
        "wall_time": 0.0,
        "best_ante": 0,
        "status": None,
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

        if kind == "episode_end":
            summary["episodes"] += 1
            if data.get("won"):
                summary["wins"] += 1
            if data.get("r") is not None:
                summary["returns"].append(data["r"])
            ante = data.get("ante") or 0
            summary["antes"][ante] += 1
            summary["best_ante"] = max(summary["best_ante"], ante)

        elif kind == "rollout":
            summary["iterations"] += 1
            summary["steps"] = data.get("global_step", summary["steps"])
            if data.get("win_ante") is not None:
                summary["win_ante"] = data["win_ante"]
            # Per-rollout counts, so a run total is their sum
            for action, count in enumerate(data.get("action_counts") or []):
                if count:
                    summary["action_counts"][action] += count

        elif kind == "curriculum_promotion":
            summary["promotions"].append(data)

        elif kind == "milestone_eval":
            summary["milestones"].append(data)

        elif kind == "session_end":
            summary["status"] = data.get("status")

    if first_t and last_t:
        summary["wall_time"] = last_t - first_t

    summary["antes"] = dict(summary["antes"])
    summary["action_counts"] = dict(summary["action_counts"])
    return summary


def headline(summary: Dict[str, Any]) -> Dict[str, str]:
    """Format a summary into display strings"""
    episodes = summary["episodes"] or 1
    returns = summary["returns"]
    milestones = summary["milestones"]

    return {
        "episodes": f"{summary['episodes']:,}",
        "steps": f"{summary['steps']:,}",
        "iterations": f"{summary['iterations']:,}",
        # Training win rate is against whatever curriculum goal was active at
        # the time, so it is labelled as such wherever it is displayed.
        "win_rate": f"{summary['wins'] / episodes * 100:.1f}%",
        "mean_return": f"{sum(returns) / len(returns):.2f}" if returns else "--",
        "best_return": f"{max(returns):.2f}" if returns else "--",
        "best_ante": f"{summary['best_ante']}",
        "win_ante": (f"ante {summary['win_ante']}"
                     if summary["win_ante"] is not None else "--"),
        "promotions": f"{len(summary['promotions'])}",
        "milestone": (f"{milestones[-1].get('win_rate', 0) * 100:.1f}%"
                      if milestones else "--"),
        "wall_time": _duration(summary["wall_time"]),
        "steps_per_sec": (f"{summary['steps'] / summary['wall_time']:,.0f}"
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
