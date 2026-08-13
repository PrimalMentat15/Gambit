"""
Latency report

Answers the question Stage 1 exists for: where does the per-step time actually go?

    python -m ai.tools.latency_report                 # latest run
    python -m ai.tools.latency_report runs/<run_id>

Python can only measure total time blocked waiting for the game. The mod reports
how many frames elapsed and how many of those were blocked on the event queue,
which separates animation waits from state-hash churn -- two causes with
completely different fixes.
"""

import argparse
from typing import Dict, List, Sequence

from .reader import load_events, resolve_run

# Stage keys in the order they occur within a step
STAGE_ORDER = ["t_map", "t_send", "t_wait", "t_obs", "t_reward"]

STAGE_LABEL = {
    "t_map": "action mapping",
    "t_send": "socket write",
    "t_wait": "waiting on game",
    "t_obs": "observation build",
    "t_reward": "reward calc",
}

# Mirrors ACTIONS in RLBridge/actions.lua
ACTION_NAMES = {
    1: "SELECT_HAND",
    2: "PLAY_HAND",
    3: "DISCARD_HAND",
    4: "START_RUN",
    5: "SELECT_BLIND",
    6: "RESTART_RUN",
}

# Steps at or above this are treated as the slow mode of a bimodal distribution
SLOW_THRESHOLD_S = 0.1

# Histogram buckets in seconds
BUCKETS = [
    (0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3),
    (0.3, 0.6), (0.6, 1.0), (1.0, float("inf")),
]


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an unsorted sequence"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Mean, median, p95 and max of a sample"""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def format_row(label: str, stats: Dict[str, float], share: float) -> str:
    """One line of the stage table"""
    bar_width = int(round(share * 30))
    bar = "#" * bar_width
    return (
        f"  {label:<20} {stats['mean'] * 1000:>8.1f} {stats['p50'] * 1000:>8.1f} "
        f"{stats['p95'] * 1000:>8.1f} {stats['max'] * 1000:>9.1f} {share * 100:>7.1f}%  {bar}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Report where per-step time goes")
    parser.add_argument("run", nargs="?", default=None,
                        help="Run directory or events.jsonl (default: latest run)")
    parser.add_argument("--runs-dir", default="runs", help="Parent directory for runs")
    args = parser.parse_args(argv)

    path = resolve_run(args.run, args.runs_dir)
    events = load_events(path, types={"step"})

    if not events:
        print(f"No step events in {path}")
        return 1

    # Collect per-stage samples
    stages: Dict[str, List[float]] = {key: [] for key in STAGE_ORDER}
    totals: List[float] = []
    for event in events:
        timings = event["data"].get("timings", {})
        total = 0.0
        for key in STAGE_ORDER:
            if key in timings:
                stages[key].append(timings[key])
                total += timings[key]
        if total:
            totals.append(total)

    total_stats = summarize(totals)
    grand_mean = total_stats["mean"] or 1e-9

    print(f"\nLatency report: {path}")
    print(f"Steps analysed: {len(events)}\n")

    print(f"  {'stage':<20} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>9} {'share':>8}")
    print(f"  {'':-<20} {'':->8} {'':->8} {'':->8} {'':->9} {'':->8}")
    for key in STAGE_ORDER:
        samples = stages[key]
        if not samples:
            continue
        stats = summarize(samples)
        share = stats["mean"] / grand_mean
        print(format_row(STAGE_LABEL.get(key, key), stats, share))

    print(f"  {'':-<20} {'':->8} {'':->8} {'':->8} {'':->9} {'':->8}")
    print(f"  {'TOTAL per step':<20} {total_stats['mean'] * 1000:>8.1f} "
          f"{total_stats['p50'] * 1000:>8.1f} {total_stats['p95'] * 1000:>8.1f} "
          f"{total_stats['max'] * 1000:>9.1f}")
    # Two different rates, both real. The per-step figure is what the stage
    # breakdown above sums to; the wall-clock figure includes episode resets and
    # seed transitions, so it is the one that predicts how long a run takes.
    print(f"\n  Per-step rate:   {1.0 / grand_mean:.2f} steps/sec  "
          f"({grand_mean * 1000:.0f} ms/step, excludes resets)")

    wall = wall_clock_rate(events)
    if wall:
        rate, span = wall
        print(f"  Wall-clock rate: {rate:.2f} steps/sec  "
              f"(over {span / 60:.1f} min, includes resets between episodes)")

    report_distribution(stages.get("t_wait") or [])
    report_by_action(events)
    report_game_side(events)
    report_verdict(stages, grand_mean)
    return 0


def wall_clock_rate(events: List[dict]):
    """
    True end-to-end step rate, including everything between steps

    The stage breakdown only accounts for time inside a step. Episode resets and
    the game starting a new seed happen between them, so this is always the lower
    and more honest number for estimating how long a run will take.

    Returns:
        (steps_per_second, elapsed_seconds), or None if unknown
    """
    stamps = [e["t"] for e in events if e.get("t")]
    if len(stamps) < 2:
        return None
    span = max(stamps) - min(stamps)
    if span <= 0:
        return None
    return (len(stamps) - 1) / span, span


def report_distribution(waits: List[float]) -> None:
    """
    Show the shape of the wait distribution

    A mean far above the median means a minority of steps owns the wall clock,
    and averages alone will hide it.
    """
    if not waits:
        return

    total = len(waits)
    print("\n  Wait distribution:")
    for low, high in BUCKETS:
        count = sum(1 for w in waits if low <= w < high)
        if not count:
            continue
        label = f"{low * 1000:.0f}-{high * 1000:.0f} ms" if high != float("inf") else f">{low * 1000:.0f} ms"
        share = count / total
        print(f"    {label:<14} {count:>6}  {share * 100:>5.1f}%  {'#' * int(share * 45)}")

    fast = [w for w in waits if w < SLOW_THRESHOLD_S]
    slow = [w for w in waits if w >= SLOW_THRESHOLD_S]
    if fast and slow:
        slow_share = sum(slow) / sum(waits) * 100
        print(f"\n    fast (<{SLOW_THRESHOLD_S * 1000:.0f} ms): {len(fast):>5} steps "
              f"({len(fast) / total * 100:.1f}%)  mean {sum(fast) / len(fast) * 1000:6.1f} ms")
        print(f"    slow (>{SLOW_THRESHOLD_S * 1000:.0f} ms): {len(slow):>5} steps "
              f"({len(slow) / total * 100:.1f}%)  mean {sum(slow) / len(slow) * 1000:6.1f} ms")
        print(f"    -> the slow {len(slow) / total * 100:.0f}% of steps own "
              f"{slow_share:.1f}% of all wall time")


def report_by_action(events: List[dict]) -> None:
    """
    Attribute wall time to the action that caused it

    This is the actionable view: it names which game interactions are expensive
    rather than reporting one blended average across all of them.
    """
    groups: Dict[str, List[float]] = {}
    for event in events:
        wait = event["data"].get("timings", {}).get("t_wait")
        if wait is None:
            continue
        action = event["data"].get("action")
        name = ACTION_NAMES.get(action, str(action))
        groups.setdefault(name, []).append(wait)

    if not groups:
        return

    grand_total = sum(sum(v) for v in groups.values()) or 1e-9

    print("\n  Wall time by action sent:")
    print(f"    {'action':<16}{'n':>6}{'mean ms':>10}{'p50 ms':>9}{'total s':>10}{'% time':>9}")
    for name, samples in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        stats = summarize(samples)
        total = sum(samples)
        print(f"    {name:<16}{len(samples):>6}{stats['mean'] * 1000:>10.1f}"
              f"{stats['p50'] * 1000:>9.1f}{total:>10.1f}{total / grand_total * 100:>8.1f}%")

    # Projected gain from making the most expensive action as cheap as the cheapest
    ranked = sorted(groups.items(), key=lambda kv: -sum(kv[1]))
    cheapest = min(groups.values(), key=lambda v: sum(v) / len(v))
    floor = sum(cheapest) / len(cheapest)
    worst_name, worst = ranked[0]
    if sum(worst) / len(worst) > floor * 2:
        saved = sum(worst) - floor * len(worst)
        speedup = grand_total / max(grand_total - saved, 1e-9)
        print(f"\n    If {worst_name} cost the same as the cheapest action "
              f"({floor * 1000:.0f} ms),\n    total time would drop {saved:.0f}s "
              f"-> {speedup:.1f}x faster overall.")


def report_game_side(events: List[dict]) -> None:
    """Break the game-side wait into blocked versus idle frames"""
    frames, blocked, not_ready, elapsed = [], [], [], []
    for event in events:
        timing = event["data"].get("game_timing") or {}
        if "frames_since_last_action" in timing:
            frames.append(timing["frames_since_last_action"])
            blocked.append(timing.get("blocked_frames", 0))
            not_ready.append(timing.get("not_ready_frames", 0))
            elapsed.append(timing.get("elapsed", 0.0))

    if not frames:
        print("\n  Game-side timing: not reported "
              "(update the RLBridge mod and restart Balatro)")
        return

    mean_frames = sum(frames) / len(frames)
    mean_blocked = sum(blocked) / len(blocked)
    mean_not_ready = sum(not_ready) / len(not_ready)
    mean_elapsed = sum(elapsed) / len(elapsed)
    blocked_share = (mean_blocked / mean_frames * 100) if mean_frames else 0.0

    print("\n  Game-side breakdown (per step, mean):")
    print(f"    frames elapsed          {mean_frames:>8.1f}")
    print(f"    blocked on event queue  {mean_blocked:>8.1f}  ({blocked_share:.1f}% of frames)")
    print(f"    not ready for action    {mean_not_ready:>8.1f}")
    print(f"    game clock elapsed      {mean_elapsed * 1000:>8.1f} ms")

    report_blocking_events(events)

    if mean_frames > 0:
        idle = mean_frames - mean_blocked
        if blocked_share > 60:
            cause = ("animations / blocking events dominate -- attack the event queue "
                     "(G.E_MANAGER) and forced waits")
        elif idle > mean_blocked:
            cause = ("most frames are NOT blocked -- the game is idle while the state hash "
                     "is unchanged; look at AI.hash_combined_state and the update cadence")
        else:
            cause = "blocked and idle frames are comparable; both paths are worth attention"
        print(f"\n  Likely cause: {cause}")


def report_blocking_events(events: List[dict]) -> None:
    """
    Rank the event-manager entries that actually hold the game up

    An event whose timer is REAL ignores G.SETTINGS.GAMESPEED entirely, so those
    rows are the ones worth patching first.
    """
    speeds = {
        e["data"]["game_timing"]["gamespeed"]
        for e in events
        if isinstance(e["data"].get("game_timing"), dict)
        and "gamespeed" in e["data"]["game_timing"]
    }
    if speeds:
        shown = ", ".join(str(s) for s in sorted(speeds))
        print(f"\n  Live G.SETTINGS.GAMESPEED: {shown}")

    drains = [
        e["data"]["game_timing"]["drain_passes"]
        for e in events
        if isinstance(e["data"].get("game_timing"), dict)
        and "drain_passes" in e["data"]["game_timing"]
    ]
    if drains:
        active = [d for d in drains if d]
        print(f"  Forced event-queue passes: {sum(drains)} total, "
              f"{sum(drains) / len(drains):.1f} per step"
              + (f" (max {max(drains)})" if active else " - draining is OFF"))

    tally: Dict[str, int] = {}
    for event in events:
        timing = event["data"].get("game_timing") or {}
        for entry in timing.get("blocking") or []:
            sig = entry.get("sig", "?")
            tally[sig] = tally.get(sig, 0) + int(entry.get("frames", 0))

    if not tally:
        if speeds:
            print("  Blocking events: none sampled "
                  "(set BALATRO_RL_DIAG=1 and restart Balatro)")
        return

    total = sum(tally.values()) or 1
    print("\n  Blocking events by frames held (top 12):")
    print(f"    {'frames':>8} {'share':>7}  event")
    for sig, frames in sorted(tally.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {frames:>8} {frames / total * 100:>6.1f}%  {sig}")

    real = sum(v for k, v in tally.items() if "timer=REAL" in k)
    if real:
        print(f"\n    {real / total * 100:.1f}% of blocked frames are on timer=REAL, "
              f"which GAMESPEED does not scale.")


def report_verdict(stages: Dict[str, List[float]], grand_mean: float) -> None:
    """State plainly whether Python or the game owns the step budget"""
    wait = stages.get("t_wait") or []
    if not wait:
        return
    wait_mean = sum(wait) / len(wait)
    python_mean = grand_mean - wait_mean
    print(f"\n  Verdict: {wait_mean / grand_mean * 100:.1f}% of each step is spent waiting on "
          f"Balatro,\n           {python_mean * 1000:.2f} ms is Python "
          f"({python_mean / grand_mean * 100:.1f}%).\n")


if __name__ == "__main__":
    raise SystemExit(main())
