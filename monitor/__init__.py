"""
Gambit training monitor

A PySide6 dashboard over the telemetry event stream. It only ever reads
runs/<id>/events.jsonl, so it can be started, stopped or restarted at any point
without affecting a training run.

Run with:  python -m monitor
"""

__all__ = ["config", "bus", "theme", "layout", "panels"]
