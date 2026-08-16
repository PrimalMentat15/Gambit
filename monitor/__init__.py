"""
Balatro training monitor

A PySide6 dashboard over the telemetry event stream. It only ever reads
runs/<id>/events.jsonl, so it can be started, stopped or restarted at any point
without affecting a training run.

Run with:  python -m monitor

The telemetry package lives in ``train/balatro_train/telemetry`` and is
stdlib-only, so the monitor imports it directly rather than duplicating the
schema. That means ``train/`` has to be importable; putting it on the path here
keeps it to one place instead of a hack at the top of every module that needs it.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ROOT = os.path.join(REPO_ROOT, "train")

for _path in (REPO_ROOT, TRAIN_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

__all__ = ["config", "bus", "theme", "layout", "panels"]
