"""
Monitor configuration

Loaded from monitor.json in the repo root when present, otherwise defaults.
Everything here is meant to be edited freely -- the point of the panel registry
is that enabling, disabling and reordering panels is configuration, not code.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

CONFIG_FILENAME = "monitor.json"


@dataclass
class MonitorConfig:
    """
    Runtime settings for the monitor application

    Attributes:
        runs_dir: Parent directory holding run directories. The trainer runs
            with train/ as its working directory and a relative log_dir, so the
            runs land under train/runs by default.
        train_dir: Working directory the trainer is launched from.
        config_path: Trainer YAML config, relative to train_dir.
        poll_ms: How often the tail thread checks the event file for new bytes
        redraw_hz: Panel repaint rate. Deliberately low: the trainer emits one
            rollout event per iteration and a few hundred episode events per
            second, so 10 Hz is already oversampled and painting is the only
            part of the monitor that costs real CPU.
        history: Ring buffer length per series. A multi-day run is millions of
            steps, so buffers must be bounded.
        max_plot_points: Series longer than this are decimated before drawing.
        panels: Panel names to show, in order. Empty means every discovered panel.
        follow_latest: Automatically switch to the newest run when one appears.
    """

    runs_dir: str = os.path.join("train", "runs")
    train_dir: str = "train"
    config_path: str = os.path.join("configs", "m3_fast.yaml")
    poll_ms: int = 100
    redraw_hz: float = 10.0
    history: int = 20000
    max_plot_points: int = 2000
    panels: List[str] = field(default_factory=list)
    follow_latest: bool = True

    @property
    def redraw_ms(self) -> int:
        """Repaint interval in milliseconds"""
        return max(16, int(1000.0 / max(self.redraw_hz, 0.1)))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MonitorConfig":
        """
        Read configuration from disk, falling back to defaults

        Unknown keys are ignored so a config file from a newer version does not
        break an older monitor.
        """
        path = path or CONFIG_FILENAME
        if not os.path.isfile(path):
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw: Dict[str, Any] = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return cls()

        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in valid})

    def save(self, path: Optional[str] = None) -> None:
        """Write configuration to disk"""
        path = path or CONFIG_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)
