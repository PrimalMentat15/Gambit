"""
Throughput panel

Env steps per second over training steps, plus the headline numbers a long run
is judged on: current rate, total steps, and projected time remaining.

The trainer reports throughput once per iteration, so this reads that field
rather than timing event arrivals -- an event-arrival rate would measure the
monitor, not the trainer.

Prefers ``sps_inst`` (rate over the last iteration) and falls back to ``sps``
(session average) for streams written before ``sps_inst`` existed. The
instantaneous rate is what the ETA needs: a session average carries the cost of
every slow startup iteration for a long time afterwards, so an ETA built on it
stays wrong for minutes after the run has settled.
"""

from typing import Any, Dict

from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class ThroughputPanel(Panel):
    """Env steps/sec, with total and ETA"""

    NAME = "throughput"
    TITLE = "Throughput"
    EVENT_TYPES = frozenset({"rollout", "session_start"})
    PAGE = "diagnostics"
    SIZE = (480, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.series = RingSeries(config.history)
        self.total_steps = 0
        self.target_steps = 0
        self.rate = 0.0
        self.trusted = False  # seen a real per-iteration rate yet?

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        grid = QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.tile_rate = StatTile("steps / sec")
        self.tile_total = StatTile("total steps")
        self.tile_eta = StatTile("est. remaining")
        for column, tile in enumerate((self.tile_rate, self.tile_total, self.tile_eta)):
            grid.addWidget(tile, 0, column)

        self.plot = theme.make_plot(y_label="steps/sec", x_label="env step")
        self.curve = self.plot.plot([], [], pen=theme.pen(theme.SERIES[0]))
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:,.0f}/s"
        )

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]

        if event["type"] == "session_start":
            self.target_steps = data.get("total_timesteps", 0) or 0
            return

        step = data.get("global_step")
        instant = data.get("sps_inst")
        # Legacy streams only carry the session average, which is wrong for a
        # resumed run: it divides an absolute step count by this session's
        # elapsed time. Show it when it is all there is, but drop it the moment
        # a real per-iteration rate arrives -- otherwise ~1000 iterations of
        # decaying wrong values sit in the same series as the right ones and
        # compress the axis until the real rate reads as a flat line at zero.
        if instant is not None and not self.trusted:
            self.trusted = True
            self.series.clear()
        sps = instant if instant is not None else data.get("sps")
        if step is None or sps is None:
            return

        self.total_steps = step
        self.rate = float(sps)
        self.series.append(step, self.rate)

    def redraw(self) -> None:
        xs, ys = self.series.arrays(self.config.max_plot_points)
        self.curve.setData(xs, ys)
        self.crosshair.set_series(xs, ys)
        self._scale_to_typical(ys)

        self.tile_rate.set_value(f"{self.rate:,.0f}" if self.rate else "--")
        self.tile_total.set_value(f"{self.total_steps:,}")

        remaining = self.target_steps - self.total_steps
        if self.rate > 0 and remaining > 0:
            self.tile_eta.set_value(self._duration(remaining / self.rate))
        else:
            self.tile_eta.set_value("--")

    def _scale_to_typical(self, ys) -> None:
        """
        Range the y axis to the bulk of the series, not its extremes

        A resumed run recorded before the ``sps_inst`` fix carries a first
        sample orders of magnitude too high (session-average throughput divided
        by a near-zero elapsed time). Autoranging to that single point flattens
        every real value onto the axis, so the chart of a healthy 5,000 steps/s
        run reads as a flat line at zero. Scaling to the 98th percentile keeps
        the working range legible; the outlier is still drawn, just clipped, and
        the tile above always shows the true current rate.
        """
        if len(ys) < 8:
            return
        ordered = sorted(ys)
        cutoff = ordered[int(len(ordered) * 0.98)]
        if cutoff <= 0:
            return
        top = max(cutoff * 1.15, 1.0)
        # Only clamp when an outlier actually distorts the view; otherwise leave
        # pyqtgraph's own autorange alone.
        if ordered[-1] > top:
            self.plot.getPlotItem().setYRange(0, top)
        else:
            self.plot.getPlotItem().enableAutoRange(axis="y")

    @staticmethod
    def _duration(seconds: float) -> str:
        """Format a coarse human duration"""
        if seconds < 90:
            return f"{seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 90:
            return f"{minutes:.0f}m"
        hours = minutes / 60
        if hours < 48:
            return f"{hours:.1f}h"
        return f"{hours / 24:.1f}d"

    def clear(self) -> None:
        self.series.clear()
        self.total_steps = 0
        self.target_steps = 0
        self.rate = 0.0
        self.trusted = False
