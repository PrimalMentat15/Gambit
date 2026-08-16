"""
Anneal panel

The two schedules that quietly change what every other number means: the reward
shaping weight (beta) and the fraction of episodes started from a mid-run
snapshot.

Worth its own view because both are confounders, not achievements. A return
curve that climbs while beta is still high is partly measuring shaping, not
skill; a win rate measured while snapshot mixing is on is measuring a different,
easier population of episodes. Having the schedules on screen next to the
curves is what stops either from being misread.

Both are fractions in [0, 1], so they share one axis honestly.
"""

from typing import Any, Dict

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class AnnealPanel(Panel):
    """Shaping beta and snapshot fraction over training steps"""

    NAME = "anneal"
    TITLE = "Anneals"
    EVENT_TYPES = frozenset({"rollout"})
    SIZE = (420, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.beta = RingSeries(config.history)
        self.snapshot = RingSeries(config.history)
        self.from_snapshot = RingSeries(config.history)
        self.latest: Dict[str, Any] = {}

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_beta = StatTile("shaping beta")
        self.tile_snapshot = StatTile("snapshot fraction")
        row.addWidget(self.tile_beta)
        row.addWidget(self.tile_snapshot)

        self.plot = theme.make_plot(y_label="fraction", x_label="env step")
        self.plot.getPlotItem().setYRange(0, 1.05)
        theme.legend(self.plot)

        self.beta_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[0]), name="shaping beta"
        )
        self.snapshot_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[1]), name="snapshot fraction"
        )
        # The scheduled fraction is what was asked for; the observed share is
        # what the envs actually delivered. They diverge when the pool has no
        # eligible entries, which is worth seeing rather than assuming.
        self.observed_curve = self.plot.plot(
            [], [],
            pen=theme.pen(theme.SERIES[4], width=1),
            name="observed share",
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:.3f}"
        )

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        step = data.get("global_step")
        if step is None:
            return
        self.latest = data

        for series, key in (
            (self.beta, "shaping_beta"),
            (self.snapshot, "snapshot_fraction"),
            (self.from_snapshot, "ep_from_snapshot_share"),
        ):
            if data.get(key) is not None:
                series.append(step, data[key])

    def redraw(self) -> None:
        for curve, series in (
            (self.beta_curve, self.beta),
            (self.snapshot_curve, self.snapshot),
            (self.observed_curve, self.from_snapshot),
        ):
            xs, ys = series.arrays(self.config.max_plot_points)
            curve.setData(xs, ys)

        bx, by = self.beta.arrays(self.config.max_plot_points)
        self.crosshair.set_series(bx, by)

        beta = self.latest.get("shaping_beta")
        self.tile_beta.set_value(f"{beta:.3f}" if beta is not None else "--")

        fraction = self.latest.get("snapshot_fraction")
        self.tile_snapshot.set_value(
            f"{fraction:.3f}" if fraction is not None else "off"
        )

    def clear(self) -> None:
        self.beta.clear()
        self.snapshot.clear()
        self.from_snapshot.clear()
        self.latest = {}
