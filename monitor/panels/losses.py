"""
PPO health panel

The four update-side numbers that say whether the optimisation itself is
healthy, independent of whether the agent is winning: approximate KL, clip
fraction, policy entropy, and value loss.

Two plots rather than one, because the units do not share a scale. KL and clip
fraction are both small dimensionless ratios and belong together -- two readings
of the same thing, how far the update moved. Entropy is in nats and gets its own
axis. Value loss is in squared return units, whose magnitude depends on the
reward scale and so says little as a curve next to either; it is a tile.

This is the panel that distinguishes "learning slowly" from "collapsed": entropy
falling to zero, KL spiking, or clip fraction pinned high are all failures a
reward curve alone reports only as a flat line, hours later.
"""

from typing import Any, Dict

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class LossesPanel(Panel):
    """KL, clip fraction, entropy and value loss over training steps"""

    NAME = "losses"
    TITLE = "PPO health"
    EVENT_TYPES = frozenset({"rollout"})
    SIZE = (480, 320)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.kl = RingSeries(config.history)
        self.clipfrac = RingSeries(config.history)
        self.entropy = RingSeries(config.history)
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
        self.tile_entropy = StatTile("entropy")
        self.tile_kl = StatTile("approx KL")
        self.tile_v_loss = StatTile("value loss")
        for tile in (self.tile_entropy, self.tile_kl, self.tile_v_loss):
            row.addWidget(tile)

        self.update_plot = theme.make_plot(y_label="ratio", x_label="")
        theme.legend(self.update_plot)
        self.kl_curve = self.update_plot.plot(
            [], [], pen=theme.pen(theme.SERIES[0]), name="approx KL"
        )
        self.clip_curve = self.update_plot.plot(
            [], [], pen=theme.pen(theme.SERIES[1]), name="clip fraction"
        )

        # No legend: one series, named by the axis. A legend box here would only
        # cover the curve it labels.
        self.policy_plot = theme.make_plot(y_label="entropy (nats)", x_label="env step")
        self.entropy_curve = self.policy_plot.plot(
            [], [], pen=theme.pen(theme.SERIES[2])
        )

        layout.addWidget(tiles)
        layout.addWidget(self.update_plot, 1)
        layout.addWidget(self.policy_plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        step = data.get("global_step")
        if step is None:
            return
        self.latest = data
        for series, key in (
            (self.kl, "approx_kl"),
            (self.clipfrac, "clipfrac"),
            (self.entropy, "entropy"),
        ):
            if data.get(key) is not None:
                series.append(step, data[key])

    def redraw(self) -> None:
        for curve, series in (
            (self.kl_curve, self.kl),
            (self.clip_curve, self.clipfrac),
            (self.entropy_curve, self.entropy),
        ):
            xs, ys = series.arrays(self.config.max_plot_points)
            curve.setData(xs, ys)

        for tile, key, fmt in (
            (self.tile_entropy, "entropy", "{:.3f}"),
            (self.tile_kl, "approx_kl", "{:.5f}"),
            (self.tile_v_loss, "v_loss", "{:.4f}"),
        ):
            value = self.latest.get(key)
            tile.set_value(fmt.format(value) if value is not None else "--")

    def clear(self) -> None:
        for series in (self.kl, self.clipfrac, self.entropy):
            series.clear()
        self.latest = {}
