"""
Win rate panel

Two win rates against the *current* curriculum goal, which is why they belong
on one axis and why the ante-8 milestone does not join them here -- it answers a
different question and has its own panel.

- **training** — sampled play, every episode the trainer collected.
- **promotion eval** — argmax play on held-out seeds. This is the one the
  trainer actually promotes on, and it sits above the training line by however
  much sampling noise is costing.

Both step down every time the ladder promotes, because the goal moved. That
discontinuity is the signal, not an artefact, so promotions are drawn as rules
on the same axis rather than smoothed over.
"""

from typing import Any, Dict

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from .base import Panel, RingSeries, StatTile


class WinRatePanel(Panel):
    """Training and promotion-eval win rates at the current curriculum goal"""

    NAME = "winrate"
    TITLE = "Win rate"
    EVENT_TYPES = frozenset(
        {"rollout", "promotion_eval", "curriculum_promotion"}
    )
    SIZE = (480, 260)

    def __init__(self, config, parent=None):
        super().__init__(config, parent)
        self.training = RingSeries(config.history)
        self.promotion = RingSeries(config.history)
        self.promotions: list = []
        self.win_ante = None
        self.latest_promo = None
        self._drawn_promotions = 0

    def build(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tiles = QWidget()
        row = QHBoxLayout(tiles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tile_goal = StatTile("curriculum goal")
        self.tile_promo = StatTile("promo eval")
        row.addWidget(self.tile_goal)
        row.addWidget(self.tile_promo)

        self.plot = theme.make_plot(y_label="win rate (%)", x_label="env step")
        self.plot.getPlotItem().setYRange(0, 100)
        theme.legend(self.plot)

        self.training_curve = self.plot.plot(
            [], [], pen=theme.pen(theme.SERIES[0]), name="training (goal)"
        )
        self.promotion_curve = self.plot.plot(
            [], [],
            pen=theme.pen(theme.SERIES[1]),
            symbol="o",
            symbolSize=theme.MARKER_SIZE - 3,
            symbolBrush=theme.SERIES[1],
            symbolPen=pg.mkPen(None),
            name="promo eval (goal)",
        )
        self.crosshair = theme.Crosshair(
            self.plot, lambda x, y: f"step {x:,.0f}   {y:.1f}%"
        )

        layout.addWidget(tiles)
        layout.addWidget(self.plot, 1)
        return container

    def on_event(self, event: Dict[str, Any]) -> None:
        data = event["data"]
        kind = event["type"]

        if kind == "rollout":
            step, rate = data.get("global_step"), data.get("ep_win_rate")
            if step is not None and rate is not None:
                self.training.append(step, rate * 100.0)
            if data.get("win_ante") is not None:
                self.win_ante = data["win_ante"]

        elif kind == "promotion_eval":
            step, rate = data.get("step"), data.get("win_rate")
            if step is not None and rate is not None:
                self.promotion.append(step, rate * 100.0)
                self.latest_promo = rate * 100.0

        elif kind == "curriculum_promotion":
            if data.get("step") is not None:
                self.promotions.append((data["step"], data.get("to_ante")))

    def redraw(self) -> None:
        for curve, series in (
            (self.training_curve, self.training),
            (self.promotion_curve, self.promotion),
        ):
            xs, ys = series.arrays(self.config.max_plot_points)
            curve.setData(xs, ys)

        px, py = self.promotion.arrays(self.config.max_plot_points)
        self.crosshair.set_series(px, py)

        # Promotions only ever accumulate, so redraw adds the new ones rather
        # than rebuilding every rule on each repaint.
        item = self.plot.getPlotItem()
        for step, ante in self.promotions[self._drawn_promotions:]:
            rule = pg.InfiniteLine(
                pos=step,
                angle=90,
                pen=pg.mkPen(theme.AXIS, width=1, style=Qt.DashLine),
                label=f"→{ante}",
                labelOpts={"color": theme.MUTED, "position": 0.95},
            )
            item.addItem(rule, ignoreBounds=True)
        self._drawn_promotions = len(self.promotions)

        self.tile_goal.set_value(
            f"ante {self.win_ante}" if self.win_ante is not None else "--"
        )

        # No status colour: this rate is relative to whichever rung is current,
        # so a fixed threshold would mean something different at each one.
        self.tile_promo.set_value(
            f"{self.latest_promo:.1f}%" if self.latest_promo is not None else "--"
        )

    def clear(self) -> None:
        self.training.clear()
        self.promotion.clear()
        self.promotions.clear()
        self._drawn_promotions = 0
        self.win_ante = None
        self.latest_promo = None
        if self.widget is not None:
            item = self.plot.getPlotItem()
            for child in list(item.items):
                if isinstance(child, pg.InfiniteLine):
                    item.removeItem(child)
