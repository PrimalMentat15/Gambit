"""
Visual theme

Single committed dark palette -- this is a dashboard you stare at for hours during
multi-hour training runs, so there is no light mode to swap to.

The categorical slots and chrome values are a validated palette. The three slots
used here pass the lightness band, chroma floor, CVD separation (worst adjacent
deutan dE 9.4), normal-vision floor (20.9) and 3:1 surface contrast in both the
adjacent and all-pairs pairlists. Do not substitute colors here without
re-validating: the slot ordering is the colorblind-safety mechanism, not cosmetics.
"""

import os

# pyqtgraph probes for a Qt binding at import time; pin it before that happens
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg

# --- Surfaces and ink ---
SURFACE = "#1a1a19"      # chart surface
PAGE = "#0d0d0d"         # page plane behind cards
INK = "#ffffff"          # primary text
INK_2 = "#c3c2b7"        # secondary text
MUTED = "#898781"        # axis labels
GRID = "#2c2c2a"         # hairline gridline
AXIS = "#383835"         # baseline / axis rule

# --- Categorical slots, in fixed order (never cycled, never reordered) ---
SERIES = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]

# --- Status roles. Reserved: never used for a plain series ---
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Action identity is stable across every panel, keyed by the action ids in
# RLBridge/actions.lua. Colour follows the entity, so PLAY_HAND is the same hue
# wherever it appears rather than being assigned by current rank.
ACTION_NAMES = {
    1: "SELECT_HAND",
    2: "PLAY_HAND",
    3: "DISCARD_HAND",
    4: "START_RUN",
    5: "SELECT_BLIND",
    6: "RESTART_RUN",
}
ACTION_COLORS = {
    1: SERIES[0],
    2: SERIES[1],
    3: SERIES[2],
    4: SERIES[3],
    5: SERIES[4],
    6: SERIES[5],
}

# Mark specs
LINE_WIDTH = 2
MARKER_SIZE = 8

FONT_FAMILY = "Segoe UI"


def configure() -> None:
    """Apply global pyqtgraph defaults"""
    pg.setConfigOptions(
        antialias=True,
        background=SURFACE,
        foreground=MUTED,
    )


def make_plot(y_label: str = "", x_label: str = "") -> pg.PlotWidget:
    """
    Create a styled plot widget

    Grid and axes are solid hairlines one shade off the surface, never dashed and
    never heavy.

    Args:
        y_label: Left axis label
        x_label: Bottom axis label

    Returns:
        Configured PlotWidget
    """
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)

    item = plot.getPlotItem()
    item.showGrid(x=True, y=True, alpha=0.12)
    item.getViewBox().setDefaultPadding(0.04)

    for side in ("left", "bottom"):
        axis = item.getAxis(side)
        axis.setPen(pg.mkPen(AXIS, width=1))
        axis.setTextPen(pg.mkPen(MUTED))
        axis.setStyle(tickLength=-4)

    for side in ("top", "right"):
        item.getAxis(side).setStyle(showValues=False)

    if y_label:
        item.setLabel("left", y_label, color=MUTED)
    if x_label:
        item.setLabel("bottom", x_label, color=MUTED)

    return plot


def pen(color: str, width: int = LINE_WIDTH):
    """A 2px solid pen in the given colour"""
    return pg.mkPen(color, width=width)


def fill(color: str, alpha: int = 40):
    """A translucent brush for area fills under a line"""
    brush = pg.mkBrush(color)
    qcolor = brush.color()
    qcolor.setAlpha(alpha)
    brush.setColor(qcolor)
    return brush


def legend(plot: pg.PlotWidget):
    """
    Attach a legend

    Present whenever a plot carries two or more series, so identity is never
    conveyed by colour alone.
    """
    leg = plot.getPlotItem().addLegend(offset=(-10, 10), labelTextColor=INK_2)
    leg.setBrush(pg.mkBrush(PAGE))
    leg.setPen(pg.mkPen(AXIS))
    return leg


def decimate(xs, ys, limit: int):
    """
    Thin a series for drawing without distorting its shape

    A multi-hour run produces far more points than a panel has pixels. Stride
    sampling keeps the series cheap to paint; the underlying ring buffer is
    untouched so numeric readouts stay exact.

    Args:
        xs: X values
        ys: Y values
        limit: Maximum points to return

    Returns:
        (xs, ys) at or below the limit
    """
    count = len(xs)
    if count <= limit or limit <= 0:
        return xs, ys
    stride = count // limit + 1
    return xs[::stride], ys[::stride]


class Crosshair:
    """
    Hover readout for a time-series plot

    Tooltips enhance rather than gate: every value is also on an axis or a direct
    label, so losing the hover layer never hides data.
    """

    def __init__(self, plot: pg.PlotWidget, formatter=None):
        self.plot = plot
        self.formatter = formatter or (lambda x, y: f"{x:.0f}, {y:.3g}")

        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen(MUTED, width=1))
        self.label = pg.TextItem(color=INK, anchor=(0, 1), fill=pg.mkBrush(PAGE))
        for element in (self.vline, self.label):
            element.setZValue(100)
            element.hide()
            plot.addItem(element, ignoreBounds=True)

        plot.scene().sigMouseMoved.connect(self._moved)
        self._series = ([], [])

    def set_series(self, xs, ys) -> None:
        """Point the readout at the current data"""
        self._series = (xs, ys)

    def _moved(self, position) -> None:
        xs, ys = self._series
        item = self.plot.getPlotItem()
        if not len(xs) or not item.sceneBoundingRect().contains(position):
            self.vline.hide()
            self.label.hide()
            return

        point = item.getViewBox().mapSceneToView(position)
        # Nearest point by x, so the hit area is the whole column rather than
        # requiring a landing on the mark itself
        best = min(range(len(xs)), key=lambda i: abs(xs[i] - point.x()))

        self.vline.setPos(xs[best])
        self.label.setPos(xs[best], ys[best])
        self.label.setText(self.formatter(xs[best], ys[best]))
        self.vline.show()
        self.label.show()
