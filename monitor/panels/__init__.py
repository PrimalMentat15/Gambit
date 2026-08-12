"""
Panel registry

Panels are discovered by scanning this package, so adding a chart is adding a
file here -- no registration list to update, no shell code to touch.
"""

import importlib
import pkgutil
from typing import Dict, List, Type

from .base import Panel, RingSeries, StatTile

__all__ = ["Panel", "RingSeries", "StatTile", "discover", "build_panels"]

# Default display order. Panels found but not listed here are appended
# alphabetically, so a newly dropped-in file still shows up.
DEFAULT_ORDER = [
    "throughput",
    "latency_action",
    "latency_time",
    "reward",
    "winrate",
    "gamestate",
    "actions",
    "log",
]


def discover() -> Dict[str, Type[Panel]]:
    """
    Import every module in this package and collect Panel subclasses

    Returns:
        Mapping of panel NAME to class
    """
    found: Dict[str, Type[Panel]] = {}

    for info in pkgutil.iter_modules(__path__):
        if info.name in ("base",):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, Panel)
                and attr is not Panel
                and attr.NAME != Panel.NAME
            ):
                found[attr.NAME] = attr

    return found


def build_panels(config) -> List[Panel]:
    """
    Instantiate the panels the config asks for, in display order

    An empty ``config.panels`` means every discovered panel.
    """
    available = discover()

    if config.panels:
        wanted = [name for name in config.panels if name in available]
    else:
        ordered = [n for n in DEFAULT_ORDER if n in available]
        extra = sorted(n for n in available if n not in DEFAULT_ORDER)
        wanted = ordered + extra

    return [available[name](config) for name in wanted]
