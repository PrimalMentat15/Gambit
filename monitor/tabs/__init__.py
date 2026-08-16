"""
Tab widgets

Full-page views, as distinct from the dockable panels in ``monitor.panels``
which compose the Live tab.
"""

from .analysis import AnalysisTab
from .control import ControlTab

__all__ = ["AnalysisTab", "ControlTab"]
