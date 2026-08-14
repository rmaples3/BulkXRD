"""Public series-ordering API shared across stages.

One pipeline, one reading of the ordering metadata: these helpers resolve a
frame series' physical axis (``frame`` | ``pressure`` | ``temperature`` |
``time``) and its scan/folder grouping from an open Analysis HDF5. They are
implemented in :mod:`seriesxrd.analysis.unknowns` (Step 3c grew them first)
and re-exported here as the supported cross-module names — Step 2 seed
propagation and the correlations stage order frames through this module
instead of hand-rolling further ``/frames`` readers.
"""
from __future__ import annotations

from .unknowns import (
    TRACKING_AXES,
    TRACKING_GROUPS,
    _tracking_groups as tracking_groups,
    _tracking_values as tracking_values,
)

__all__ = [
    "TRACKING_AXES",
    "TRACKING_GROUPS",
    "tracking_groups",
    "tracking_values",
]
