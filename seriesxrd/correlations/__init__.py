"""Correlation maps for SeriesXRD analysis files.

The package is deliberately independent of the historical research prototype:
it consumes the public Analysis-HDF5 contract and exposes headless numerical
and plotting APIs suitable for both the CLI and the desktop application.
"""
from __future__ import annotations

from .processing import (
    SCHEMA_VERSION,
    directional_anchor_iou,
    integrated_iou,
    location_similarity,
    log_squared_transform,
    relative_feature_similarity,
    run_correlations,
)
from .review import inspect_correlations, load_anchor_map

__all__ = [
    "SCHEMA_VERSION",
    "directional_anchor_iou",
    "inspect_correlations",
    "integrated_iou",
    "load_anchor_map",
    "location_similarity",
    "log_squared_transform",
    "relative_feature_similarity",
    "run_correlations",
]
