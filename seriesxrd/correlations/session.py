"""Correlation-stage session configuration.

Dependency-light by design: importing or seeding this module never imports
Tk, Matplotlib, h5py, or the numerical correlation implementation.
"""
from __future__ import annotations

from pathlib import Path

from ..core.config import (
    default_python_exe,
    default_workspace_paths,
    now_iso,
    read_json,
    write_json,
)


CONFIG_FILENAME = "correlation_session_config.json"
_ANALYSIS_CONFIG_FILENAME = "analysis_session_config.json"

_DEFAULTS = {
    "session_name": "correlations",
    "analysis_h5_file": "",
    "sample_type": "powder",
    # The scientific pipeline is intentionally Log²-only. This persisted
    # marker is provenance, not a user-selectable transform switch.
    "transform": "log_squared",
    "radial_min": "",
    "radial_max": "",
    "window_width": "5.0",
    "window_step": "1.0",
    "location_tolerance": "0.02",
    "scale_quantile": "0.995",
    "order_by": "frame",
    "skip_plots": False,
    "max_anchor_plots": "",
    "make_tracks": True,
    "track_min_similarity": "0.2",
    "track_min_frames": "3",
    "track_link_tol_fwhm": "1.5",
    "track_max_gap": "2",
    "track_group_by": "none",
}


def correlation_config_path(workspace_dir: "str | Path") -> Path:
    """Return the correlation-stage config path for ``workspace_dir``."""
    return Path(workspace_dir).expanduser().resolve() / CONFIG_FILENAME


def seed_correlation_config(workspace_dir: "str | Path") -> Path:
    """Create or refresh the workspace's correlation-stage config.

    A completed Analysis HDF5 is discovered from
    ``analysis_session_config.json`` when the correlation config has no input
    yet. Existing values, including a user-selected analysis file, are never
    overwritten.
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    config_path = correlation_config_path(workspace)
    config = read_json(config_path)

    defaults = dict(_DEFAULTS)
    defaults.update({
        "created_at": now_iso(),
        "workspace_root": str(workspace),
        "backend_dir": str(Path(__file__).resolve().parents[1]),
        "python_exe": default_python_exe(),
        "logs_root": default_workspace_paths(workspace)["logs_root"],
        "result_root": str(workspace / "correlations"),
    })
    for key, value in defaults.items():
        config.setdefault(key, value)
    config.setdefault(
        "source",
        "spots" if config.get("sample_type") == "single_crystal" else "fit",
    )

    if not str(config.get("analysis_h5_file", "") or "").strip():
        analysis_config = read_json(workspace / _ANALYSIS_CONFIG_FILENAME)
        candidate = str(analysis_config.get("analysis_h5_file", "") or "").strip()
        if candidate and Path(candidate).expanduser().is_file():
            config["analysis_h5_file"] = candidate

    # This is the location of this config, not a user scientific setting.
    config["session_config_path"] = str(config_path)
    write_json(config_path, config)
    return config_path
