"""SeriesXRD calibrates, reduces, analyzes, and correlates XRD series.

The workflow is organized as one subpackage per pipeline stage, with shared
infrastructure in `core` (session config, IO, naming, masks, env checks) and
`guikit` (shared Tk/matplotlib theming):

    seriesxrd.core      shared utilities (stdlib + numpy only)
    seriesxrd.guikit    shared GUI/plot theme
    seriesxrd.calib     calibration review (pyFAI QA GUI + worker)
    seriesxrd.reduce    dataset reduction and batch integration
    seriesxrd.analysis  fitting, identification, mapping, and exports

The light, dependency-free core API is re-exported here; stage APIs are
imported from their subpackage, e.g. `from seriesxrd.calib import run_app`.
"""
from __future__ import annotations

from .core import (
    VERSION,
    TOOL_NAME,
    SessionConfig,
    config_path_for_notebook,
    copy_file,
    default_backend_dir_from_notebook,
    default_python_exe,
    default_workspace_paths,
    ensure_dir,
    json_default,
    load_session_config,
    now_iso,
    now_timestamp,
    print_status,
    read_json,
    safe_stem,
    save_session_config,
    sha256_file,
    validate_session_config,
    write_json,
    OPTIONAL_IMPORTS,
    REQUIRED_IMPORTS,
    DependencyStatus,
    check_dependencies,
    find_conda_exe,
    package_install_command,
    run_install_command,
    gen_label,
    generation_paths,
    generation_stem,
    next_available_path,
)

__version__ = VERSION
