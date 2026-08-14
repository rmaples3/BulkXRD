"""Session config and core utilities for the seriesxrd workflow.

This module is intentionally dependency-light. It can be imported before
pyFAI, fabio, matplotlib, or Tk are installed so the notebook can report
missing packages instead of crashing silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys

# Keep in sync with [project] version in pyproject.toml.
VERSION = "0.5.0"
TOOL_NAME = "SeriesXRD"

# Files that must exist inside a valid backend (seriesxrd package) folder.
_BACKEND_REQUIRED_FILES = ["calib/gui.py", "calib/processing.py", "calib/run_gui.py"]


def now_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def make_stdio_robust() -> None:
    """Stop a non-ASCII print from crashing a process on a legacy console.

    Windows consoles default to cp1252, which cannot encode the Greek letters,
    arrows, etc. that appear in our log lines, docstrings, and tracebacks — a
    stray one raises ``UnicodeEncodeError`` and aborts the run (and then the
    error handler, when it echoes the offending source line, crashes again).
    Switching the *error handler* to ``replace`` (without changing the encoding,
    so a parent process reading the pipe still decodes it the same way) turns an
    unencodable character into ``?`` instead of an exception. Best-effort; a noop
    where ``reconfigure`` is unavailable. Call once at a process entry point.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")            # TextIOWrapper, py3.7+
        except Exception:
            pass


def print_status(message: str, level: str = "INFO") -> None:
    line = f"[{now_iso()}] [{level}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Console can't encode some character (e.g. cp1252 + a Greek letter):
        # degrade it rather than abort the run.
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def safe_stem(value: str, default: str = "calibration") -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_-")
    return value or default


def ensure_dir(path: Path | str) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def write_json(path: Path | str, data: Dict[str, Any]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_json(path: Path | str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # File exists but is corrupt — preserve it and warn.
        ts = now_timestamp()
        corrupt_path = p.with_name(p.name + f".corrupt-{ts}")
        try:
            shutil.copy2(p, corrupt_path)
        except Exception:
            pass
        print_status(f"WARN: Could not parse JSON from {p}; corrupt copy saved to {corrupt_path}. Returning default.", "WARN")
        return dict(default or {})


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path | str, dst: Path | str, *, required: bool = True) -> Optional[Path]:
    s = Path(src).expanduser()
    d = Path(dst).expanduser()
    if not s.exists():
        if required:
            raise FileNotFoundError(f"Required source file not found: {s}")
        return None
    ensure_dir(d.parent)
    shutil.copy2(s, d)
    return d


def default_workspace_paths(base_dir: Path) -> Dict[str, str]:
    """Generates standard output directories relative to a given base workspace."""
    base = Path(base_dir).expanduser().resolve()
    return {
        "workspace_root": str(base),
        "raw_data_dir": str(base / "data" / "raw"),
        "processed_root": str(base / "data" / "processed"),
        "figures_root": str(base / "figures"),
        "metadata_root": str(base / "metadata"),
        "accepted_output_root": str(base / "accepted_calibrations"),
        "logs_root": str(base / "logs" / "seriesxrd"),
    }


def output_base(config: Dict[str, Any]) -> Path:
    """Safe base directory for outputs when a specific root isn't configured.

    Never returns the current working directory: the GUI and the worker both
    run with cwd set to the seriesxrd package folder, so a cwd fallback scatters
    output files into the installed package. Falls back to the session
    workspace, then the notebook dir, then a per-user sessions folder.
    """
    for key in ("workspace_root", "notebook_dir"):
        v = config.get(key)
        if v and str(v).strip():
            return Path(v).expanduser()
    return Path.home() / "seriesxrd_sessions"


def _backend_dir_valid(candidate: Path) -> bool:
    try:
        return all((candidate / rel).exists() for rel in _BACKEND_REQUIRED_FILES)
    except Exception:
        return False


def default_backend_dir_from_notebook(notebook_dir: Path) -> Path:
    # The package directory this module lives in is always a valid backend
    # when seriesxrd is imported from the repo or an installed copy.
    package_dir = Path(__file__).resolve().parents[1]
    candidates = [
        # Repository layout: workspace dir is a sibling of seriesxrd/
        notebook_dir.parent / "seriesxrd",
        notebook_dir / "seriesxrd",
        package_dir,
    ]
    for c in candidates:
        if _backend_dir_valid(c):
            return c.resolve()
    return package_dir


def default_python_exe() -> str:
    return sys.executable


@dataclass
class SessionConfig:
    version: str = VERSION
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    notebook_dir: str = ""
    backend_dir: str = ""
    python_exe: str = field(default_factory=default_python_exe)
    conda_exe: str = ""
    conda_env_name: str = ""
    dioptas_command: str = ""
    dioptas_python: str = ""
    dioptas_image_flip: bool = False
    workspace_root: str = ""
    raw_data_dir: str = ""
    processed_root: str = ""
    figures_root: str = ""
    metadata_root: str = ""
    accepted_output_root: str = ""
    logs_root: str = ""
    session_name: str = "xrd_calibration"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["updated_at"] = now_iso()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in fields}
        cfg = cls(**kwargs)
        return cfg

    @classmethod
    def default(cls, notebook_dir: Optional[Path] = None, backend_dir: Optional[Path] = None) -> "SessionConfig":
        nb = Path(notebook_dir or Path.cwd()).expanduser().resolve()
        
        paths = default_workspace_paths(base_dir=nb)
        
        cfg = cls(
            notebook_dir=str(nb),
            backend_dir=str((backend_dir or default_backend_dir_from_notebook(nb)).expanduser().resolve()),
            python_exe=default_python_exe(),
            workspace_root=paths["workspace_root"],
            raw_data_dir=paths["raw_data_dir"],
            processed_root=paths["processed_root"],
            figures_root=paths["figures_root"],
            metadata_root=paths["metadata_root"],
            accepted_output_root=paths["accepted_output_root"],
            logs_root=paths["logs_root"],
        )
        return cfg


def config_path_for_notebook(notebook_dir: Path | str) -> Path:
    return Path(notebook_dir).expanduser().resolve() / "calibration_session_config.json"


def load_session_config(notebook_dir: Path | str) -> SessionConfig:
    p = config_path_for_notebook(notebook_dir)
    if p.exists():
        return SessionConfig.from_dict(read_json(p))
    return SessionConfig.default(Path(notebook_dir))


def save_session_config(cfg: SessionConfig, path: Optional[Path | str] = None) -> Path:
    p = Path(path) if path else config_path_for_notebook(Path(cfg.notebook_dir or Path.cwd()))
    return write_json(p, cfg.to_dict())


def validate_session_config(cfg: SessionConfig) -> List[str]:
    problems: List[str] = []
    checks = {
        "notebook_dir": cfg.notebook_dir,
        "backend_dir": cfg.backend_dir,
        "python_exe": cfg.python_exe,
        "raw_data_dir": cfg.raw_data_dir,
        "processed_root": cfg.processed_root,
        "figures_root": cfg.figures_root,
        "metadata_root": cfg.metadata_root,
        "accepted_output_root": cfg.accepted_output_root,
    }
    for key, value in checks.items():
        if not value:
            problems.append(f"Missing {key}")
            continue
        p = Path(value).expanduser()
        if key == "python_exe":
            if not p.exists() or not p.is_file():
                problems.append(f"Python executable does not exist: {p}")
        elif key in {"backend_dir", "notebook_dir"}:
            if not p.exists() or not p.is_dir():
                problems.append(f"{key} does not exist: {p}")
        else:
            # Output dirs may not exist yet; caller can create them.
            pass
    backend = Path(cfg.backend_dir) if cfg.backend_dir else None
    if backend and backend.exists():
        for required in _BACKEND_REQUIRED_FILES:
            if not (backend / required).exists():
                problems.append(f"Backend folder is missing required file: {required}")
    return problems
