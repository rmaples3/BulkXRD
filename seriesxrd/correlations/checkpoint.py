"""Render-checkpoint state for the correlation stage.

Figure export renders into a hidden staging directory and swaps it into
place atomically. That staging directory doubles as the checkpoint: a
figure is done exactly when its PNG exists, because :func:`plots.save_figure`
writes to a temporary name and renames, so a half-written PNG cannot exist.
Nothing separate has to record what finished.

This module adds the small amount of state that files alone cannot express:
whether a staging directory belongs to a run that is still going, or was
abandoned by one that died. An abandoned tree holds perfectly good figures
and is offered for recovery; a live one is left alone.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from ..core.config import now_iso

STATE_FILENAME = ".render_state.json"
INCOMPLETE_FILENAME = "INCOMPLETE.json"
#: A heartbeat older than this means the writing process is gone. Renders
#: touch the state file every progress tick, so the bar is generous.
STALE_AFTER_SECONDS = 120.0


def staging_prefix(sample_type: str) -> str:
    return f".{sample_type}.tmp-"


def write_state(staging: Path, **fields: Any) -> Path:
    """Merge fields into the staging state file; also the heartbeat.

    Merging matters: a progress tick writes only ``status``/``done``, and
    must not erase the identity a resume needs to decide whether this
    checkpoint matches the run being restarted.
    """
    target = Path(staging) / STATE_FILENAME
    payload = {
        **read_state(staging),
        "updated_at": now_iso(),
        "pid": os.getpid(),
        **fields,
    }
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        # State is an aid to recovery, never a reason to fail a render.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def read_state(staging: Path) -> Dict[str, Any]:
    """State for a staging directory; ``{}`` when absent or unreadable.

    Directories written before this module existed simply have no state,
    and are judged by their modification time instead.
    """
    try:
        raw = (Path(staging) / STATE_FILENAME).read_text(encoding="utf-8")
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _newest_mtime(path: Path) -> float:
    newest = 0.0
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest


def is_live(staging: Path, *, stale_after: float = STALE_AFTER_SECONDS) -> bool:
    """Whether a render is still writing into this staging directory.

    Judged by heartbeat freshness rather than by the pid alone: pids are
    reused, and a directory left by an older version has no pid recorded.
    A same-process directory is always live (we are the writer).
    """
    staging = Path(staging)
    state = read_state(staging)
    if state.get("status") in ("interrupted", "complete"):
        return False
    if int(state.get("pid", -1)) == os.getpid():
        return True
    age = time.time() - _newest_mtime(staging)
    return age < float(stale_after)


def find_staging(result_root: Path, sample_type: str) -> List[Path]:
    """Staging directories for one sample type, newest first."""
    root = Path(result_root) / "heatmaps"
    if not root.is_dir():
        return []
    prefix = staging_prefix(sample_type)
    found = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    return sorted(found, key=_newest_mtime, reverse=True)


def find_recoverable(result_root: Path) -> List[Dict[str, Any]]:
    """Abandoned staging trees that still hold renderable figures.

    Each entry reports the sample type, the directory, how many PNGs
    survived, and what the writing run had planned, so the caller can say
    "4120 of 6183 figures" rather than just "some files".
    """
    out: List[Dict[str, Any]] = []
    for sample_type in ("powder", "single_crystal"):
        for staging in find_staging(Path(result_root), sample_type):
            if is_live(staging):
                continue
            pngs = [p for p in staging.rglob("*.png") if p.is_file()]
            if not pngs:
                continue
            state = read_state(staging)
            out.append(
                {
                    "sample_type": sample_type,
                    "staging": staging,
                    "n_figures": len(pngs),
                    "planned": int(state.get("planned", 0) or 0),
                    "status": str(state.get("status", "") or "unknown"),
                    "updated_at": str(state.get("updated_at", "") or ""),
                }
            )
    return out


def publish_staging(staging: Path, destination: Path) -> None:
    """Swap a staging tree into place, keeping the sibling sample untouched.

    Same move the renderer makes on success: back the old tree up, rename
    the new one in, then drop the backup — and put the old one back if the
    rename fails.
    """
    staging = Path(staging)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.old-{os.getpid()}")
    _remove(backup)
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    _remove(backup)


def mark_incomplete(
    destination: Path,
    *,
    n_figures: int,
    planned: int,
    source: str,
    artifact: str = "",
) -> Path:
    """Record that a published tree came from an interrupted render.

    Every other output in this pipeline says how it was made; a promoted
    partial tree must not be the exception, or a folder of figures that is
    silently missing most of its anchors looks like a complete result.
    """
    target = Path(destination) / INCOMPLETE_FILENAME
    payload = {
        "incomplete": True,
        "reason": "promoted from an interrupted render",
        "n_figures": int(n_figures),
        "planned": int(planned),
        "recovered_from": str(source),
        "artifact": str(artifact),
        "created_at": now_iso(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return target


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


__all__ = [
    "INCOMPLETE_FILENAME",
    "STALE_AFTER_SECONDS",
    "STATE_FILENAME",
    "find_recoverable",
    "find_staging",
    "is_live",
    "mark_incomplete",
    "publish_staging",
    "read_state",
    "staging_prefix",
    "write_state",
]
