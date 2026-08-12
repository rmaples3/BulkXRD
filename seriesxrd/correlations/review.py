"""Read-only inspection helpers for sample-specific correlation artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def inspect_correlations(path: str | Path) -> Dict[str, Any]:
    """Return a defensive schema summary without loading large matrices."""

    source = Path(path).expanduser().resolve()
    result: Dict[str, Any] = {
        "path": str(source),
        "ok": False,
        "schema_version": "",
        "sample_type": "",
        "source_resolved": "",
        "n_frames": 0,
        "n_peaks": 0,
        "n_windows": 0,
        "transform": {},
        "products": [],
        "anomalies": [],
    }
    if not source.is_file():
        result["anomalies"].append(f"File does not exist: {source}")
        return result
    try:
        import h5py  # type: ignore

        with h5py.File(str(source), "r") as h5:
            result.update(
                {
                    "schema_version": str(h5.attrs.get("schema_version", "")),
                    "sample_type": str(h5.attrs.get("sample_type", "")),
                    "source_resolved": str(h5.attrs.get("source_resolved", "")),
                    "n_frames": int(h5.attrs.get("n_frames", 0)),
                    "n_peaks": int(h5.attrs.get("n_peaks", 0)),
                    "n_windows": int(h5.attrs.get("n_windows", 0)),
                }
            )
            # Absent on artifacts written before /peaks/valid existed --
            # never an anomaly.
            if "peaks/valid" in h5:
                result["n_anchors_valid"] = int(
                    np.count_nonzero(np.asarray(h5["peaks/valid"][:], bool))
                )
            # Same for /tracks: older or --no-tracks artifacts open cleanly.
            if "tracks" in h5:
                result["n_tracks"] = int(h5["tracks"].attrs.get("n_tracks", 0))
                if "tracks/intervals/transition_candidate" in h5:
                    result["n_transition_candidates"] = int(
                        np.count_nonzero(
                            np.asarray(
                                h5["tracks/intervals/transition_candidate"][:],
                                bool,
                            )
                        )
                    )
            required = (
                "patterns/original_positive",
                "patterns/log_squared",
                "patterns/log_squared_signed",
                "peaks/id",
                "anchor_maps/roi_area",
                "anchor_maps/location",
                "windows/acf_features",
                "windows/across_direct",
                "windows/across_acf",
                "windows/within_acf",
            )
            for name in required:
                if name not in h5:
                    result["anomalies"].append(f"Missing /{name}")
                else:
                    result["products"].append(name)
            if "transform" in h5:
                result["transform"] = {
                    key: h5["transform"].attrs[key] for key in h5["transform"].attrs
                }
                if str(result["transform"].get("method", "")) != "log_squared":
                    result["anomalies"].append("Transform is not fixed Log-squared")
            else:
                result["anomalies"].append("Missing /transform")
            result["ok"] = not result["anomalies"]
    except Exception as exc:
        result["anomalies"].append(f"Unreadable correlation HDF5: {exc}")
    return result


def load_anchor_map(
    path: str | Path,
    kind: str,
    anchor: int,
) -> Dict[str, Any]:
    """Load one all-peak anchor vector and its frame/slot target grid."""

    if kind not in ("roi_area", "location"):
        raise ValueError("kind must be 'roi_area' or 'location'")
    import h5py  # type: ignore

    source = Path(path).expanduser().resolve()
    with h5py.File(str(source), "r") as h5:
        matrix = h5[f"anchor_maps/{kind}"]
        index = int(anchor)
        if index < 0 or index >= matrix.shape[0]:
            raise IndexError(f"anchor {index} outside 0..{matrix.shape[0] - 1}")
        row = np.asarray(matrix[index], dtype=float)
        frame = np.asarray(h5["peaks/frame_row"][:], dtype=int)
        slot = np.asarray(h5["peaks/local_peak"][:], dtype=int)
        n_frames = int(h5.attrs["n_frames"])
        n_slots = max(int(slot.max()) + 1 if slot.size else 1, 1)
        grid = np.full((n_frames, n_slots), np.nan, dtype=float)
        grid[frame, slot] = row
        return {
            "kind": kind,
            "anchor": index,
            "vector": row,
            "grid": grid,
            "anchor_frame": int(h5["peaks/original_frame"][index]),
            "anchor_local_peak": int(h5["peaks/local_peak"][index]),
            "anchor_center": float(h5["peaks/center"][index]),
            "anchor_pressure": float(h5["peaks/pressure"][index]),
        }


__all__ = ["inspect_correlations", "load_anchor_map"]
