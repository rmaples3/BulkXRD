"""Machine-readable CSV exports of correlation artifacts.

Every plotted correlation result already has its full numerical form in the
sample-specific HDF5; these exports give the testbase the same numbers in a
spreadsheet-friendly shape, written through the shared
:func:`seriesxrd.core.io.write_table_csv`. Summary exports are compact
(per-anchor top matches, long-format window tables, tracks, transition
intervals); the full K x K matrix dumps are separate and on demand because
they grow quadratically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..core.io import write_table_csv

DEFAULT_TOP_MATCHES = 5


def _decode(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(
        value, (bytes, bytearray)
    ) else str(value)


def _cell(value: Any) -> Any:
    """NaN-safe CSV cell: non-finite numerics become empty cells."""
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def export_summary_csvs(
    correlations_h5: "str | Path",
    out_dir: "str | Path",
    *,
    top_n: int = DEFAULT_TOP_MATCHES,
) -> List[Path]:
    """Write the summary CSV set beside the correlation artifact."""

    import h5py  # type: ignore

    source = Path(correlations_h5).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    with h5py.File(str(source), "r") as h5:
        peaks = h5["peaks"]
        roi = np.asarray(h5["anchor_maps/roi_area"][:], dtype=float)
        count = int(peaks["id"].shape[0])
        valid = (
            np.asarray(peaks["valid"][:], bool)
            if "valid" in peaks
            else np.ones(count, dtype=bool)
        )
        columns = {
            name: np.asarray(peaks[name][:])
            for name in (
                "id", "source_index", "original_frame", "frame_row",
                "local_peak", "center", "width", "half_width", "area",
                "pressure", "track",
            )
        }
        anchor_rows: List[Dict[str, Any]] = []
        top_n = max(int(top_n), 0)
        for index in range(count):
            row: Dict[str, Any] = {
                name: _cell(values[index]) for name, values in columns.items()
            }
            row["valid"] = bool(valid[index])
            scores = roi[index]
            order = np.argsort(-np.nan_to_num(scores, nan=-1.0))
            rank = 0
            for target in order:
                if rank >= top_n:
                    break
                score = float(scores[target])
                if not np.isfinite(score) or not valid[target]:
                    continue
                rank += 1
                row[f"match{rank}_anchor"] = int(target)
                row[f"match{rank}_score"] = score
                row[f"match{rank}_frame"] = _cell(
                    columns["original_frame"][target]
                )
            anchor_rows.append(row)
        written.append(
            write_table_csv(destination / "anchors_summary.csv", anchor_rows)
        )

        starts = np.asarray(h5["windows/start"][:], dtype=float)
        ends = np.asarray(h5["windows/end"][:], dtype=float)
        across_direct = np.asarray(h5["windows/across_direct"][:], float)
        across_acf = np.asarray(h5["windows/across_acf"][:], float)
        within_acf = np.asarray(h5["windows/within_acf"][:], float)
        frame_index = np.asarray(h5["frames/index"][:], dtype=int)

        across_rows: List[Dict[str, Any]] = []
        n_frames = across_direct.shape[1] if across_direct.ndim == 3 else 0
        for window in range(across_direct.shape[0]):
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    across_rows.append(
                        {
                            "window": window,
                            "start": float(starts[window]),
                            "end": float(ends[window]),
                            "frame_i": int(frame_index[i]),
                            "frame_j": int(frame_index[j]),
                            "direct": _cell(across_direct[window, i, j]),
                            "acf": _cell(across_acf[window, i, j]),
                        }
                    )
        written.append(
            write_table_csv(destination / "window_across_long.csv", across_rows)
        )

        within_rows: List[Dict[str, Any]] = []
        n_windows = within_acf.shape[1] if within_acf.ndim == 3 else 0
        for frame in range(within_acf.shape[0]):
            for i in range(n_windows):
                for j in range(i + 1, n_windows):
                    within_rows.append(
                        {
                            "original_frame": int(frame_index[frame]),
                            "window_i": i,
                            "window_j": j,
                            "acf": _cell(within_acf[frame, i, j]),
                        }
                    )
        written.append(
            write_table_csv(destination / "window_within_long.csv", within_rows)
        )

        if "tracks" in h5 and "tracks/summary/id" in h5:
            summary = h5["tracks/summary"]
            summary_names = (
                "id", "n_obs", "first_frame_row", "last_frame_row",
                "center_first", "center_last", "axis_first", "axis_last",
                "group", "mean_similarity",
            )
            track_rows = [
                {
                    name: _cell(summary[name][index])
                    for name in summary_names
                }
                for index in range(summary["id"].shape[0])
            ]
            written.append(
                write_table_csv(destination / "tracks_summary.csv", track_rows)
            )

            obs_track = np.asarray(h5["tracks/obs/track"][:], int)
            obs_peak = np.asarray(h5["tracks/obs/peak_id"][:], int)
            observation_rows = [
                {
                    "track": int(track),
                    "peak_id": int(peak),
                    "original_frame": _cell(
                        columns["original_frame"][peak]
                    ),
                    "frame_row": _cell(columns["frame_row"][peak]),
                    "center": _cell(columns["center"][peak]),
                    "pressure": _cell(columns["pressure"][peak]),
                }
                for track, peak in zip(obs_track, obs_peak)
            ]
            written.append(
                write_table_csv(
                    destination / "track_observations.csv", observation_rows
                )
            )

            if "tracks/intervals/order_pos" in h5:
                intervals = h5["tracks/intervals"]
                interval_names = (
                    "order_pos", "frame_row_from", "frame_row_to",
                    "axis_from", "axis_to", "group", "births", "deaths",
                    "n_active", "median_center_shift",
                    "window_direct_median", "transition_candidate",
                )
                interval_rows = [
                    {
                        name: _cell(intervals[name][index])
                        for name in interval_names
                        if name in intervals
                    }
                    for index in range(intervals["order_pos"].shape[0])
                ]
                written.append(
                    write_table_csv(
                        destination / "transition_intervals.csv",
                        interval_rows,
                    )
                )
    return written


def export_matrices(
    correlations_h5: "str | Path",
    out_dir: "str | Path",
) -> List[Path]:
    """Full K x K matrix dumps (quadratic in size — on demand only)."""

    import h5py  # type: ignore

    source = Path(correlations_h5).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    with h5py.File(str(source), "r") as h5:
        for name in ("roi_area", "location"):
            matrix = np.asarray(h5[f"anchor_maps/{name}"][:], dtype=float)
            rows = [
                {
                    "anchor": index,
                    **{
                        f"t{target:04d}": _cell(matrix[index, target])
                        for target in range(matrix.shape[1])
                    },
                }
                for index in range(matrix.shape[0])
            ]
            written.append(write_table_csv(destination / f"{name}.csv", rows))
    return written


__all__ = ["export_matrices", "export_summary_csvs"]
