"""ROI-gated all-peak track linking and exploratory transition screening.

The pipeline has exactly one track linker — :func:`seriesxrd.analysis.
unknowns.link_tracks` (Step 3c grew it: width-scaled position gate, gap
tolerance, drift prediction, scan grouping, greedy one-to-one). This module
reuses it over the correlations stage's all-peak table and supplies the
second gate issue #43's validated prototype used: the mutual ROI similarity
``sqrt(S(A→B)·S(B→A))`` taken from the directional ROI matrix this stage
already computes. Coupling stays correlations → analysis only.

Everything here is exploratory screening. A track is a linking hypothesis;
a flagged interval is a coincidence of changes worth looking at, never a
confirmed phase transition.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_MIN_ROI_SIMILARITY = 0.2
_MIN_INTERVALS_FOR_WINDOW_CLAUSE = 4
TRANSITION_RULE = (
    "births+deaths >= max(2, ceil(0.25*n_active)) OR window_direct_median < "
    "median(window_direct_median over intervals) - 2*MAD (window clause only "
    "with >= 4 finite intervals; fewer cannot support a robust baseline); "
    "exploratory candidates, not confirmed transitions"
)


def mutual_roi_similarity(roi_area: np.ndarray) -> np.ndarray:
    """``sqrt(S(A→B)·S(B→A))`` — symmetric, NaN-propagating.

    The powder directional matrix becomes the issue-#43 mutual score; the
    single-crystal matrix is already symmetric, for which this is the matrix
    itself, so one code path serves both sample types. Same-frame cells are
    NaN by construction and never gate a link (adjacent linkable frames are
    always distinct).
    """

    matrix = np.asarray(roi_area, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("roi_area must be a square matrix")
    return np.sqrt(np.clip(matrix, 0.0, None) * np.clip(matrix.T, 0.0, None))


def build_tracks(
    peaks,
    roi_area: np.ndarray,
    *,
    n_frames: int,
    order_key: str = "frame",
    order_values: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
    link_tol_fwhm: float = 1.5,
    max_gap: int = 2,
    min_track_frames: int = 3,
    min_roi_similarity: float = DEFAULT_MIN_ROI_SIMILARITY,
) -> Dict[str, Any]:
    """Link the all-peak table into ROI-gated tracks.

    ``min_roi_similarity`` is the evidence gate on the mutual score; ``0.0``
    disables it entirely (pure positional linking, the documented escape
    hatch). On the synthetic contracts genuinely related ROIs across
    adjacent frames score well above 0.5 while near-disjoint overlaps score
    below 0.1, so the 0.2 default rejects the latter without punishing
    ordinary compression drift.
    """

    from ..analysis.unknowns import link_tracks

    mutual = mutual_roi_similarity(roi_area)
    if mutual.shape[0] != peaks.size:
        raise ValueError("roi_area size does not match the peak table")
    tracks = link_tracks(
        peaks.frame_row,
        peaks.center,
        np.nan_to_num(np.asarray(peaks.area, dtype=float), nan=0.0),
        peaks.width,
        n_frames=int(n_frames),
        link_tol_fwhm=float(link_tol_fwhm),
        max_gap=int(max_gap),
        min_track_frames=int(min_track_frames),
        tracking_axis_values=(
            None if order_key == "frame" else np.asarray(order_values, float)
        ),
        tracking_axis=order_key,
        group_values=group_ids,
        similarity=lambda a, b: float(mutual[a, b]),
        min_similarity=float(min_roi_similarity),
    )

    obs_track: List[int] = []
    obs_peak: List[int] = []
    edge_keys = (
        "track", "peak_from", "peak_to", "similarity", "center_shift",
        "axis_gap",
    )
    edges: Dict[str, List[float]] = {key: [] for key in edge_keys}
    summary_keys = (
        "id", "n_obs", "first_frame_row", "last_frame_row", "center_first",
        "center_last", "axis_first", "axis_last", "group", "mean_similarity",
    )
    summary: Dict[str, List[float]] = {key: [] for key in summary_keys}
    for track_id, track in enumerate(tracks):
        rows = np.asarray(track["rows"], dtype=int)
        centers = np.asarray(track["centers"], dtype=float)
        axis = np.asarray(track["axis"], dtype=float)
        obs_track.extend([track_id] * rows.size)
        obs_peak.extend(int(row) for row in rows)
        similarities: List[float] = []
        for position in range(rows.size - 1):
            score = float(mutual[rows[position], rows[position + 1]])
            similarities.append(score)
            edges["track"].append(track_id)
            edges["peak_from"].append(int(rows[position]))
            edges["peak_to"].append(int(rows[position + 1]))
            edges["similarity"].append(score)
            edges["center_shift"].append(
                float(centers[position + 1] - centers[position])
            )
            edges["axis_gap"].append(float(axis[position + 1] - axis[position]))
        summary["id"].append(track_id)
        summary["n_obs"].append(int(rows.size))
        summary["first_frame_row"].append(int(track["frames"][0]))
        summary["last_frame_row"].append(int(track["frames"][-1]))
        summary["center_first"].append(float(centers[0]))
        summary["center_last"].append(float(centers[-1]))
        summary["axis_first"].append(float(axis[0]))
        summary["axis_last"].append(float(axis[-1]))
        summary["group"].append(int(track["group"][0]))
        summary["mean_similarity"].append(
            float(np.mean(similarities)) if similarities else math.nan
        )

    int_keys = {"track", "peak_from", "peak_to"}
    return {
        "tracks": tracks,
        "obs": {
            "track": np.asarray(obs_track, dtype="i4"),
            "peak_id": np.asarray(obs_peak, dtype="i4"),
        },
        "edges": {
            key: np.asarray(
                values, dtype="i4" if key in int_keys else float
            )
            for key, values in edges.items()
        },
        "summary": {
            key: np.asarray(
                values,
                dtype=float if key in (
                    "center_first", "center_last", "axis_first", "axis_last",
                    "mean_similarity",
                ) else "i4",
            )
            for key, values in summary.items()
        },
        "settings": {
            "link_tol_fwhm": float(link_tol_fwhm),
            "max_gap": int(max_gap),
            "min_track_frames": int(min_track_frames),
            "min_roi_similarity": float(min_roi_similarity),
            "order_by": str(order_key),
        },
    }


def transition_summary(
    bundle: Dict[str, Any],
    *,
    n_frames: int,
    order_key: str,
    order_values: np.ndarray,
    group_ids: Optional[np.ndarray],
    across_direct: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per adjacent ordered frame pair: births, deaths, shifts, window drops.

    Exploratory by design (issue #43): the flagged intervals are candidates
    for a closer look, decided by the recorded :data:`TRANSITION_RULE`, and
    never an automatic structural or phase assignment.
    """

    axis_values = np.asarray(order_values, dtype=float)
    groups = (
        np.zeros(int(n_frames), dtype=int)
        if group_ids is None
        else np.asarray(group_ids, dtype=int)
    )
    summary = bundle["summary"]
    first = np.asarray(summary["first_frame_row"], dtype=int)
    last = np.asarray(summary["last_frame_row"], dtype=int)
    track_group = np.asarray(summary["group"], dtype=int)

    # Per-track lookup: frame row -> center, for the median-shift measure.
    track_frames: Dict[int, Dict[int, float]] = {}
    for track_id, track in enumerate(bundle["tracks"]):
        frames = np.asarray(track["frames"], dtype=int)
        centers = np.asarray(track["centers"], dtype=float)
        track_frames[track_id] = {
            int(frame): float(center)
            for frame, center in zip(frames, centers)
        }

    keys = (
        "order_pos", "frame_row_from", "frame_row_to", "axis_from", "axis_to",
        "group", "births", "deaths", "n_active", "median_center_shift",
        "window_direct_median", "transition_candidate",
    )
    rows: Dict[str, List[float]] = {key: [] for key in keys}
    across = np.asarray(across_direct, dtype=float)
    for group_id in sorted(set(int(g) for g in groups)):
        members = np.nonzero(groups == group_id)[0]
        if order_key != "frame":
            members = members[np.isfinite(axis_values[members])]
        if members.size < 2:
            continue
        in_group = track_group == group_id
        for position in range(members.size - 1):
            row_from = int(members[position])
            row_to = int(members[position + 1])
            births = int(np.count_nonzero(in_group & (first == row_to)))
            deaths = int(np.count_nonzero(in_group & (last == row_from)))
            active = int(
                np.count_nonzero(
                    in_group & (first <= row_from) & (last >= row_to)
                )
            )
            shifts = [
                abs(mapping[row_to] - mapping[row_from])
                for mapping in track_frames.values()
                if row_from in mapping and row_to in mapping
            ]
            window_median = (
                float(np.nanmedian(across[:, row_from, row_to]))
                if across.size
                else math.nan
            )
            rows["order_pos"].append(position)
            rows["frame_row_from"].append(row_from)
            rows["frame_row_to"].append(row_to)
            rows["axis_from"].append(float(axis_values[row_from]))
            rows["axis_to"].append(float(axis_values[row_to]))
            rows["group"].append(int(group_id))
            rows["births"].append(births)
            rows["deaths"].append(deaths)
            rows["n_active"].append(active)
            rows["median_center_shift"].append(
                float(np.median(shifts)) if shifts else math.nan
            )
            rows["window_direct_median"].append(window_median)
            rows["transition_candidate"].append(False)  # decided below

    window_medians = np.asarray(rows["window_direct_median"], dtype=float)
    finite = np.isfinite(window_medians)
    if np.count_nonzero(finite) >= _MIN_INTERVALS_FOR_WINDOW_CLAUSE:
        center = float(np.median(window_medians[finite]))
        mad = 1.4826 * float(
            np.median(np.abs(window_medians[finite] - center))
        )
        window_drop = window_medians < (center - 2.0 * mad)
    else:
        window_drop = np.zeros(window_medians.size, dtype=bool)
    births = np.asarray(rows["births"], dtype=int)
    deaths = np.asarray(rows["deaths"], dtype=int)
    active = np.asarray(rows["n_active"], dtype=int)
    threshold = np.maximum(2, np.ceil(0.25 * active).astype(int))
    candidate = ((births + deaths) >= threshold) | window_drop

    int_keys = {
        "order_pos", "frame_row_from", "frame_row_to", "group", "births",
        "deaths", "n_active",
    }
    result = {
        key: np.asarray(
            values, dtype="i4" if key in int_keys else float
        )
        for key, values in rows.items()
        if key != "transition_candidate"
    }
    result["transition_candidate"] = candidate.astype(bool)
    return result


__all__ = [
    "DEFAULT_MIN_ROI_SIMILARITY",
    "TRANSITION_RULE",
    "build_tracks",
    "mutual_roi_similarity",
    "transition_summary",
]
