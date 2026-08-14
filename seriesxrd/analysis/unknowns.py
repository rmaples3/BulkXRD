"""Step 3c — unknown-phase detection by co-occurrence clustering.

After Step 3a subtracts the identified phases, ``/residual/peaks`` holds the
re-fitted peaks nothing known explains. Real undiscovered phases do not appear
as isolated blips: their reflections form COHERENT TRACKS across the pressure
series (drifting smoothly as the lattice compresses) and those tracks appear,
disappear and move TOGETHER. This module turns that physics into the final
pipeline stage:

  1. **Track linking** — residual peaks are chained into tracks in frame order
     or in a chosen physical-axis order (pressure, temperature, or elapsed
     time). Greedy one-to-one nearest-center linking uses a width-scaled
     tolerance and, for physical axes, predicts the next center from the local
     track slope so reflections may drift smoothly under compression/heating.
     Tracks seen in too few frames are noise and dropped.
  2. **Co-occurrence clustering** — tracks whose presence/absence patterns
     agree (Jaccard similarity of the frame sets) are merged single-link into
     clusters: "these reflections belong to the same unknown substance".
  3. **Fingerprinting** — each cluster reports its d-spacings at a reference
     frame (the frame where most of its tracks coexist), the set to search
     against COD/ICSD/MP or to flag as genuinely new. Appearance/disappearance
     frames of a cluster are phase-transition candidates.

Appends ``/unknowns`` to the analysis HDF5. Pure numpy + h5py; requires
``/residual`` (Step 3a) to exist. Consumed by the heatmap (per-cluster layers)
and by whoever hunts the unknowns.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .identify import radial_to_d
from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance

SCHEMA_VERSION = "1"
TRACKING_AXES = ("frame", "pressure", "temperature", "time")
TRACKING_GROUPS = ("none", "scan", "folder")
_SCAN_RE = re.compile(r"(?:^|[^A-Za-z])scan[_\- ]*0*(\d+)", re.IGNORECASE)


def _decode_text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def _elapsed_seconds(raw: np.ndarray) -> np.ndarray:
    secs = np.full(raw.shape[0], np.nan, dtype=float)
    for i, value in enumerate(raw):
        txt = _decode_text(value).strip()
        if not txt:
            continue
        try:
            secs[i] = datetime.fromisoformat(txt).timestamp()
        except ValueError:
            continue
    if np.any(np.isfinite(secs)):
        secs -= np.nanmin(secs)
    return secs


def _tracking_values(h5, axis: str, n_frames: int) -> "Tuple[str, np.ndarray, str]":
    """Return per-frame values for the requested tracking axis on an open HDF5."""
    key = (axis or "frame").strip().lower()
    if key in ("index", "", "same"):   # "same" = mirror seed order; resolved upstream
        key = "frame"
    if key not in TRACKING_AXES:
        raise ValueError(f"Unknown unknown-tracking axis {axis!r} "
                         f"(choose from {', '.join(TRACKING_AXES)}).")
    if key == "frame":
        return key, np.arange(int(n_frames), dtype=float), "Frame index"
    fr = h5.get("frames")
    if fr is None:
        raise ValueError(f"No /frames group — cannot track unknowns by {key}.")
    if key == "pressure":
        if "pressure" not in fr:
            raise ValueError("No /frames/pressure — import/extract pressures before "
                             "using pressure-aware unknown tracking.")
        vals = np.asarray(fr["pressure"][:], dtype=float)
        label = "Pressure (GPa)"
    elif key == "temperature":
        if "temperature" not in fr:
            raise ValueError("No /frames/temperature — import a temperature_K column "
                             "before using temperature-aware unknown tracking.")
        vals = np.asarray(fr["temperature"][:], dtype=float)
        label = "Temperature (K)"
    else:
        if "timestamp" not in fr:
            raise ValueError("No /frames/timestamp — cannot track unknowns by time.")
        vals = _elapsed_seconds(fr["timestamp"][:])
        label = "Elapsed time (s)"
    if vals.size != int(n_frames):
        raise ValueError(f"/frames/{key} length ({vals.size}) does not match the "
                         f"residual frame count ({n_frames}).")
    if not np.any(np.isfinite(vals)):
        raise ValueError(f"No finite {label} values — cannot use {key}-aware "
                         "unknown tracking.")
    return key, vals, label


def _frame_names(h5, n_frames: int) -> List[str]:
    fr = h5.get("frames")
    if fr is None or "filename" not in fr:
        return [""] * int(n_frames)
    raw = fr["filename"][:]
    names = [_decode_text(v) for v in raw]
    if len(names) < int(n_frames):
        names += [""] * (int(n_frames) - len(names))
    return names[:int(n_frames)]


def _scan_label(name: str) -> str:
    m = _SCAN_RE.search(str(name))
    if m:
        return f"scan{int(m.group(1)):03d}"
    return "scan_unknown"


def _folder_label(name: str) -> str:
    s = str(name).replace("\\", "/")
    parent = s.rsplit("/", 1)[0] if "/" in s else ""
    return parent.rsplit("/", 1)[-1] if parent else "folder_unknown"


def _tracking_groups(h5, group_by: str, n_frames: int) -> "Tuple[str, np.ndarray, List[str]]":
    """Return per-frame integer group ids plus their display labels."""
    key = (group_by or "none").strip().lower()
    if key in ("", "all", "same"):   # "same" = mirror seed grouping; resolved upstream
        key = "none"
    if key not in TRACKING_GROUPS:
        raise ValueError(f"Unknown unknown-tracking group {group_by!r} "
                         f"(choose from {', '.join(TRACKING_GROUPS)}).")
    if key == "none":
        return key, np.zeros(int(n_frames), dtype=int), ["all"]
    names = _frame_names(h5, n_frames)
    labels = [(_scan_label(nm) if key == "scan" else _folder_label(nm)) for nm in names]
    label_to_id: Dict[str, int] = {}
    ids = np.zeros(int(n_frames), dtype=int)
    for i, label in enumerate(labels):
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        ids[i] = label_to_id[label]
    ordered_labels = [label for label, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
    return key, ids, ordered_labels


def _predict_center(track: Dict[str, list], axis_now: float,
                    *, use_axis_predictor: bool) -> float:
    """Predict the next center from the recent local slope along the axis."""
    if not use_axis_predictor or len(track["centers"]) < 2:
        return float(track["centers"][-1])
    x = np.asarray(track["axis"][-4:], dtype=float)
    y = np.asarray(track["centers"][-4:], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.nanmax(x) - np.nanmin(x)) <= 0:
        return float(track["centers"][-1])
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    denom = float(np.sum((x - xm) ** 2))
    if denom <= 0:
        return float(track["centers"][-1])
    slope = float(np.sum((x - xm) * (y - ym)) / denom)
    return float(track["centers"][-1]) + slope * (float(axis_now) - float(track["axis"][-1]))


# ---------------------------------------------------------------------------
# Track linking
# ---------------------------------------------------------------------------

def link_tracks(frames: np.ndarray, centers: np.ndarray, amplitudes: np.ndarray,
                fwhms: np.ndarray, *, n_frames: int,
                link_tol_fwhm: float = 1.5, max_gap: int = 2,
                min_track_frames: int = 3,
                tracking_axis_values: "Optional[np.ndarray]" = None,
                tracking_axis: str = "frame",
                max_axis_gap: "Optional[float]" = None,
                axis_predictor: bool = True,
                group_values: "Optional[np.ndarray]" = None,
                similarity: "Optional[Any]" = None,
                min_similarity: float = 0.0) -> "List[Dict[str, np.ndarray]]":
    """Chain per-frame residual peaks into cross-frame/physical-axis tracks.

    Greedy one-to-one linking, closest pair first: a peak joins an open track
    when it is within ``link_tol_fwhm × max(track width, peak width)`` of the
    predicted center. With ``tracking_axis='frame'`` the prediction is the last
    center, preserving the historical behavior. With pressure/temperature/time,
    the next center is predicted from the local center-vs-axis slope once a
    track has at least two points. ``max_gap`` counts missing ordered samples;
    ``max_axis_gap`` optionally caps physical-axis distance between linked
    observations. ``group_values`` separates independent scans/runs: tracks are
    closed at group boundaries and never link across them. Tracks observed in
    fewer than ``min_track_frames`` frames are discarded as noise. Returns one
    dict per kept track:
    ``{frames, centers, amplitudes, fwhms, axis, group, rows}`` where ``rows``
    are the source observation indices of the linked peaks.

    ``similarity`` optionally adds a second, evidence-based gate on top of the
    position gate: a callable ``(row_last, row_candidate) -> float`` scored
    between a track's most recent observation row and a candidate row (the
    correlations stage passes its mutual ROI similarity). A candidate is
    rejected unless the score is finite and ``>= min_similarity`` — NaN means
    no usable evidence and rejects while the gate is active. Geometry stays
    the primary prior: candidates are still consumed closest first, with the
    similarity only breaking distance ties (a blended score would silently
    re-rank well-separated candidates and make the gate impossible to reason
    about). With ``similarity=None`` (the default) linking is bit-identical
    to the historical position-only behavior.
    """
    frames = np.asarray(frames, int)
    centers = np.asarray(centers, float)
    amplitudes = np.asarray(amplitudes, float)
    fwhms = np.asarray(fwhms, float)
    axis_key = (tracking_axis or "frame").strip().lower()
    if axis_key in ("index", ""):
        axis_key = "frame"
    if axis_key not in TRACKING_AXES:
        raise ValueError(f"tracking_axis must be one of {TRACKING_AXES}, got {tracking_axis!r}")
    if tracking_axis_values is None:
        axis_values = np.arange(int(n_frames), dtype=float)
        axis_key = "frame"
    else:
        axis_values = np.asarray(tracking_axis_values, dtype=float)
        if axis_values.size != int(n_frames):
            raise ValueError("tracking_axis_values length must match n_frames")
    if group_values is None:
        groups = np.zeros(int(n_frames), dtype=int)
    else:
        groups = np.asarray(group_values)
        if groups.size != int(n_frames):
            raise ValueError("group_values length must match n_frames")

    group_order: List[Any] = []
    for g in groups:
        if g not in group_order:
            group_order.append(g)

    all_done: List[Dict[str, list]] = []
    for group_id in group_order:
        in_group = np.asarray(groups == group_id, dtype=bool)
        if axis_key == "frame":
            frame_order = np.nonzero(in_group)[0].astype(int)
            use_predictor = False
        else:
            finite_frames = np.where(in_group & np.isfinite(axis_values))[0]
            if not finite_frames.size:
                continue
            frame_order = np.asarray(
                sorted((int(f) for f in finite_frames),
                       key=lambda f: (float(axis_values[f]), f)),
                dtype=int,
            )
            use_predictor = bool(axis_predictor)

        open_tracks: List[Dict[str, list]] = []
        done: List[Dict[str, list]] = []
        for order_pos, fi in enumerate(frame_order):
            axis_now = float(axis_values[fi])
            rows = np.nonzero(frames == fi)[0]
            # Retire tracks that fell too far behind. max_gap counts missing
            # ordered samples inside this scan/group.
            still: List[Dict[str, list]] = []
            for t in open_tracks:
                order_gap = int(order_pos) - int(t["order"][-1])
                axis_gap = abs(axis_now - float(t["axis"][-1]))
                too_far_axis = (
                    max_axis_gap is not None
                    and axis_key != "frame"
                    and axis_gap > float(max_axis_gap)
                )
                if order_gap > int(max_gap) + 1 or too_far_axis:
                    done.append(t)
                else:
                    still.append(t)
            open_tracks = still
            if rows.size == 0:
                continue
            # Candidate (gap, [similarity,] track, row) tuples within
            # tolerance, closest first. Without the similarity gate the tuple
            # shape and sort are the historical ones, so linking stays
            # bit-identical for Step 3c.
            use_gate = similarity is not None
            cands: List[Tuple] = []
            for ti, t in enumerate(open_tracks):
                c_last = _predict_center(t, axis_now, use_axis_predictor=use_predictor)
                w_last = t["fwhms"][-1]
                for r in rows:
                    tol = float(link_tol_fwhm) * max(w_last, fwhms[r], 1e-9)
                    g = abs(centers[r] - c_last)
                    if g > tol:
                        continue
                    if use_gate:
                        score = float(similarity(int(t["rows"][-1]), int(r)))
                        if not np.isfinite(score) or score < float(min_similarity):
                            continue
                        cands.append((g, -score, ti, int(r)))
                    else:
                        cands.append((g, ti, int(r)))
            if use_gate:
                cands.sort()
            else:
                cands.sort(key=lambda x: x[0])
            used_t: set = set()
            used_r: set = set()
            for cand in cands:
                ti, r = cand[-2], cand[-1]
                if ti in used_t or r in used_r:
                    continue
                used_t.add(ti)
                used_r.add(r)
                t = open_tracks[ti]
                t["frames"].append(fi)
                t["centers"].append(float(centers[r]))
                t["amplitudes"].append(float(amplitudes[r]))
                t["fwhms"].append(float(fwhms[r]))
                t["axis"].append(axis_now)
                t["group"].append(int(group_id))
                t["order"].append(int(order_pos))
                t["rows"].append(int(r))
            for r in rows:
                if int(r) not in used_r:             # unmatched peak → new track
                    open_tracks.append({"frames": [fi],
                                        "centers": [float(centers[r])],
                                        "amplitudes": [float(amplitudes[r])],
                                        "fwhms": [float(fwhms[r])],
                                        "axis": [axis_now],
                                        "group": [int(group_id)],
                                        "order": [int(order_pos)],
                                        "rows": [int(r)]})
        done.extend(open_tracks)
        all_done.extend(done)

    kept = [{k: np.asarray(v) for k, v in t.items() if k != "order"}
            for t in all_done if len(t["frames"]) >= int(min_track_frames)]
    kept.sort(key=lambda t: -float(np.sum(t["amplitudes"])))
    return kept


# ---------------------------------------------------------------------------
# Co-occurrence clustering
# ---------------------------------------------------------------------------

def cluster_tracks(tracks: "List[Dict[str, np.ndarray]]", *,
                   jaccard_threshold: float = 0.6) -> np.ndarray:
    """Cluster ids (0-based) by single-link merging of tracks whose frame sets
    have Jaccard similarity ≥ ``jaccard_threshold`` — reflections of one phase
    appear and vanish together across the series."""
    n = len(tracks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [set(int(f) for f in t["frames"]) for t in tracks]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union and inter / union >= float(jaccard_threshold):
                parent[find(i)] = find(j)
    roots: Dict[int, int] = {}
    out = np.zeros(n, int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def run_unknowns(
    analysis_h5: "str | Path",
    *,
    link_tol_fwhm: float = 1.5,
    max_gap: int = 2,
    min_track_frames: int = 3,
    jaccard_threshold: float = 0.6,
    tracking_axis: str = "frame",
    group_by: str = "none",
    max_axis_gap: "Optional[float]" = None,
    axis_predictor: bool = True,
    out_h5: "Optional[str | Path]" = None,
) -> Dict[str, Any]:
    """Cluster the Step-3a residual peaks into candidate unknown phases and
    append ``/unknowns`` to the analysis HDF5.

        /unknowns  attrs: schema_version, link_tol_fwhm, max_gap,
                          min_track_frames, jaccard_threshold, tracking_axis,
                          max_axis_gap, axis_predictor, group_by,
                          n_tracks, n_clusters
        /unknowns/obs/{track,frame,center,amplitude,fwhm,axis,group} (M,) every
                          kept observation; ``axis`` is the tracking-axis value
        /unknowns/tracks/{id,cluster,n_frames,first_frame,last_frame,
                          center_first,center_last,axis_first,axis_last,group} (T,)
        /unknowns/groups/{id,label}                              (G,)
        /unknowns/clusters/{id,n_tracks,first_frame,last_frame,ref_frame} (C,)
        /unknowns/fingerprint/{cluster,d}                     (F,)  per-cluster
                          d-spacings at its reference frame — the set to search
                          against external databases

    The cluster boundaries (first/last frame) are phase-transition candidates.
    Returns a manifest with a per-cluster summary.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    with h5py.File(str(src), "r") as h5:
        rg = h5.get("residual")
        if rg is None or "peaks" not in rg:
            raise ValueError("No /residual/peaks — run Step 3a (+ residual) first.")
        gp = rg["peaks"]
        frames = np.asarray(gp["frame"][:], int)
        centers = np.asarray(gp["center"][:], float)
        amps = np.asarray(gp["amplitude"][:], float)
        fwhms = np.asarray(gp["fwhm"][:], float)
        n_frames = int(np.asarray(gp["counts"][:]).size)
        unit = str(h5.attrs.get("unit", ""))
        wavelength = float(h5.attrs.get("wavelength", 0.0) or 0.0) or None
        axis_key, axis_values, axis_label = _tracking_values(h5, tracking_axis, n_frames)
        group_key, group_ids, group_labels = _tracking_groups(h5, group_by, n_frames)

    tracks = link_tracks(frames, centers, amps, fwhms, n_frames=n_frames,
                         link_tol_fwhm=link_tol_fwhm, max_gap=max_gap,
                         min_track_frames=min_track_frames,
                         tracking_axis_values=axis_values,
                         tracking_axis=axis_key,
                         max_axis_gap=max_axis_gap,
                         axis_predictor=axis_predictor,
                         group_values=group_ids)
    cluster_of = (cluster_tracks(tracks, jaccard_threshold=jaccard_threshold)
                  if tracks else np.zeros(0, int))
    n_clusters = int(cluster_of.max()) + 1 if tracks else 0

    # Per-cluster reference frame (most member tracks present) + fingerprint.
    fp_cluster: List[int] = []
    fp_d: List[float] = []
    summaries: List[Dict[str, Any]] = []
    for c in range(n_clusters):
        members = [t for t, cl in zip(tracks, cluster_of) if cl == c]
        count = np.zeros(n_frames, int)
        for t in members:
            count[np.asarray(t["frames"], int)] += 1
        ref = int(np.argmax(count))
        ds: List[float] = []
        for t in members:
            hit = np.nonzero(np.asarray(t["frames"], int) == ref)[0]
            if hit.size:
                ds.append(float(t["centers"][hit[0]]))
        d_vals = sorted(radial_to_d(np.asarray(ds, float), unit, wavelength).tolist(),
                        reverse=True) if ds and unit else sorted(ds, reverse=True)
        fp_cluster += [c] * len(d_vals)
        fp_d += d_vals
        lo = min(int(t["frames"].min()) for t in members)
        hi = max(int(t["frames"].max()) for t in members)
        axis_all = np.concatenate([np.asarray(t["axis"], dtype=float) for t in members])
        finite_axis = axis_all[np.isfinite(axis_all)]
        group_member_ids = sorted({int(np.asarray(t["group"], dtype=int)[0])
                                   for t in members})
        group_id = group_member_ids[0] if len(group_member_ids) == 1 else -1
        summaries.append({"cluster": c, "n_tracks": len(members),
                          "first_frame": lo, "last_frame": hi,
                          "ref_frame": ref,
                          "group_id": group_id,
                          "group_label": (group_labels[group_id]
                                          if 0 <= group_id < len(group_labels)
                                          else "mixed"),
                          "axis_min": (float(np.min(finite_axis))
                                       if finite_axis.size else np.nan),
                          "axis_max": (float(np.max(finite_axis))
                                       if finite_axis.size else np.nan),
                          "d_fingerprint": [round(v, 4) for v in d_vals]})

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "unknowns" in o:
                del o["unknowns"]
            g = o.create_group("unknowns")
            write_step_provenance(o, "unknowns",
                                  tool="seriesxrd.analysis.unknowns",
                                  schema_version=SCHEMA_VERSION)
            g.attrs.update({"schema_version": SCHEMA_VERSION,
                            "seriesxrd_version": VERSION,
                            "link_tol_fwhm": float(link_tol_fwhm),
                            "max_gap": int(max_gap),
                            "min_track_frames": int(min_track_frames),
                            "jaccard_threshold": float(jaccard_threshold),
                            "tracking_axis": axis_key,
                            "tracking_axis_label": axis_label,
                            "group_by": group_key,
                            "max_axis_gap": (float(max_axis_gap)
                                             if max_axis_gap is not None else np.nan),
                            "axis_predictor": bool(axis_predictor),
                            "n_tracks": len(tracks),
                            "n_clusters": n_clusters})
            go = g.create_group("obs")
            flat = {"track": [], "frame": [], "center": [],
                    "amplitude": [], "fwhm": [], "axis": [], "group": []}
            for ti, t in enumerate(tracks):
                for j in range(t["frames"].size):
                    flat["track"].append(ti)
                    flat["frame"].append(int(t["frames"][j]))
                    flat["center"].append(float(t["centers"][j]))
                    flat["amplitude"].append(float(t["amplitudes"][j]))
                    flat["fwhm"].append(float(t["fwhms"][j]))
                    flat["axis"].append(float(t["axis"][j]))
                    flat["group"].append(int(t["group"][j]))
            for k, v in flat.items():
                go.create_dataset(k, data=np.asarray(
                    v, dtype=("i4" if k in ("track", "frame", "group") else "f8")))
            gg = g.create_group("groups")
            gg.create_dataset("id", data=np.arange(len(group_labels), dtype="i4"))
            gg.create_dataset("label", data=np.asarray(group_labels, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            gt = g.create_group("tracks")
            gt.create_dataset("id", data=np.arange(len(tracks), dtype="i4"))
            gt.create_dataset("cluster", data=cluster_of.astype("i4"))
            gt.create_dataset("group",
                              data=np.array([int(t["group"][0]) for t in tracks], "i4"))
            gt.create_dataset("n_frames",
                              data=np.array([t["frames"].size for t in tracks], "i4"))
            gt.create_dataset("first_frame",
                              data=np.array([t["frames"].min() if t["frames"].size else -1
                                             for t in tracks], "i4"))
            gt.create_dataset("last_frame",
                              data=np.array([t["frames"].max() if t["frames"].size else -1
                                             for t in tracks], "i4"))
            gt.create_dataset("center_first",
                              data=np.array([t["centers"][0] for t in tracks], "f8"))
            gt.create_dataset("center_last",
                              data=np.array([t["centers"][-1] for t in tracks], "f8"))
            gt.create_dataset("axis_first",
                              data=np.array([t["axis"][0] for t in tracks], "f8"))
            gt.create_dataset("axis_last",
                              data=np.array([t["axis"][-1] for t in tracks], "f8"))
            gc = g.create_group("clusters")
            gc.create_dataset("id", data=np.arange(n_clusters, dtype="i4"))
            gc.create_dataset("n_tracks",
                              data=np.array([s["n_tracks"] for s in summaries], "i4"))
            gc.create_dataset("first_frame",
                              data=np.array([s["first_frame"] for s in summaries], "i4"))
            gc.create_dataset("last_frame",
                              data=np.array([s["last_frame"] for s in summaries], "i4"))
            gc.create_dataset("ref_frame",
                              data=np.array([s["ref_frame"] for s in summaries], "i4"))
            gc.create_dataset("group",
                              data=np.array([s["group_id"] for s in summaries], "i4"))
            gc.create_dataset("axis_min",
                              data=np.array([s["axis_min"] for s in summaries], "f8"))
            gc.create_dataset("axis_max",
                              data=np.array([s["axis_max"] for s in summaries], "f8"))
            gf = g.create_group("fingerprint")
            gf.create_dataset("cluster", data=np.asarray(fp_cluster, "i4"))
            gf.create_dataset("d", data=np.asarray(fp_d, "f8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    manifest = {**manifest_provenance("seriesxrd.analysis.unknowns", SCHEMA_VERSION),
                "out_h5": str(dst),
                "n_residual_peaks": int(frames.size),
                "n_tracks": len(tracks), "n_clusters": n_clusters,
                "tracking_axis": axis_key, "tracking_axis_label": axis_label,
                "group_by": group_key, "n_groups": len(group_labels),
                "group_labels": list(group_labels),
                "link_tol_fwhm": float(link_tol_fwhm), "max_gap": int(max_gap),
                "min_track_frames": int(min_track_frames),
                "jaccard_threshold": float(jaccard_threshold),
                "max_axis_gap": float(max_axis_gap) if max_axis_gap is not None else None,
                "axis_predictor": bool(axis_predictor),
                "clusters": summaries}
    print(f"[UNKNOWNS] {frames.size} residual peaks -> {len(tracks)} coherent "
          f"track(s) -> {n_clusters} cluster(s) "
          f"(tracking_axis={axis_key}, group_by={group_key}) -> {dst}", flush=True)
    for s in summaries:
        print(f"[UNKNOWNS]   cluster {s['cluster']}: {s['n_tracks']} tracks, "
              f"{s.get('group_label', 'all')} frames {s['first_frame']}-{s['last_frame']}, "
              f"d = {s['d_fingerprint']}", flush=True)
    return manifest
