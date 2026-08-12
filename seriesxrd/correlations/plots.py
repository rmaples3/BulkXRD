"""Agg-only plot exporters for correlation artifacts.

No pyplot or Tk backend is imported.  Figures are built with
``FigureCanvasAgg`` so the same code works in CI, over SSH, and from a worker
subprocess with no display server.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np


def _mpl():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    return Figure, FigureCanvasAgg, Normalize


def _atomic_save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        fig.savefig(str(tmp), format="png", dpi=140, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
        fig.clear()


def _normalize_trace(raw: np.ndarray) -> np.ndarray:
    """Per-frame waterfall normalization shared with analysis.stackplot.

    The scale is the max of a median-filtered copy floored by the MAD noise,
    so a 1-2 bin zinger cannot flatten the whole trace the way a plain
    percentile scale could. NaN bins stay NaN.
    """

    from ..analysis.stackplot import mad_noise, robust_amp

    values = np.asarray(raw, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full_like(values, np.nan)
    scale = max(robust_amp(values), 10.0 * mad_noise(values),
                np.finfo(float).eps)
    return np.where(finite, np.clip(values, 0.0, None) / scale, np.nan)


def _safe_pressure(value: float) -> str:
    if not np.isfinite(value):
        return "pressure_unknown"
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"pressure_{text}_GPa"


def _target_grid(
    row: np.ndarray,
    frame_row: np.ndarray,
    local_peak: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    slots = max(int(np.max(local_peak)) + 1 if local_peak.size else 1, 1)
    grid = np.full((n_frames, slots), np.nan, dtype=float)
    grid[frame_row.astype(int), local_peak.astype(int)] = np.asarray(row, float)
    return grid


def _strict_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return a plotting copy with the diagonal and upper half hidden.

    Window-correlation matrices are symmetric, so rendering both halves and
    the identity diagonal repeats information.  This helper is deliberately
    confined to the plot layer: the complete numerical matrix remains in the
    correlation HDF5 artifact.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("triangular correlation plots need a square 2D matrix")
    shown = values.copy()
    shown[np.triu_indices(shown.shape[0], k=0)] = np.nan
    return shown


def _heatmap(
    matrix: np.ndarray,
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    Figure, FigureCanvasAgg, _ = _mpl()
    values = np.asarray(matrix, dtype=float)
    width = max(5.2, min(12.0, 0.28 * max(values.shape[-1], 1) + 3.0))
    height = max(3.8, min(10.0, 0.25 * max(values.shape[0], 1) + 2.5))
    fig = Figure(figsize=(width, height), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.ax.tick_params(labelsize=8)
    _atomic_save(fig, path)


def _waterfall(
    path: Path,
    *,
    radial: np.ndarray,
    original_positive: np.ndarray,
    frame_indices: np.ndarray,
    frame_pressure: np.ndarray,
    peak_frame: np.ndarray,
    centers: np.ndarray,
    half_width: np.ndarray,
    score: np.ndarray,
    anchor: int,
    unit: str,
) -> None:
    Figure, FigureCanvasAgg, Normalize = _mpl()
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    n_frames = original_positive.shape[0]
    # Cap the figure height: 0.55 in/frame is readable for tens of frames but
    # a few hundred frames would render a 100+ inch, 15000+ px PNG.
    height = min(max(4.0, 0.55 * n_frames + 2.2), 18.0)
    label_stride = 1
    if 0.55 * n_frames + 2.2 > 18.0:
        label_stride = max(1, -(-n_frames // 40))
    fig = Figure(figsize=(9.0, height), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    offset_step = 1.15

    for frame in range(n_frames):
        trace = _normalize_trace(original_positive[frame])
        offset = frame * offset_step
        ax.plot(radial, trace + offset, color="#4c566a", linewidth=0.65, zorder=2)
        targets = np.nonzero(peak_frame == frame)[0]
        for target in targets:
            value = float(score[target])
            if not np.isfinite(value):
                continue
            support = (
                (radial >= centers[target] - half_width[target])
                & (radial <= centers[target] + half_width[target])
                & np.isfinite(trace)
            )
            if np.count_nonzero(support) < 2:
                continue
            ax.fill_between(
                radial[support],
                offset,
                trace[support] + offset,
                color=cmap(norm(value)),
                alpha=0.78,
                linewidth=0.0,
                zorder=3,
            )
        if frame % label_stride and frame != n_frames - 1:
            continue
        pressure = frame_pressure[frame]
        label = (
            f"frame {int(frame_indices[frame])}, {pressure:g} GPa"
            if np.isfinite(pressure)
            else f"frame {int(frame_indices[frame])}"
        )
        ax.text(
            radial[-1],
            offset + 0.05,
            label,
            ha="right",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xlabel(unit or "radial")
    ax.set_ylabel("original-positive intensity (normalized, offset)")
    ax.set_yticks([])
    ax.set_title(
        f"Anchor {anchor}: original-positive waterfall; color = Log-squared ROI correlation",
        fontsize=10,
    )
    ax.set_xlim(float(radial[0]), float(radial[-1]))
    ax.set_ylim(-0.05, n_frames * offset_step + 0.2)
    ax.grid(axis="x", color="#d8dee9", linewidth=0.5)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("Log-squared ROI correlation", fontsize=8)
    _atomic_save(fig, path)


def _track_map(
    path: Path,
    *,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    obs_track: np.ndarray,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    edge_similarity: np.ndarray,
    band_from: np.ndarray,
    band_to: np.ndarray,
    x_label: str,
    y_label: str,
    group_label: str,
) -> None:
    """Peak-position-vs-condition track plot, edges colored by similarity.

    Exploratory transition-candidate intervals are drawn as translucent
    vertical bands. Same viridis 0..1 convention as the waterfall shading.
    """

    Figure, FigureCanvasAgg, Normalize = _mpl()
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    fig = Figure(figsize=(9.5, 6.0), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)

    for lo, hi in zip(band_from, band_to):
        ax.axvspan(float(lo), float(hi), color="#bf616a", alpha=0.14, zorder=1)
    for (x0, x1), (y0, y1), score in zip(
        edge_x.reshape(-1, 2), edge_y.reshape(-1, 2), edge_similarity
    ):
        color = cmap(norm(score)) if np.isfinite(score) else "#c8c8c8"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.4, zorder=2)
    ax.scatter(obs_x, obs_y, s=9, color="#4c566a", zorder=3)
    for track_id in np.unique(obs_track):
        member = obs_track == track_id
        xs = obs_x[member]
        ys = obs_y[member]
        ax.scatter([xs[0]], [ys[0]], marker="^", s=34, color="#a3be8c",
                   zorder=4, linewidths=0)
        ax.scatter([xs[-1]], [ys[-1]], marker="v", s=34, color="#bf616a",
                   zorder=4, linewidths=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"ROI-gated peak tracks ({group_label}); ▲ birth, ▼ death, "
        "bands = exploratory transition candidates",
        fontsize=10,
    )
    ax.grid(color="#d8dee9", linewidth=0.5)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("mutual Log-squared ROI similarity", fontsize=8)
    _atomic_save(fig, path)


def _render_into(
    correlations_h5: Path,
    base: Path,
    max_anchor_plots: "int | None" = None,
) -> List[Path]:
    """Render one complete sample tree into an empty staging directory."""

    import h5py  # type: ignore

    source = Path(correlations_h5).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as h5:
        unit = str(h5.attrs.get("unit", "radial"))
        radial = np.asarray(h5["patterns/radial"][:], float)
        original = np.asarray(h5["patterns/original_positive"][:], float)
        frame_indices = np.asarray(h5["frames/index"][:], int)
        frame_pressure = np.asarray(h5["frames/pressure"][:], float)
        peak_frame = np.asarray(h5["peaks/frame_row"][:], int)
        local_peak = np.asarray(h5["peaks/local_peak"][:], int)
        peak_pressure = np.asarray(h5["peaks/pressure"][:], float)
        centers = np.asarray(h5["peaks/center"][:], float)
        half_width = np.asarray(h5["peaks/half_width"][:], float)
        roi = np.asarray(h5["anchor_maps/roi_area"][:], float)
        location = np.asarray(h5["anchor_maps/location"][:], float)
        # Artifacts written before /peaks/valid existed plot every anchor.
        if "peaks/valid" in h5:
            anchor_valid = np.asarray(h5["peaks/valid"][:], bool)
        else:
            anchor_valid = np.ones(peak_frame.size, dtype=bool)
        starts = np.asarray(h5["windows/start"][:], float)
        ends = np.asarray(h5["windows/end"][:], float)
        across_direct = np.asarray(h5["windows/across_direct"][:], float)
        across_acf = np.asarray(h5["windows/across_acf"][:], float)
        within_acf = np.asarray(h5["windows/within_acf"][:], float)
        # Track data is optional: --no-tracks runs and older artifacts.
        tracks = None
        if "tracks" in h5 and "tracks/summary/id" in h5:
            order_by = str(h5.attrs.get("order_by", "frame"))
            tracks = {
                "order_by": order_by,
                "order_label": str(h5.attrs.get("order_label", "Frame index")),
                "obs_track": np.asarray(h5["tracks/obs/track"][:], int),
                "obs_peak": np.asarray(h5["tracks/obs/peak_id"][:], int),
                "edge_track": np.asarray(h5["tracks/edges/track"][:], int),
                "edge_from": np.asarray(h5["tracks/edges/peak_from"][:], int),
                "edge_to": np.asarray(h5["tracks/edges/peak_to"][:], int),
                "edge_similarity": np.asarray(
                    h5["tracks/edges/similarity"][:], float
                ),
                "summary_group": np.asarray(h5["tracks/summary/group"][:], int),
            }
            for name in (
                "axis_from", "axis_to", "group", "transition_candidate",
            ):
                key = f"tracks/intervals/{name}"
                tracks[f"interval_{name}"] = (
                    np.asarray(h5[key][:]) if key in h5 else np.empty(0)
                )
            if "tracks/group_label" in h5:
                tracks["group_labels"] = [
                    value.decode("utf-8", "replace")
                    if isinstance(value, bytes)
                    else str(value)
                    for value in h5["tracks/group_label"][:]
                ]
            else:
                tracks["group_labels"] = []
            tracks["order_value"] = (
                np.asarray(h5["frames/order_value"][:], float)
                if "frames/order_value" in h5
                else np.arange(original.shape[0], dtype=float)
            )

    plot_anchor = anchor_valid.copy()
    if max_anchor_plots is not None:
        # Deterministic selection: the first N valid anchors in id order.
        selected = np.nonzero(anchor_valid)[0][: max(int(max_anchor_plots), 0)]
        plot_anchor = np.zeros_like(anchor_valid)
        plot_anchor[selected] = True
    track_groups: List[int] = []
    if tracks is not None and tracks["obs_track"].size:
        track_groups = sorted(
            int(g)
            for g in np.unique(tracks["summary_group"][tracks["obs_track"]])
        )
    n_selected = int(np.count_nonzero(plot_anchor))
    planned = (
        3 * n_selected
        + across_direct.shape[0]
        + across_acf.shape[0]
        + within_acf.shape[0]
        + len(track_groups)
    )

    files: List[Path] = []

    def _tick() -> None:
        if len(files) % 25 == 0 or len(files) == planned:
            print(f"[CORRELATIONS] {len(files)} {planned}", flush=True)

    any_pressure = bool(np.any(np.isfinite(peak_pressure)))
    for kind, matrix in (("roi_area", roi), ("location", location)):
        for anchor in range(matrix.shape[0]):
            if not plot_anchor[anchor]:
                continue
            folder = base / kind
            if any_pressure:
                folder = folder / _safe_pressure(float(peak_pressure[anchor]))
            path = folder / f"anchor_{anchor:04d}.png"
            grid = _target_grid(
                matrix[anchor], peak_frame, local_peak, original.shape[0]
            )
            _heatmap(
                grid,
                path,
                title=f"{kind.replace('_', ' ').title()} anchor {anchor}",
                x_label="peak slot within frame",
                y_label="frame row",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            files.append(path)
            _tick()

    for anchor in range(roi.shape[0]):
        if not plot_anchor[anchor]:
            continue
        folder = base / "waterfall"
        if any_pressure:
            folder = folder / _safe_pressure(float(peak_pressure[anchor]))
        path = folder / f"anchor_{anchor:04d}.png"
        _waterfall(
            path,
            radial=radial,
            original_positive=original,
            frame_indices=frame_indices,
            frame_pressure=frame_pressure,
            peak_frame=peak_frame,
            centers=centers,
            half_width=half_width,
            score=roi[anchor],
            anchor=anchor,
            unit=unit,
        )
        files.append(path)
        _tick()

    for label, matrices in (
        ("direct", across_direct),
        ("acf", across_acf),
    ):
        for window in range(matrices.shape[0]):
            path = (
                base
                / "window_across"
                / label
                / f"window_{window:03d}_{starts[window]:g}_{ends[window]:g}.png"
            )
            _heatmap(
                _strict_lower_triangle(matrices[window]),
                path,
                title=(
                    f"Across frames ({label.upper()}), "
                    f"window {starts[window]:g}-{ends[window]:g}"
                ),
                x_label="frame row",
                y_label="frame row",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
            )
            files.append(path)
            _tick()

    for frame in range(within_acf.shape[0]):
        path = base / "window_within" / "acf" / f"frame_{int(frame_indices[frame]):04d}.png"
        _heatmap(
            _strict_lower_triangle(within_acf[frame]),
            path,
            title=f"Within frame {int(frame_indices[frame])} (ACF)",
            x_label="window",
            y_label="window",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        files.append(path)
        _tick()

    if tracks is not None and track_groups:
        if tracks["order_by"] == "frame":
            axis_used = np.arange(original.shape[0], dtype=float)
            x_label = "frame row"
        else:
            axis_used = tracks["order_value"]
            x_label = tracks["order_label"]
        obs_x_all = axis_used[peak_frame[tracks["obs_peak"]]]
        obs_y_all = centers[tracks["obs_peak"]]
        obs_group = tracks["summary_group"][tracks["obs_track"]]
        edge_group = tracks["summary_group"][tracks["edge_track"]]
        edge_x_all = np.column_stack(
            (
                axis_used[peak_frame[tracks["edge_from"]]],
                axis_used[peak_frame[tracks["edge_to"]]],
            )
        )
        edge_y_all = np.column_stack(
            (centers[tracks["edge_from"]], centers[tracks["edge_to"]])
        )
        has_intervals = tracks["interval_group"].size > 0
        interval_group = np.asarray(tracks["interval_group"], int)
        interval_candidate = np.asarray(
            tracks["interval_transition_candidate"], bool
        )
        for group_id in track_groups:
            labels = tracks["group_labels"]
            label = (
                labels[group_id]
                if 0 <= group_id < len(labels)
                else f"group{group_id}"
            )
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
            name = (
                "tracks.png"
                if len(track_groups) == 1
                else f"tracks_{safe}.png"
            )
            path = base / "tracks" / name
            member = obs_group == group_id
            edge_member = edge_group == group_id
            if has_intervals:
                band = (interval_group == group_id) & interval_candidate
                band_from = np.asarray(
                    tracks["interval_axis_from"], float
                )[band]
                band_to = np.asarray(tracks["interval_axis_to"], float)[band]
            else:
                band_from = np.empty(0)
                band_to = np.empty(0)
            _track_map(
                path,
                obs_x=obs_x_all[member],
                obs_y=obs_y_all[member],
                obs_track=tracks["obs_track"][member],
                edge_x=edge_x_all[edge_member],
                edge_y=edge_y_all[edge_member],
                edge_similarity=tracks["edge_similarity"][edge_member],
                band_from=band_from,
                band_to=band_to,
                x_label=x_label,
                y_label=unit or "radial",
                group_label=label,
            )
            files.append(path)
            _tick()

    return files


def _remove_exact_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def render_all(
    correlations_h5: str | Path,
    heatmap_root: str | Path,
    *,
    max_anchor_plots: "int | None" = None,
) -> List[str]:
    """Render and replace one sample's managed heatmap tree.

    A complete tree is written beside the destination first. Only then is the
    exact ``heatmaps/<sample_type>`` directory swapped, so a rerun with fewer
    anchors/windows cannot leave stale PNGs. The sibling sample type is never
    touched.
    """

    import h5py  # type: ignore

    source = Path(correlations_h5).expanduser().resolve()
    root = Path(heatmap_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as h5:
        sample_type = str(h5.attrs.get("sample_type", "unknown"))
    destination = root / sample_type
    staging = Path(tempfile.mkdtemp(prefix=f".{sample_type}.tmp-", dir=str(root)))
    backup = root / f".{sample_type}.old-{os.getpid()}"
    _remove_exact_path(backup)
    try:
        staged_files = _render_into(source, staging, max_anchor_plots)
        had_destination = destination.exists()
        if had_destination:
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if had_destination and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        _remove_exact_path(backup)
    except Exception:
        _remove_exact_path(staging)
        raise

    output_parent = source.parent
    final_files = [destination / path.relative_to(staging) for path in staged_files]
    # POSIX-normalized: these relative paths land in a JSON manifest that must
    # read the same whoever opens it, and pathlib accepts forward slashes on
    # every platform when joining them back onto the output directory.
    return [path.relative_to(output_parent).as_posix() for path in final_files]


__all__ = ["render_all"]
