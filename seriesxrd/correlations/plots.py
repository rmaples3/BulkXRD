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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _mpl():
    """Figure and Normalize only — deliberately no backend.

    Builders return a canvas-less ``Figure``. The exporter attaches Agg in
    :func:`save_figure`; the GUI attaches TkAgg when it embeds. Neither path
    can drag the other's backend into a process that cannot support it, which
    is what keeps this module usable from a display-less worker.
    """
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    return Figure, Normalize


def save_figure(fig, path: Path) -> None:
    """Atomically write one built figure to PNG, then release it."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        FigureCanvasAgg(fig)
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
    *,
    title: str,
    x_label: str,
    y_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
):
    """Build (do not save) one heatmap figure."""
    Figure, _ = _mpl()
    values = np.asarray(matrix, dtype=float)
    width = max(5.2, min(12.0, 0.28 * max(values.shape[-1], 1) + 3.0))
    height = max(3.8, min(10.0, 0.25 * max(values.shape[0], 1) + 2.5))
    fig = Figure(figsize=(width, height), facecolor="white")
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
    return fig


def _peaks_by_frame(peak_frame: np.ndarray, n_frames: int) -> "list[np.ndarray]":
    """Bucket peak rows by frame once, in ascending id order.

    The waterfall used to scan the whole peak table per frame, which is
    O(frames x peaks) inside *every* figure — invisible in a bulk render,
    but it is now interactive latency. One O(peaks) pass gives the same
    ascending row order the previous ``np.nonzero`` produced.
    """
    buckets: "list[list[int]]" = [[] for _ in range(int(n_frames))]
    frames = np.asarray(peak_frame, dtype=int)
    for row in range(frames.size):
        frame = int(frames[row])
        if 0 <= frame < n_frames:
            buckets[frame].append(row)
    return [np.asarray(rows, dtype=int) for rows in buckets]


def _waterfall(
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
    peak_rows_by_frame: "list[np.ndarray] | None" = None,
):
    """Build (do not save) one correlation-shaded waterfall figure."""
    Figure, Normalize = _mpl()
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    n_frames = original_positive.shape[0]
    if peak_rows_by_frame is None:
        peak_rows_by_frame = _peaks_by_frame(peak_frame, n_frames)
    # Cap the figure height: 0.55 in/frame is readable for tens of frames but
    # a few hundred frames would render a 100+ inch, 15000+ px PNG.
    height = min(max(4.0, 0.55 * n_frames + 2.2), 18.0)
    label_stride = 1
    if 0.55 * n_frames + 2.2 > 18.0:
        label_stride = max(1, -(-n_frames // 40))
    fig = Figure(figsize=(9.0, height), facecolor="white")
    ax = fig.add_subplot(111)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    offset_step = 1.15

    for frame in range(n_frames):
        trace = _normalize_trace(original_positive[frame])
        offset = frame * offset_step
        ax.plot(radial, trace + offset, color="#4c566a", linewidth=0.65, zorder=2)
        for target in peak_rows_by_frame[frame]:
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
    return fig


def _track_map(
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
):
    """Build (do not save) the peak-position-vs-condition track plot.

    Edges are colored by similarity; exploratory transition-candidate
    intervals are drawn as translucent vertical bands. Same viridis 0..1
    convention as the waterfall shading.
    """

    Figure, Normalize = _mpl()
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    fig = Figure(figsize=(9.5, 6.0), facecolor="white")
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
    return fig


FAMILIES = (
    "roi_area",
    "location",
    "waterfall",
    "window_across",
    "window_within",
    "tracks",
)


def _pressure_text(value: float) -> str:
    """Display label for a pressure, matching ``_pressure_label``'s output."""
    return f"{float(value):g} GPa" if np.isfinite(value) else "Pressure unavailable"


@dataclass(frozen=True)
class FigureSpec:
    """One renderable figure, identified without reference to any file.

    The catalogue of specs is what the GUI lists and what an export loops
    over, so a browsed figure and an exported PNG are by construction the
    same picture.
    """

    kind: str
    index: int
    relpath: str
    label: str
    variant: str = ""
    pressure: float = float("nan")
    pressure_label: str = "All pressures"


def _read_tracks(h5) -> "Dict[str, Any] | None":
    """Track arrays, or None for --no-tracks runs and older artifacts."""
    if "tracks" not in h5 or "tracks/summary/id" not in h5:
        return None
    tracks: "Dict[str, Any]" = {
        "obs_track": np.asarray(h5["tracks/obs/track"][:], int),
        "obs_peak": np.asarray(h5["tracks/obs/peak_id"][:], int),
        "edge_track": np.asarray(h5["tracks/edges/track"][:], int),
        "edge_from": np.asarray(h5["tracks/edges/peak_from"][:], int),
        "edge_to": np.asarray(h5["tracks/edges/peak_to"][:], int),
        "edge_similarity": np.asarray(h5["tracks/edges/similarity"][:], float),
        "summary_group": np.asarray(h5["tracks/summary/group"][:], int),
    }
    for name in ("axis_from", "axis_to", "group", "transition_candidate"):
        key = f"tracks/intervals/{name}"
        tracks[f"interval_{name}"] = (
            np.asarray(h5[key][:]) if key in h5 else np.empty(0)
        )
    tracks["group_labels"] = (
        [
            value.decode("utf-8", "replace")
            if isinstance(value, bytes)
            else str(value)
            for value in h5["tracks/group_label"][:]
        ]
        if "tracks/group_label" in h5
        else []
    )
    return tracks


class FigureContext:
    """Numpy snapshot of one correlation artifact, with lazy row access.

    The HDF5 handle is **closed** after the small index arrays are read.
    Holding one open would stop a later run from ``os.replace``-ing a fresh
    artifact onto the same path on Windows, and would pin the (K, K) and
    (W, M, M) matrices in memory — on a 1288-frame series those are hundreds
    of megabytes. Anything large is sliced on demand instead, which is what
    makes drawing a single figure cheap enough to do on a mouse click.

    Wrap a bulk export in :meth:`session` to keep one handle open for its
    duration rather than reopening per figure.
    """

    def __init__(self, path: "str | Path"):
        import h5py  # type: ignore

        self.path = Path(path).expanduser().resolve()
        self._h5 = None
        with h5py.File(str(self.path), "r") as h5:
            self.sample_type = str(h5.attrs.get("sample_type", "unknown"))
            self.unit = str(h5.attrs.get("unit", "radial"))
            self.order_by = str(h5.attrs.get("order_by", "frame"))
            self.order_label = str(h5.attrs.get("order_label", "Frame index"))
            self.radial = np.asarray(h5["patterns/radial"][:], float)
            self.patterns = np.asarray(
                h5["patterns/original_positive"][:], float
            )
            self.frame_index = np.asarray(h5["frames/index"][:], int)
            self.frame_pressure = np.asarray(h5["frames/pressure"][:], float)
            self.order_value = (
                np.asarray(h5["frames/order_value"][:], float)
                if "frames/order_value" in h5
                else np.arange(self.patterns.shape[0], dtype=float)
            )
            self.peak_frame = np.asarray(h5["peaks/frame_row"][:], int)
            self.local_peak = np.asarray(h5["peaks/local_peak"][:], int)
            self.peak_pressure = np.asarray(h5["peaks/pressure"][:], float)
            self.centers = np.asarray(h5["peaks/center"][:], float)
            self.half_width = np.asarray(h5["peaks/half_width"][:], float)
            # Artifacts written before /peaks/valid existed plot every anchor.
            self.anchor_valid = (
                np.asarray(h5["peaks/valid"][:], bool)
                if "peaks/valid" in h5
                else np.ones(self.peak_frame.size, dtype=bool)
            )
            self.window_start = np.asarray(h5["windows/start"][:], float)
            self.window_end = np.asarray(h5["windows/end"][:], float)
            self.tracks = _read_tracks(h5)
        self.n_frames = int(self.patterns.shape[0])
        self.n_peaks = int(self.centers.size)
        self.n_windows = int(self.window_start.size)
        self.peak_rows_by_frame = _peaks_by_frame(self.peak_frame, self.n_frames)
        self.any_pressure = bool(np.any(np.isfinite(self.peak_pressure)))

    @contextmanager
    def session(self):
        """Keep one handle open across many reads (bulk export)."""
        import h5py  # type: ignore

        if self._h5 is not None:
            yield self
            return
        self._h5 = h5py.File(str(self.path), "r")
        try:
            yield self
        finally:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    @contextmanager
    def _reader(self):
        import h5py  # type: ignore

        if self._h5 is not None:
            yield self._h5
        else:
            with h5py.File(str(self.path), "r") as h5:
                yield h5

    def anchor_row(self, kind: str, index: int) -> np.ndarray:
        with self._reader() as h5:
            return np.asarray(h5[f"anchor_maps/{kind}"][int(index)], float)

    def across(self, variant: str, window: int) -> np.ndarray:
        with self._reader() as h5:
            return np.asarray(
                h5[f"windows/across_{variant}"][int(window)], float
            )

    def within(self, frame: int) -> np.ndarray:
        with self._reader() as h5:
            return np.asarray(h5["windows/within_acf"][int(frame)], float)


def _track_groups(ctx: FigureContext) -> "List[tuple]":
    """(group_id, filename, label) for each group that has observations."""
    tracks = ctx.tracks
    if tracks is None or not tracks["obs_track"].size:
        return []
    ids = sorted(
        int(g) for g in np.unique(tracks["summary_group"][tracks["obs_track"]])
    )
    labels = tracks["group_labels"]
    out = []
    for group_id in ids:
        label = (
            labels[group_id]
            if 0 <= group_id < len(labels)
            else f"group{group_id}"
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        name = "tracks.png" if len(ids) == 1 else f"tracks_{safe}.png"
        out.append((group_id, name, label))
    return out


def figure_index(
    ctx: FigureContext,
    *,
    families: "Sequence[str] | None" = None,
    max_anchor_plots: "int | None" = None,
) -> "List[FigureSpec]":
    """Catalogue every figure this artifact can produce, in render order.

    This is the single source of truth for both the GUI's browsable list and
    what an export writes, so the two can never disagree about which figures
    exist. Structurally unusable anchors (``/peaks/valid`` false) are omitted
    from both, exactly as the exporter always skipped them.
    """

    chosen = tuple(families) if families else FAMILIES
    unknown = [name for name in chosen if name not in FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown figure families {unknown}; choose from {list(FAMILIES)}"
        )
    plot_anchor = ctx.anchor_valid.copy()
    if max_anchor_plots is not None:
        # Deterministic selection: the first N valid anchors in id order.
        selected = np.nonzero(ctx.anchor_valid)[0][: max(int(max_anchor_plots), 0)]
        plot_anchor = np.zeros_like(ctx.anchor_valid)
        plot_anchor[selected] = True

    specs: "List[FigureSpec]" = []
    for kind in ("roi_area", "location", "waterfall"):
        if kind not in chosen:
            continue
        for anchor in range(ctx.n_peaks):
            if not plot_anchor[anchor]:
                continue
            pressure = float(ctx.peak_pressure[anchor])
            folder = kind
            if ctx.any_pressure:
                folder = f"{kind}/{_safe_pressure(pressure)}"
            specs.append(
                FigureSpec(
                    kind=kind,
                    index=anchor,
                    relpath=f"{folder}/anchor_{anchor:04d}.png",
                    label=f"Anchor {anchor:04d}",
                    pressure=pressure,
                    pressure_label=_pressure_text(pressure),
                )
            )
    if "window_across" in chosen:
        for variant in ("direct", "acf"):
            for window in range(ctx.n_windows):
                start = float(ctx.window_start[window])
                end = float(ctx.window_end[window])
                specs.append(
                    FigureSpec(
                        kind="window_across",
                        index=window,
                        variant=variant,
                        relpath=(
                            f"window_across/{variant}/"
                            f"window_{window:03d}_{start:g}_{end:g}.png"
                        ),
                        label=f"Window {window:03d} — {start:g}–{end:g}",
                    )
                )
    if "window_within" in chosen:
        for frame in range(ctx.n_frames):
            original = int(ctx.frame_index[frame])
            pressure = float(ctx.frame_pressure[frame])
            specs.append(
                FigureSpec(
                    kind="window_within",
                    index=frame,
                    variant="acf",
                    relpath=f"window_within/acf/frame_{original:04d}.png",
                    label=f"Frame {original:04d}",
                    pressure=pressure,
                    pressure_label=_pressure_text(pressure),
                )
            )
    if "tracks" in chosen:
        for group_id, name, label in _track_groups(ctx):
            specs.append(
                FigureSpec(
                    kind="tracks",
                    index=group_id,
                    relpath=f"tracks/{name}",
                    label=f"Tracks ({label})",
                )
            )
    return specs


def _build_track_figure(ctx: FigureContext, group_id: int):
    tracks = ctx.tracks
    if tracks is None:
        raise ValueError("this artifact has no /tracks group")
    if ctx.order_by == "frame":
        axis_used = np.arange(ctx.n_frames, dtype=float)
        x_label = "frame row"
    else:
        axis_used = ctx.order_value
        x_label = ctx.order_label
    obs_x_all = axis_used[ctx.peak_frame[tracks["obs_peak"]]]
    obs_y_all = ctx.centers[tracks["obs_peak"]]
    obs_group = tracks["summary_group"][tracks["obs_track"]]
    edge_group = tracks["summary_group"][tracks["edge_track"]]
    edge_x_all = np.column_stack(
        (
            axis_used[ctx.peak_frame[tracks["edge_from"]]],
            axis_used[ctx.peak_frame[tracks["edge_to"]]],
        )
    )
    edge_y_all = np.column_stack(
        (ctx.centers[tracks["edge_from"]], ctx.centers[tracks["edge_to"]])
    )
    member = obs_group == group_id
    edge_member = edge_group == group_id
    if np.asarray(tracks["interval_group"]).size:
        band = (
            np.asarray(tracks["interval_group"], int) == group_id
        ) & np.asarray(tracks["interval_transition_candidate"], bool)
        band_from = np.asarray(tracks["interval_axis_from"], float)[band]
        band_to = np.asarray(tracks["interval_axis_to"], float)[band]
    else:
        band_from = np.empty(0)
        band_to = np.empty(0)
    label = next(
        (text for gid, _name, text in _track_groups(ctx) if gid == group_id),
        f"group{group_id}",
    )
    return _track_map(
        obs_x=obs_x_all[member],
        obs_y=obs_y_all[member],
        obs_track=tracks["obs_track"][member],
        edge_x=edge_x_all[edge_member],
        edge_y=edge_y_all[edge_member],
        edge_similarity=tracks["edge_similarity"][edge_member],
        band_from=band_from,
        band_to=band_to,
        x_label=x_label,
        y_label=ctx.unit or "radial",
        group_label=label,
    )


def build_figure(ctx: FigureContext, spec: FigureSpec):
    """Build one figure from the artifact. Never touches the filesystem."""

    if spec.kind in ("roi_area", "location"):
        grid = _target_grid(
            ctx.anchor_row(spec.kind, spec.index),
            ctx.peak_frame,
            ctx.local_peak,
            ctx.n_frames,
        )
        return _heatmap(
            grid,
            title=f"{spec.kind.replace('_', ' ').title()} anchor {spec.index}",
            x_label="peak slot within frame",
            y_label="frame row",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
    if spec.kind == "waterfall":
        return _waterfall(
            radial=ctx.radial,
            original_positive=ctx.patterns,
            frame_indices=ctx.frame_index,
            frame_pressure=ctx.frame_pressure,
            peak_frame=ctx.peak_frame,
            centers=ctx.centers,
            half_width=ctx.half_width,
            score=ctx.anchor_row("roi_area", spec.index),
            anchor=spec.index,
            unit=ctx.unit,
            peak_rows_by_frame=ctx.peak_rows_by_frame,
        )
    if spec.kind == "window_across":
        start = float(ctx.window_start[spec.index])
        end = float(ctx.window_end[spec.index])
        return _heatmap(
            _strict_lower_triangle(ctx.across(spec.variant, spec.index)),
            title=(
                f"Across frames ({spec.variant.upper()}), "
                f"window {start:g}-{end:g}"
            ),
            x_label="frame row",
            y_label="frame row",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
    if spec.kind == "window_within":
        original = int(ctx.frame_index[spec.index])
        return _heatmap(
            _strict_lower_triangle(ctx.within(spec.index)),
            title=f"Within frame {original} (ACF)",
            x_label="window",
            y_label="window",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
    if spec.kind == "tracks":
        return _build_track_figure(ctx, spec.index)
    raise ValueError(f"unknown figure kind: {spec.kind!r}")


def _render_into(
    correlations_h5: Path,
    base: Path,
    max_anchor_plots: "int | None" = None,
    families: "Sequence[str] | None" = None,
    *,
    resume: bool = False,
    progress: bool = True,
    state_dir: "Path | None" = None,
) -> List[Path]:
    """Render one sample tree into a staging directory.

    With ``resume=True`` a figure whose PNG already exists is skipped, which
    is what lets an interrupted export continue where it stopped: a partial
    PNG cannot exist, because :func:`save_figure` writes to a temporary name
    and renames.
    """

    source = Path(correlations_h5).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    ctx = FigureContext(source)
    specs = figure_index(
        ctx, families=families, max_anchor_plots=max_anchor_plots
    )
    planned = len(specs)
    files: List[Path] = []
    if state_dir is not None:
        # Record the denominator before the first figure: a run that dies
        # early must still be able to say "5 of 6183".
        from . import checkpoint as _checkpoint

        _checkpoint.write_state(state_dir, status="running", planned=planned)
    if resume:
        # A crash between savefig and os.replace can strand a *.png.tmp.
        for stray in base.rglob("*.png.tmp"):
            stray.unlink(missing_ok=True)

    with ctx.session():
        for spec in specs:
            path = base / spec.relpath
            if not (resume and path.is_file()):
                save_figure(build_figure(ctx, spec), path)
            files.append(path)
            if progress and (len(files) % 25 == 0 or len(files) == planned):
                print(f"[CORRELATIONS] {len(files)} {planned}", flush=True)
                if state_dir is not None:
                    # Doubles as the heartbeat that tells a later session
                    # whether this staging tree is still being written.
                    from . import checkpoint as _checkpoint

                    _checkpoint.write_state(
                        state_dir, status="running",
                        done=len(files), planned=planned,
                    )
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
    families: "Sequence[str] | None" = None,
    resume: bool = False,
) -> List[str]:
    """Render and replace one sample's managed heatmap tree.

    A complete tree is written beside the destination first. Only then is the
    exact ``heatmaps/<sample_type>`` directory swapped, so a rerun with fewer
    anchors/windows cannot leave stale PNGs. The sibling sample type is never
    touched.

    The staging directory is the checkpoint. With ``resume=True`` a
    compatible abandoned one is adopted and its finished figures skipped;
    and an interrupted or failed render **keeps** its staging directory so
    the work already done can be resumed or recovered rather than thrown
    away.
    """

    import h5py  # type: ignore

    from . import checkpoint as _checkpoint
    from ..core.provenance import file_fingerprint

    source = Path(correlations_h5).expanduser().resolve()
    root = Path(heatmap_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as h5:
        sample_type = str(h5.attrs.get("sample_type", "unknown"))
    destination = root / sample_type

    fingerprint = file_fingerprint(source)
    identity = {
        "artifact": str(source),
        "artifact_sha256": str(fingerprint.get("sha256", "")),
        "artifact_bytes": int(fingerprint.get("bytes", -1)),
        "max_anchor_plots": (
            None if max_anchor_plots is None else int(max_anchor_plots)
        ),
        "families": list(families) if families else None,
    }
    staging = None
    if resume:
        for candidate in _checkpoint.find_staging(root.parent, sample_type):
            state = _checkpoint.read_state(candidate)
            if _checkpoint.is_live(candidate):
                continue
            if all(state.get(k) == v for k, v in identity.items()):
                staging = candidate
                break
    if staging is None:
        staging = Path(
            tempfile.mkdtemp(prefix=_checkpoint.staging_prefix(sample_type),
                             dir=str(root))
        )
    backup = root / f".{sample_type}.old-{os.getpid()}"
    _remove_exact_path(backup)
    _checkpoint.write_state(staging, status="running", sample_type=sample_type,
                            **identity)
    try:
        staged_files = _render_into(
            source, staging, max_anchor_plots, families,
            resume=resume, state_dir=staging,
        )
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
        _remove_exact_path(destination / _checkpoint.STATE_FILENAME)
    except BaseException:
        # Keep the staging tree: the figures already rendered are good, and
        # discarding them is what made an interrupted export unresumable.
        _checkpoint.write_state(
            staging, status="interrupted", sample_type=sample_type, **identity
        )
        raise

    output_parent = source.parent
    final_files = [destination / path.relative_to(staging) for path in staged_files]
    # POSIX-normalized: these relative paths land in a JSON manifest that must
    # read the same whoever opens it, and pathlib accepts forward slashes on
    # every platform when joining them back onto the output directory.
    return [path.relative_to(output_parent).as_posix() for path in final_files]


__all__ = [
    "FAMILIES",
    "FigureContext",
    "FigureSpec",
    "build_figure",
    "figure_index",
    "render_all",
    "save_figure",
]
