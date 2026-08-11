"""Agg-only plot exporters for correlation artifacts.

No pyplot or Tk backend is imported.  Figures are built with
``FigureCanvasAgg`` so the same code works in CI, over SSH, and from a worker
subprocess with no display server.
"""
from __future__ import annotations

import os
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
    fig = Figure(figsize=(9.0, max(4.0, 0.55 * n_frames + 2.2)), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    offset_step = 1.15

    for frame in range(n_frames):
        raw = np.asarray(original_positive[frame], dtype=float)
        finite = np.isfinite(raw)
        scale = float(np.nanpercentile(raw[finite], 99.0)) if np.any(finite) else 0.0
        scale = max(scale, np.finfo(float).eps)
        trace = np.where(finite, np.clip(raw, 0.0, None) / scale, np.nan)
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


def _render_into(correlations_h5: Path, base: Path) -> List[Path]:
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

    files: List[Path] = []
    any_pressure = bool(np.any(np.isfinite(peak_pressure)))
    for kind, matrix in (("roi_area", roi), ("location", location)):
        for anchor in range(matrix.shape[0]):
            if not anchor_valid[anchor]:
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

    for anchor in range(roi.shape[0]):
        if not anchor_valid[anchor]:
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

    return files


def _remove_exact_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def render_all(correlations_h5: str | Path, heatmap_root: str | Path) -> List[str]:
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
        staged_files = _render_into(source, staging)
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
    return [str(path.relative_to(output_parent)) for path in final_files]


__all__ = ["render_all"]
