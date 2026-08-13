"""Focused rendering contracts for Correlations window heatmaps."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from seriesxrd.correlations import plots


def test_strict_lower_triangle_masks_mirror_and_diagonal_without_mutation():
    matrix = np.arange(1.0, 10.0).reshape(3, 3)
    original = matrix.copy()

    shown = plots._strict_lower_triangle(matrix)

    expected = np.asarray(
        [
            [np.nan, np.nan, np.nan],
            [4.0, np.nan, np.nan],
            [7.0, 8.0, np.nan],
        ]
    )
    np.testing.assert_allclose(shown, expected, equal_nan=True)
    np.testing.assert_array_equal(matrix, original)

    with pytest.raises(ValueError, match="square 2D matrix"):
        plots._strict_lower_triangle(np.ones((2, 3)))


def _write_plot_artifact(path: Path):
    direct = np.asarray(
        [
            [[1.0, 0.2], [0.2, 1.0]],
            [[1.0, -0.4], [-0.4, 1.0]],
        ]
    )
    across_acf = np.asarray(
        [
            [[1.0, 0.7], [0.7, 1.0]],
            [[1.0, 0.1], [0.1, 1.0]],
        ]
    )
    within_acf = np.asarray(
        [
            [[1.0, 0.8], [0.8, 1.0]],
            [[1.0, -0.3], [-0.3, 1.0]],
        ]
    )
    with h5py.File(str(path), "w") as h5:
        h5.attrs["sample_type"] = "powder"
        h5.attrs["unit"] = "2th_deg"
        patterns = h5.create_group("patterns")
        patterns.create_dataset("radial", data=np.asarray([1.0, 2.0, 3.0]))
        patterns.create_dataset("original_positive", data=np.ones((2, 3)))
        frames = h5.create_group("frames")
        frames.create_dataset("index", data=np.asarray([10, 11]))
        frames.create_dataset("pressure", data=np.asarray([1.0, 2.0]))
        peaks = h5.create_group("peaks")
        peaks.create_dataset("frame_row", data=np.empty(0, dtype=int))
        peaks.create_dataset("local_peak", data=np.empty(0, dtype=int))
        peaks.create_dataset("pressure", data=np.empty(0, dtype=float))
        peaks.create_dataset("center", data=np.empty(0, dtype=float))
        peaks.create_dataset("half_width", data=np.empty(0, dtype=float))
        anchors = h5.create_group("anchor_maps")
        anchors.create_dataset("roi_area", data=np.empty((0, 0), dtype=float))
        anchors.create_dataset("location", data=np.empty((0, 0), dtype=float))
        windows = h5.create_group("windows")
        windows.create_dataset("start", data=np.asarray([1.0, 2.0]))
        windows.create_dataset("end", data=np.asarray([2.0, 3.0]))
        windows.create_dataset("across_direct", data=direct)
        windows.create_dataset("across_acf", data=across_acf)
        windows.create_dataset("within_acf", data=within_acf)
    return direct, across_acf, within_acf


def test_window_render_paths_use_triangle_mask_and_leave_h5_full(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "correlations_powder.h5"
    stored = _write_plot_artifact(artifact)
    helper_inputs = []
    shown_matrices = []
    saved_paths = []
    real_mask = plots._strict_lower_triangle

    class _StubFigure:
        def savefig(self, *_args, **_kwargs):
            pass

        def clear(self):
            pass

    def spy_mask(matrix):
        helper_inputs.append(np.asarray(matrix, dtype=float).copy())
        return real_mask(matrix)

    def capture_heatmap(matrix, **_kwargs):
        shown_matrices.append(np.asarray(matrix, dtype=float).copy())
        return _StubFigure()

    monkeypatch.setattr(plots, "_strict_lower_triangle", spy_mask)
    monkeypatch.setattr(plots, "_heatmap", capture_heatmap)
    monkeypatch.setattr(
        plots, "save_figure", lambda fig, path: saved_paths.append(Path(path))
    )

    files = plots._render_into(artifact, tmp_path / "rendered")

    full_matrices = [matrix for group in stored for matrix in group]
    assert len(helper_inputs) == len(shown_matrices) == len(files) == 6
    assert len(saved_paths) == 6
    assert all(
        "window_across" in path.parts or "window_within" in path.parts
        for path in saved_paths
    )
    for source, helper_input, shown in zip(
        full_matrices, helper_inputs, shown_matrices
    ):
        np.testing.assert_array_equal(helper_input, source)
        np.testing.assert_allclose(shown, real_mask(source), equal_nan=True)

    # Rendering is a view concern: the artifact retains both symmetric halves
    # and its unit diagonal for downstream numerical use.
    with h5py.File(str(artifact), "r") as h5:
        np.testing.assert_array_equal(h5["windows/across_direct"][:], stored[0])
        np.testing.assert_array_equal(h5["windows/across_acf"][:], stored[1])
        np.testing.assert_array_equal(h5["windows/within_acf"][:], stored[2])


def test_waterfall_height_capped(tmp_path, monkeypatch):
    """A many-frame series renders a bounded figure, not a 100-inch PNG."""
    heights = []

    radial = np.linspace(1.0, 3.0, 50)

    def _build(n_frames):
        fig = plots._waterfall(
            radial=radial,
            original_positive=np.ones((n_frames, radial.size)),
            frame_indices=np.arange(n_frames),
            frame_pressure=np.full(n_frames, np.nan),
            peak_frame=np.asarray([0]),
            centers=np.asarray([2.0]),
            half_width=np.asarray([0.2]),
            score=np.asarray([0.5]),
            anchor=0,
            unit="q_A^-1",
        )
        height = float(fig.get_size_inches()[1])
        fig.clear()
        return height

    heights.append(_build(200))
    assert heights[0] <= 18.0
    # A small series keeps its full per-frame layout.
    heights.append(_build(6))
    assert heights[1] == pytest.approx(0.55 * 6 + 2.2)


def test_catalogue_matches_what_export_writes(tmp_path):
    """One source of truth: every catalogued spec is exported, and nothing
    else is — so a browsed figure and an exported PNG cannot disagree."""
    from seriesxrd.correlations.plots import (
        FigureContext, figure_index, render_all,
    )
    from seriesxrd.correlations.processing import run_correlations
    from tests.test_correlations_processing import _write_analysis

    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder",
    )
    artifact = Path(manifest["correlations_h5"])
    ctx = FigureContext(artifact)
    specs = figure_index(ctx)
    written = render_all(artifact, tmp_path / "res" / "heatmaps")

    assert {s.relpath for s in specs} == {
        f.split("heatmaps/powder/", 1)[1] for f in written
    }
    # Invalid anchors are absent from both, and the render order is the
    # catalogue order.
    assert [s.relpath for s in specs] == [
        f.split("heatmaps/powder/", 1)[1] for f in written
    ]

    # Selecting a family exports exactly that family.
    only_tracks = render_all(
        artifact, tmp_path / "res" / "heatmaps2", families=("tracks",)
    )
    assert only_tracks and all("tracks/" in f for f in only_tracks)
    assert len(only_tracks) == len(
        [s for s in specs if s.kind == "tracks"]
    )


def test_live_build_matches_exported_png(tmp_path):
    """The same spec rendered live and saved to disk is the same picture."""
    from seriesxrd.correlations.plots import (
        FigureContext, build_figure, figure_index, save_figure,
    )
    from seriesxrd.correlations.processing import run_correlations
    from tests.test_correlations_processing import _write_analysis

    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder",
        plots=("all",), max_anchor_plots=1,
    )
    artifact = Path(manifest["correlations_h5"])
    ctx = FigureContext(artifact)
    for spec in figure_index(ctx, max_anchor_plots=1)[:4]:
        exported = tmp_path / "res" / "heatmaps" / "powder" / spec.relpath
        assert exported.is_file()
        live = tmp_path / "live.png"
        save_figure(build_figure(ctx, spec), live)
        assert live.read_bytes() == exported.read_bytes(), spec.relpath


def test_waterfall_normalization_resists_zingers():
    """A 1-2 bin spike cannot flatten the trace: the stackplot-shared robust
    scale wins over any percentile of the raw values."""
    radial = np.linspace(0.0, 4.0, 200)
    signal = 1.0 + np.exp(-0.5 * ((radial - 2.0) / 0.1) ** 2) * 5.0
    spiked = signal.copy()
    spiked[50] = 5000.0                      # single-bin zinger

    clean_trace = plots._normalize_trace(signal)
    spiked_trace = plots._normalize_trace(spiked)
    # The real peak keeps essentially its full height despite the zinger.
    peak_bin = int(np.argmax(signal))
    assert spiked_trace[peak_bin] > 0.8 * clean_trace[peak_bin]
    assert np.isnan(plots._normalize_trace(np.full(5, np.nan))).all()


def test_anchor_plot_cap_and_validity_skip(tmp_path):
    """max_anchor_plots caps per-anchor PNGs deterministically; the HDF5
    matrices stay complete."""
    from tests.test_correlations_processing import _write_analysis
    from seriesxrd.correlations.processing import run_correlations

    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis,
        tmp_path / "res",
        sample_type="powder",
        plots=("all",),
        max_anchor_plots=2,
    )
    assert manifest["anchor_plot_cap"] == 2
    per_anchor = [f for f in manifest["plot_files"] if "anchor_" in f]
    assert len(per_anchor) == 3 * 2
    named = {Path(f).name for f in per_anchor}
    assert named == {"anchor_0000.png", "anchor_0001.png"}
    with h5py.File(manifest["correlations_h5"], "r") as h5:
        assert h5["anchor_maps/roi_area"].shape == (8, 8)
