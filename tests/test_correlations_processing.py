"""Numerical and HDF5 contracts for the Correlations stage."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from seriesxrd.correlations.processing import (
    PeakTable,
    _roi_profiles,
    compute_window_correlations,
    directional_anchor_iou,
    integrated_iou,
    location_similarity,
    log_squared_transform,
    relative_feature_similarity,
    run_correlations,
)
from seriesxrd.correlations.review import inspect_correlations, load_anchor_map


def _gaussian(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def _write_analysis(
    path: Path,
    *,
    include_spots: bool = True,
    excluded_idx: tuple = (),
    edge_peak: bool = False,
    pressures: "tuple | None" = None,
) -> Path:
    radial = np.linspace(0.0, 10.0, 241)
    n_frames = 4
    clean = np.zeros((n_frames, radial.size), dtype=float)
    peak_rows = []
    spot_rows = []
    if edge_peak:
        # Center inside the axis but ROI support [-0.2, 0.4] crossing radial[0].
        peak_rows.append((0, 0.1, 0.4, 2.0))
    for frame in range(n_frames):
        shift = 0.03 * frame
        clean[frame] = (
            0.4
            + _gaussian(radial, 2.0 + shift, 0.10, 8.0 + frame)
            + _gaussian(radial, 6.0 - shift, 0.16, 5.0 + 0.5 * frame)
        )
        for center, width, area in (
            (2.0 + shift, 0.24, 8.0 + frame),
            (6.0 - shift, 0.32, 5.0 + frame),
        ):
            peak_rows.append((frame, center, width, area))
        if include_spots:
            # Three observations per frame but only two track ids. Correlations
            # must retain all 12 observations as independent anchors.
            for local, (center, width, area) in enumerate(
                (
                    (2.0 + shift, 0.12, 8.0 + frame),
                    (4.0 + 0.01 * frame, 0.10, 4.0 + frame),
                    (6.0 - shift, 0.15, 5.0 + frame),
                )
            ):
                spot_rows.append(
                    (frame, center, width, area, local % 2)
                )
    # A signed residual bin outside every peak support distinguishes the
    # positive ROI path from the signed-input window path.
    clean[:, 100] = -np.asarray([0.75, 1.0, 1.25, 1.5])

    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "2th_deg"
        h5.attrs["source_reduced"] = str(path.with_name("synthetic_reduced.h5"))
        h5.create_dataset("radial", data=radial)
        background = h5.create_group("background")
        background.create_dataset("clean", data=clean)
        background.create_dataset("baseline", data=np.zeros_like(clean))
        background.create_dataset("spot_residual", data=0.2 * clean)
        frames = h5.create_group("frames")
        frames.create_dataset(
            "filename",
            data=np.asarray([f"scan_{i:03d}.tif" for i in range(n_frames)], object),
            dtype=h5py.string_dtype("utf-8"),
        )
        frame_pressures = np.asarray(
            [1.0, 1.0, 5.0, 5.0] if pressures is None else pressures, float
        )
        frames.create_dataset("pressure", data=frame_pressures)
        excluded = np.zeros(n_frames, bool)
        for index in excluded_idx:
            excluded[index] = True
        frames.create_dataset("excluded", data=excluded)

        peaks = h5.create_group("peaks")
        peaks.attrs["source"] = "clean"
        peak_array = np.asarray(peak_rows, dtype=float)
        peaks.create_dataset("frame", data=peak_array[:, 0].astype("i4"))
        peaks.create_dataset("center", data=peak_array[:, 1])
        peaks.create_dataset("fwhm", data=peak_array[:, 2])
        peaks.create_dataset("area", data=peak_array[:, 3])
        peaks.create_dataset("flag", data=np.zeros(len(peak_rows), "i4"))
        peaks.create_dataset("counts", data=np.full(n_frames, 2, "i4"))

        if include_spots:
            spots = h5.create_group("spots").create_group("obs")
            spot_array = np.asarray(spot_rows, dtype=float)
            spots.create_dataset("frame", data=spot_array[:, 0].astype("i4"))
            spots.create_dataset("q", data=spot_array[:, 1])
            spots.create_dataset("q_width", data=spot_array[:, 2])
            spots.create_dataset("area", data=spot_array[:, 3])
            spots.create_dataset("intensity", data=spot_array[:, 3])
            spots.create_dataset("track", data=spot_array[:, 4].astype("i4"))
            spots.create_dataset(
                "pressure", data=np.repeat([1.0, 1.0, 5.0, 5.0], 3)
            )
    return path


def test_log_squared_is_fixed_bounded_and_uses_one_scale():
    values = np.asarray([[-2.0, 0.0, 5.0], [10.0, 20.0, np.nan]])
    transformed, params = log_squared_transform(
        values, scale=10.0, noise_floor=1.0
    )
    epsilon = 0.01
    expected_mid = np.log1p(0.5**2 / epsilon) / np.log1p(1.0 / epsilon)
    assert params.method == "log_squared"
    assert params.scale == 10.0 and params.epsilon == pytest.approx(epsilon)
    assert transformed[0, 0] == 0.0
    assert transformed[0, 1] == 0.0
    assert transformed[0, 2] == pytest.approx(expected_mid)
    assert transformed[1, 0] == 1.0 and transformed[1, 1] == 1.0
    assert np.isnan(transformed[1, 2])
    assert np.nanmin(transformed) >= 0.0 and np.nanmax(transformed) <= 1.0


def test_core_similarity_contracts_include_scalar_location():
    x = np.linspace(-1.0, 1.0, 21)
    peak = np.exp(-0.5 * (x / 0.25) ** 2)
    assert integrated_iou(peak, peak, x) == pytest.approx(1.0)
    assert integrated_iou(
        np.r_[np.ones(10), np.zeros(11)],
        np.r_[np.zeros(10), np.ones(11)],
        x,
    ) == pytest.approx(0.0)
    assert integrated_iou(np.zeros_like(x), np.zeros_like(x), x) == 0.0
    scalar = location_similarity(1.0, 1.01, tolerance=0.02)
    assert np.asarray(scalar).shape == ()
    assert float(scalar) == pytest.approx(0.5)
    assert np.isnan(float(location_similarity(np.nan, 1.0, tolerance=0.02)))


def test_window_fft_acf_uses_every_positive_lag_and_preserves_invalidity():
    radial = np.linspace(0.0, 4.0, 65)
    phase = np.linspace(0.0, 4.0 * np.pi, radial.size)
    transformed = np.vstack(
        (
            np.sin(phase) ** 2,
            np.cos(phase + 0.2) ** 2,
            np.ones(radial.size),
        )
    )
    result = compute_window_correlations(
        radial,
        transformed,
        window_width=4.0,
        window_step=1.0,
    )
    assert result["start"].shape == (1,)
    assert result["acf_features"].shape == (3, 1, 63)
    for fingerprint in result["acf_features"][:2, 0]:
        assert np.all(np.isfinite(fingerprint))
        assert np.mean(fingerprint) == pytest.approx(0.0, abs=1.0e-12)
        assert np.std(fingerprint) == pytest.approx(1.0, abs=1.0e-12)
    assert np.all(np.isnan(result["acf_features"][2, 0]))
    assert np.all(np.isnan(result["across_acf"][0, 2]))
    assert np.all(np.isnan(result["across_acf"][0, :, 2]))
    assert np.isnan(result["within_acf"][2, 0, 0])

    masked = transformed.copy()
    masked[0, 10] = np.nan
    masked_result = compute_window_correlations(
        radial,
        masked,
        window_width=4.0,
        window_step=1.0,
    )
    assert np.all(np.isnan(masked_result["signals"][0, 0]))
    assert np.all(np.isnan(masked_result["acf_features"][0, 0]))

    with pytest.raises(ValueError, match="exceed the selected radial span"):
        compute_window_correlations(
            radial,
            transformed,
            window_width=4.01,
            window_step=1.0,
        )


def test_frozen_powder_directional_absolute_support_contract():
    radial = np.linspace(0.0, 4.0, 401)
    anchor = np.ones(radial.size)
    target = np.ones(radial.size)
    broad_to_narrow = directional_anchor_iou(
        radial,
        anchor,
        target,
        anchor_support=(1.0, 3.0),
        target_support=(1.5, 2.5),
    )
    narrow_to_broad = directional_anchor_iou(
        radial,
        target,
        anchor,
        anchor_support=(1.5, 2.5),
        target_support=(1.0, 3.0),
    )
    assert broad_to_narrow == pytest.approx(0.5)
    assert narrow_to_broad == pytest.approx(1.0)
    assert broad_to_narrow != narrow_to_broad
    assert directional_anchor_iou(
        radial,
        anchor,
        target,
        anchor_support=(0.5, 1.0),
        target_support=(2.0, 2.5),
    ) == pytest.approx(0.0)
    assert directional_anchor_iou(
        radial,
        np.zeros_like(anchor),
        np.zeros_like(target),
        anchor_support=(1.0, 2.0),
        target_support=(1.0, 2.0),
    ) == pytest.approx(0.0)


def test_frozen_single_scalar_minmax_contract_including_both_zero():
    features = np.asarray([0.0, 2.0, 4.0])
    matrix = relative_feature_similarity(features[:, None], features[None, :])
    # Two dead ROIs share absence, not similarity: both-zero scores 0, the
    # same zero-signal convention as the powder directional IoU.
    assert matrix[0, 0] == 0.0
    assert matrix[0, 1] == 0.0
    assert matrix[1, 2] == pytest.approx(0.5)
    assert np.allclose(np.diag(matrix)[1:], 1.0)


def test_roi_nan_policy_is_structural():
    """A masked bin inside a support voids the comparison; outside it doesn't."""
    radial = np.linspace(0.0, 4.0, 401)
    flat = np.ones(radial.size)

    poisoned_anchor = flat.copy()
    poisoned_anchor[200] = np.nan          # radial 2.0, inside (1.0, 3.0)
    assert np.isnan(directional_anchor_iou(
        radial, poisoned_anchor, flat,
        anchor_support=(1.0, 3.0), target_support=(1.0, 3.0),
    ))

    poisoned_target = flat.copy()
    poisoned_target[200] = np.nan          # inside the support overlap
    assert np.isnan(directional_anchor_iou(
        radial, flat, poisoned_target,
        anchor_support=(1.0, 3.0), target_support=(1.5, 2.5),
    ))

    # The same masked bin outside the anchor support and outside the overlap
    # leaves the score finite.
    assert directional_anchor_iou(
        radial, poisoned_anchor, flat,
        anchor_support=(2.5, 3.5), target_support=(2.5, 3.5),
    ) == pytest.approx(1.0)
    assert directional_anchor_iou(
        radial, flat, poisoned_target,
        anchor_support=(2.5, 3.5), target_support=(2.5, 3.5),
    ) == pytest.approx(1.0)

    # _roi_profiles no longer bridges a masked bin: the poisoned samples of
    # the affected anchor come back NaN instead of interpolated.
    peaks = PeakTable(
        source_index=np.asarray([0]),
        frame_row=np.asarray([0]),
        original_frame=np.asarray([0]),
        local_peak=np.asarray([0]),
        center=np.asarray([2.0]),
        width=np.asarray([1.0]),
        half_width=np.asarray([0.75]),
        area=np.asarray([1.0]),
        pressure=np.asarray([np.nan]),
        track=np.asarray([-1]),
    )
    coordinate, profiles = _roi_profiles(
        radial, poisoned_anchor[None, :], peaks
    )
    assert np.any(np.isnan(profiles[0]))
    grid = 2.0 + coordinate * 0.75
    near_masked = np.abs(grid - 2.0) < 0.02
    assert np.all(np.isnan(profiles[0][near_masked]))
    assert np.all(np.isfinite(profiles[0][np.abs(grid - 2.0) > 0.1]))


def test_order_by_pressure_reorders_frames_and_peaks(tmp_path):
    """--order-by pressure permutes frames, patterns, and peak frame_rows
    coherently; the default keeps Analysis file order byte-identically."""
    analysis = _write_analysis(
        tmp_path / "analysis.h5", pressures=(5.0, 1.0, 5.0, 1.0)
    )
    default = run_correlations(
        analysis, tmp_path / "default", sample_type="powder", make_plots=False
    )
    ordered = run_correlations(
        analysis, tmp_path / "ordered", sample_type="powder",
        make_plots=False, order_by="pressure",
    )
    assert default["order_by"] == "frame"
    assert ordered["order_by"] == "pressure"
    assert ordered["order_label"] == "Pressure (GPa)"

    with h5py.File(default["correlations_h5"], "r") as h5:
        assert np.asarray(h5["frames/index"][:]).tolist() == [0, 1, 2, 3]
        assert np.asarray(h5["frames/order_value"][:]).tolist() == [0, 1, 2, 3]

    with h5py.File(analysis, "r") as src:
        clean = np.asarray(src["background/clean"][:], float)
    with h5py.File(ordered["correlations_h5"], "r") as h5:
        assert h5.attrs["order_by"] == "pressure"
        index = np.asarray(h5["frames/index"][:], int)
        # Ascending pressure, original order as tie-break: 1 GPa frames first.
        assert index.tolist() == [1, 3, 0, 2]
        assert np.asarray(h5["frames/order_value"][:]).tolist() == [
            1.0, 1.0, 5.0, 5.0,
        ]
        assert np.asarray(h5["frames/pressure"][:]).tolist() == [
            1.0, 1.0, 5.0, 5.0,
        ]
        frame_row = np.asarray(h5["peaks/frame_row"][:], int)
        original_frame = np.asarray(h5["peaks/original_frame"][:], int)
        centers = np.asarray(h5["peaks/center"][:], float)
        original_positive = np.asarray(
            h5["patterns/original_positive"][:], float
        )
    # Pattern rows follow the permutation.
    expected = np.where(
        np.isfinite(clean[index]), np.clip(clean[index], 0.0, None), np.nan
    )
    assert np.allclose(original_positive, expected, equal_nan=True)
    # Peaks stay attached to their frames through the permutation.
    assert np.array_equal(original_frame, index[frame_row])
    for row, frame in enumerate(index):
        shift = 0.03 * frame
        got = np.sort(centers[frame_row == row])
        assert got == pytest.approx([2.0 + shift, 6.0 - shift])


def test_vectorized_powder_matrix_matches_scalar_reference():
    """The vectorized K x K matrix reproduces directional_anchor_iou exactly.

    The fixture deliberately mixes on/off-grid supports, overlapping,
    disjoint, nested, and edge-crossing supports, same-frame pairs, a masked
    bin inside one frame, and a NaN bin outside every support (which the
    scalar bridges via finite-subset interpolation)."""
    from seriesxrd.correlations.processing import _powder_roi_matrix

    rng = np.random.default_rng(11)
    radial = np.linspace(0.0, 8.0, 161)
    n_frames = 3
    profiles = 0.2 + rng.random((n_frames, radial.size))
    profiles[1, 40] = np.nan          # masked bin at radial 2.0 (frame 1)
    profiles[2, 150] = np.nan         # masked bin at radial 7.5, outside supports

    frame_row = np.asarray([0, 0, 1, 1, 2, 2, 0, 2])
    center = np.asarray([1.0, 3.0, 2.0, 3.05, 1.0, 5.0, 0.05, 3.0])
    half_width = np.asarray([0.4, 0.5, 0.3, 0.45, 0.4, 0.6, 0.2, 0.512])
    size = center.size
    peaks = PeakTable(
        source_index=np.arange(size),
        frame_row=frame_row,
        original_frame=frame_row.copy(),
        local_peak=np.zeros(size, int),
        center=center,
        width=half_width / 0.75,
        half_width=half_width,
        area=np.ones(size),
        pressure=np.full(size, np.nan),
        track=np.full(size, -1),
    )

    matrix = _powder_roi_matrix(radial, profiles, peaks)
    reference = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            reference[i, j] = directional_anchor_iou(
                radial,
                profiles[frame_row[i]],
                profiles[frame_row[j]],
                anchor_support=(center[i] - half_width[i],
                                center[i] + half_width[i]),
                target_support=(center[j] - half_width[j],
                                center[j] + half_width[j]),
            )
    assert np.allclose(matrix, reference, atol=1.0e-9, equal_nan=True)
    # The fixture must actually exercise the interesting regimes.
    assert np.all(np.isnan(reference[2]))            # masked-bin anchor row
    assert np.all(np.isnan(reference[6]))            # edge-crossing anchor row
    finite = np.isfinite(reference)
    assert finite.sum() > 10
    assert np.any(reference[finite] == 0.0)          # disjoint pairs present


def test_batched_kernels_match_reference():
    """Gram-matrix Pearson, batched ACF, and vectorized window resampling
    agree with their scalar/loop references, including NaN and constant rows."""
    from seriesxrd.correlations.processing import (
        _acf,
        _acf_batch,
        _pairwise_correlations,
        _pearson,
        _resample_windows,
    )

    rng = np.random.default_rng(5)
    rows = rng.normal(size=(6, 40))
    rows[2, 7] = np.nan                     # invalid row
    rows[3] = 1.25                          # constant row -> zero variance
    rows[4] = rows[0]                       # perfectly correlated pair

    pearson = _pairwise_correlations(rows)
    for i in range(rows.shape[0]):
        for j in range(rows.shape[0]):
            expected = _pearson(rows[i], rows[j])
            got = pearson[i, j]
            assert (np.isnan(got) and np.isnan(expected)) or (
                got == pytest.approx(expected, abs=1.0e-12)
            )
    assert pearson[0, 4] == pytest.approx(1.0)

    acf = _acf_batch(rows)
    for i in range(rows.shape[0]):
        row = rows[i]
        length = row.size - 1
        expected = np.full(length, np.nan)
        if np.all(np.isfinite(row)):
            centered = row - np.mean(row)
            scale = np.std(centered)
            if scale > np.finfo(float).eps * max(np.max(np.abs(centered)), 1.0):
                standardized = centered / scale
                spectrum = np.fft.rfft(standardized, n=2 * row.size)
                corr = np.fft.irfft(
                    spectrum * np.conjugate(spectrum), n=2 * row.size
                )[: row.size]
                if corr[0] > np.finfo(float).eps:
                    lags = corr[1:] / corr[0]
                    lags = lags - np.mean(lags)
                    fscale = np.std(lags)
                    if fscale > 1.0e-12:
                        expected = lags / fscale
        assert np.allclose(acf[i], expected, atol=1.0e-9, equal_nan=True)
        assert np.allclose(_acf(row), expected, atol=1.0e-9, equal_nan=True)

    radial = np.linspace(0.0, 4.0, 81)
    values = rng.normal(size=(4, radial.size))
    values[1, 30] = np.nan
    starts = np.asarray([0.0, 0.55, 1.9])
    ends = starts + 2.0
    resampled = _resample_windows(radial, values, starts, ends, points=33)
    for w, (start, end) in enumerate(zip(starts, ends)):
        grid = np.linspace(start, end, 33)
        li = max(int(np.searchsorted(radial, start, side="right")) - 1, 0)
        ri = min(int(np.searchsorted(radial, end, side="left")), radial.size - 1)
        for f in range(values.shape[0]):
            support = values[f, li : ri + 1]
            if support.size < 2 or not np.all(np.isfinite(support)):
                expected = np.full(33, np.nan)
            else:
                expected = np.interp(
                    grid, radial[li : ri + 1], support,
                    left=np.nan, right=np.nan,
                )
            assert np.allclose(
                resampled[f, w], expected, atol=1.0e-12, equal_nan=True
            )
    # The masked frame is structurally voided exactly where the NaN bin
    # falls inside the window support.
    assert np.all(np.isnan(resampled[1, 0]))
    assert np.all(np.isnan(resampled[1, 1]))
    assert np.all(np.isfinite(resampled[1, 2]))


def test_edge_anchor_flagged_and_unplotted(tmp_path):
    """A support crossing the radial boundary flags the anchor invalid,
    counts it in the manifest, and produces no per-anchor PNGs for it."""
    from seriesxrd.correlations.plots import render_all

    analysis = _write_analysis(tmp_path / "analysis.h5", edge_peak=True)
    out = tmp_path / "res"
    manifest = run_correlations(
        analysis, out, sample_type="powder", make_plots=False
    )
    assert manifest["n_peaks"] == 9
    assert manifest["n_anchors_valid"] == 8
    assert manifest["n_anchors_edge"] == 1
    assert manifest["n_anchors_masked"] == 0

    h5_path = Path(manifest["correlations_h5"])
    with h5py.File(str(h5_path), "r") as h5:
        valid = np.asarray(h5["peaks/valid"][:], bool)
        centers = np.asarray(h5["peaks/center"][:], float)
        roi = np.asarray(h5["anchor_maps/roi_area"][:], float)
    assert valid.sum() == 8
    edge = np.nonzero(~valid)[0]
    assert centers[edge] == pytest.approx([0.1])
    # The invalid anchor's whole score row is structurally NaN.
    assert np.all(np.isnan(roi[edge[0]]))

    files = render_all(h5_path, out / "heatmaps")
    names = {Path(f).name for f in files}
    assert f"anchor_{int(edge[0]):04d}.png" not in names
    # Valid anchors still get all three per-anchor plots.
    per_anchor = [f for f in files if "anchor_" in f]
    assert len(per_anchor) == 3 * 8

    # An artifact written without /peaks/valid still renders and inspects.
    with h5py.File(str(h5_path), "r+") as h5:
        del h5["peaks/valid"]
    legacy = inspect_correlations(h5_path)
    assert legacy["ok"]
    assert "n_anchors_valid" not in legacy
    legacy_files = render_all(h5_path, out / "heatmaps_legacy")
    assert len([f for f in legacy_files if "anchor_" in f]) == 3 * 9


def test_excluded_frames_remap_peak_rows(tmp_path):
    """Excluding a frame drops its patterns AND remaps every peak's frame_row."""
    analysis = _write_analysis(tmp_path / "analysis.h5", excluded_idx=(1,))
    out = tmp_path / "res"
    manifest = run_correlations(
        analysis, out, sample_type="powder", make_plots=False
    )
    assert manifest["n_frames"] == 3
    assert manifest["n_excluded_frames"] == 1
    assert manifest["n_peaks"] == 6            # 2 per kept frame

    with h5py.File(str(analysis), "r") as src:
        clean = np.asarray(src["background/clean"][:], float)
    with h5py.File(str(manifest["correlations_h5"]), "r") as h5:
        index = np.asarray(h5["frames/index"][:], int)
        frame_row = np.asarray(h5["peaks/frame_row"][:], int)
        original_frame = np.asarray(h5["peaks/original_frame"][:], int)
        centers = np.asarray(h5["peaks/center"][:], float)
        original_positive = np.asarray(
            h5["patterns/original_positive"][:], float
        )
    assert index.tolist() == [0, 2, 3]
    assert 1 not in original_frame.tolist()
    # frame_row indexes the compacted frame axis and round-trips through it.
    assert np.array_equal(original_frame, index[frame_row])
    # Pattern rows are exactly the kept source rows (positive-clipped).
    expected = np.where(
        np.isfinite(clean[index]), np.clip(clean[index], 0.0, None), np.nan
    )
    assert np.allclose(original_positive, expected, equal_nan=True)
    # Each kept frame's peaks sit at that frame's true shifted centers.
    for row, frame in enumerate(index):
        shift = 0.03 * frame
        got = np.sort(centers[frame_row == row])
        assert got == pytest.approx([2.0 + shift, 6.0 - shift])


def test_powder_end_to_end_writes_atomic_schema_and_manifest(tmp_path):
    analysis = _write_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "powder_correlations"
    manifest = run_correlations(
        analysis,
        out,
        sample_type="powder",
        window_width=4.0,
        window_step=3.0,
        location_tolerance=0.25,
        make_plots=False,
    )

    artifact = out / "correlations_powder.h5"
    manifest_path = out / "manifest_powder.json"
    assert artifact.is_file() and manifest_path.is_file()
    assert not list(out.glob("*.tmp*"))
    assert manifest["n_peaks"] == 8
    assert manifest["track_collapsed"] is False
    assert manifest["plots_written"] == 0
    assert manifest["source_requested"] == "fit"

    with h5py.File(str(artifact), "r") as h5:
        assert h5.attrs["sample_type"] == "powder"
        assert h5.attrs["all_peak_policy"].startswith("one anchor")
        assert h5["transform"].attrs["method"] == "log_squared"
        assert h5["patterns/original_positive"].shape == (4, 241)
        assert h5["patterns/log_squared"].shape == (4, 241)
        assert h5["patterns/log_squared_signed"].shape == (4, 241)
        assert np.all(h5["patterns/log_squared"][:, 100] == 0.0)
        assert np.all(h5["patterns/log_squared_signed"][:, 100] > 0.0)
        with h5py.File(str(analysis), "r") as source_h5:
            signed_source = np.asarray(source_h5["background/clean"][:], float)
        _, expected_transform = log_squared_transform(signed_source)
        assert h5["transform"].attrs["scale"] == pytest.approx(
            expected_transform.scale
        )
        assert h5["transform"].attrs["noise_floor"] == pytest.approx(
            expected_transform.noise_floor
        )
        assert h5["anchor_maps/roi_area"].shape == (8, 8)
        assert h5["anchor_maps/location"].shape == (8, 8)
        assert h5["anchor_maps/location"].attrs["intensity_transform"] == "none"
        assert bool(h5.attrs["roi_area_directional"])
        assert "no_recentering" in h5.attrs["roi_area_method"]
        np.testing.assert_allclose(
            h5["peaks/half_width"][:],
            0.75 * h5["peaks/width"][:],
            rtol=0.0,
            atol=0.0,
        )
        peak_frames = np.asarray(h5["peaks/frame_row"][:], int)
        same_frame = peak_frames[:, None] == peak_frames[None, :]
        roi_area = np.asarray(h5["anchor_maps/roi_area"][:], float)
        location = np.asarray(h5["anchor_maps/location"][:], float)
        assert np.all(np.isnan(roi_area[same_frame]))
        assert np.all(np.isnan(location[same_frame]))
        assert np.any(np.isfinite(roi_area[~same_frame]))
        assert np.any(np.isfinite(location[~same_frame]))
        assert h5["windows/across_direct"].shape == (3, 4, 4)
        assert h5["windows/across_acf"].shape == (3, 4, 4)
        assert h5["windows/within_acf"].shape == (4, 3, 3)
        assert h5["windows/acf_features"].shape == (4, 3, 63)
        assert (
            h5["windows"].attrs["acf_method"]
            == "standardized_positive_lag_fft_acf"
        )
        assert (
            h5["windows"].attrs["acf_lag_policy"]
            == "all_positive_lags;zero_lag_excluded"
        )
        assert h5["windows"].attrs["acf_lag_count"] == 63
        assert "provenance" in h5

    review = inspect_correlations(artifact)
    assert review["ok"], review["anomalies"]
    anchor = load_anchor_map(artifact, "roi_area", 0)
    assert anchor["grid"].shape == (4, 2)
    assert anchor["vector"].shape == (8,)
    on_disk_manifest = json.loads(manifest_path.read_text())
    assert on_disk_manifest["transform"]["method"] == "log_squared"
    assert on_disk_manifest["same_frame_policy"] == "NaN"
    assert (
        on_disk_manifest["window_acf_method"]
        == "standardized_positive_lag_fft_acf"
    )
    assert on_disk_manifest["window_acf_lag_count"] == 63


def test_single_crystal_retains_every_observation_not_tracks(tmp_path):
    analysis = _write_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "shared_correlations"
    powder_manifest = run_correlations(
        analysis,
        out,
        sample_type="powder",
        window_width=5.0,
        window_step=5.0,
        make_plots=False,
    )
    manifest = run_correlations(
        analysis,
        out,
        sample_type="single_crystal",
        window_width=5.0,
        window_step=5.0,
        make_plots=False,
    )
    assert manifest["n_peaks"] == 12
    assert manifest["source_requested"] == "spots"
    assert manifest["source_resolved"] == "spots"
    assert Path(powder_manifest["correlations_h5"]).name == "correlations_powder.h5"
    assert (out / "correlations_powder.h5").is_file()
    assert (out / "manifest_powder.json").is_file()
    assert (out / "correlations_single_crystal.h5").is_file()
    assert (out / "manifest_single_crystal.json").is_file()
    with h5py.File(str(out / "correlations_single_crystal.h5"), "r") as h5:
        assert h5["peaks/id"].shape == (12,)
        assert np.unique(h5["peaks/track"][:]).size == 2
        assert h5["anchor_maps/roi_area"].shape == (12, 12)
        features = h5["anchor_maps/roi_feature_log_squared"][:]
        expected = relative_feature_similarity(features[:, None], features[None, :])
        peak_frames = np.asarray(h5["peaks/frame_row"][:], int)
        expected[peak_frames[:, None] == peak_frames[None, :]] = np.nan
        np.testing.assert_allclose(
            h5["anchor_maps/roi_area"][:], expected, equal_nan=True
        )
        assert np.all(
            np.isnan(
                h5["anchor_maps/location"][:][
                    peak_frames[:, None] == peak_frames[None, :]
                ]
            )
        )
        assert not bool(h5.attrs["roi_area_directional"])
        assert (
            h5["anchor_maps/roi_feature_log_squared"].attrs["method"]
            == "one_dimensional_radial_mean_approximation_of_raw_pixel_roi_feature"
        )
        assert bool(h5["peaks"].attrs["all_peak"])
        assert not bool(h5["peaks"].attrs["track_used_for_grouping"])


def test_missing_scientific_prerequisites_are_actionable(tmp_path):
    analysis = _write_analysis(tmp_path / "analysis.h5", include_spots=False)
    with pytest.raises(ValueError, match="spot tracking"):
        run_correlations(
            analysis,
            tmp_path / "bad_single",
            sample_type="single_crystal",
            make_plots=False,
        )

    with h5py.File(str(analysis), "r+") as h5:
        del h5["peaks"]
    with pytest.raises(ValueError, match="Analysis Step 2"):
        run_correlations(
            analysis,
            tmp_path / "bad_powder",
            sample_type="powder",
            make_plots=False,
        )
