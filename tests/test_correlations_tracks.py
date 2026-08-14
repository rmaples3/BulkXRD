"""ROI-gated track linking + exploratory transition screening contracts."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from seriesxrd.correlations.batch import main
from seriesxrd.correlations.processing import run_correlations
from seriesxrd.correlations.review import inspect_correlations
from seriesxrd.correlations.tracks import mutual_roi_similarity

from tests.test_correlations_processing import _write_analysis


def test_mutual_roi_similarity_contract():
    matrix = np.asarray([[np.nan, 0.5, 0.0], [0.2, np.nan, 1.0],
                         [0.0, 0.4, np.nan]])
    mutual = mutual_roi_similarity(matrix)
    assert mutual[0, 1] == pytest.approx(np.sqrt(0.5 * 0.2))
    assert mutual[1, 0] == pytest.approx(mutual[0, 1])
    assert mutual[0, 2] == 0.0
    assert np.isnan(mutual[0, 0])
    with pytest.raises(ValueError, match="square"):
        mutual_roi_similarity(np.ones((2, 3)))


def test_end_to_end_tracks_two_drifting_peaks(tmp_path):
    """The fixture's two drifting reflections become exactly two 4-frame
    tracks, deterministically, with finite gated edge similarities, and the
    track overview PNG is rendered."""
    analysis = _write_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "res"
    manifest = run_correlations(
        analysis, out, sample_type="powder", plots=("all",),
        max_anchor_plots=1,
    )
    assert manifest["tracks"]["n_tracks"] == 2
    assert manifest["tracks"]["n_track_obs"] == 8
    assert manifest["tracks"]["exploratory"] is True
    # Manifest path lists are POSIX-normalized so the artifact reads the same
    # on every platform (a native-separator list failed only on Windows).
    assert all("\\" not in entry for entry in manifest["plot_files"])
    assert any(f.endswith("tracks/tracks.png") for f in manifest["plot_files"])
    assert (out / "heatmaps" / "powder" / "tracks" / "tracks.png").is_file()

    with h5py.File(manifest["correlations_h5"], "r") as h5:
        tracks = h5["tracks"]
        assert tracks.attrs["linker"] == (
            "seriesxrd.analysis.unknowns.link_tracks"
        )
        assert bool(tracks.attrs["exploratory"]) is True
        obs_track = np.asarray(tracks["obs/track"][:], int)
        obs_peak = np.asarray(tracks["obs/peak_id"][:], int)
        n_obs = np.asarray(tracks["summary/n_obs"][:], int)
        similarity = np.asarray(tracks["edges/similarity"][:], float)
        centers = np.asarray(h5["peaks/center"][:], float)
        intervals = np.asarray(
            tracks["intervals/frame_row_from"][:], int
        )
    assert n_obs.tolist() == [4, 4]
    assert similarity.shape == (6,)
    assert np.all(np.isfinite(similarity)) and np.all(similarity >= 0.2)
    # Each track follows ONE physical reflection, never mixing the two.
    for track_id in (0, 1):
        member_centers = centers[obs_peak[obs_track == track_id]]
        assert np.ptp(member_centers) < 0.2
    assert intervals.tolist() == [0, 1, 2]

    # Determinism: an identical rerun writes identical track arrays.
    rerun = run_correlations(
        analysis, tmp_path / "res2", sample_type="powder", make_plots=False,
    )
    with h5py.File(manifest["correlations_h5"], "r") as first, h5py.File(
        rerun["correlations_h5"], "r"
    ) as second:
        for name in (
            "obs/track", "obs/peak_id", "summary/id", "summary/n_obs",
            "edges/similarity", "intervals/transition_candidate",
        ):
            np.testing.assert_array_equal(
                first[f"tracks/{name}"][:], second[f"tracks/{name}"][:]
            )


def test_births_flag_exploratory_transition_interval(tmp_path):
    """Two reflections appearing together at frame 2 flag the (1, 2) interval
    through the births+deaths clause, and only that interval."""
    analysis = _write_analysis(
        tmp_path / "analysis.h5",
        extra_peaks=(
            (2, 4.5, 0.3, 6.0), (3, 4.5, 0.3, 6.0),
            (2, 5.2, 0.3, 6.0), (3, 5.2, 0.3, 6.0),
        ),
    )
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder",
        make_plots=False, track_min_frames=2,
    )
    assert manifest["tracks"]["n_tracks"] == 4
    with h5py.File(manifest["correlations_h5"], "r") as h5:
        births = np.asarray(h5["tracks/intervals/births"][:], int)
        deaths = np.asarray(h5["tracks/intervals/deaths"][:], int)
        active = np.asarray(h5["tracks/intervals/n_active"][:], int)
        candidate = np.asarray(
            h5["tracks/intervals/transition_candidate"][:], bool
        )
    assert births.tolist() == [0, 2, 0]
    assert deaths.tolist() == [0, 0, 0]
    events = births + deaths
    threshold = np.maximum(2, np.ceil(0.25 * active).astype(int))
    assert (events >= threshold).tolist() == [False, True, False]
    assert bool(candidate[1])


def test_no_tracks_and_impossible_gate(tmp_path):
    """--no-tracks omits /tracks entirely; a gate nothing can pass yields
    zero tracks but a complete, reviewable artifact either way."""
    analysis = _write_analysis(tmp_path / "analysis.h5")
    rc = main(
        [
            str(analysis), "--out", str(tmp_path / "none"),
            "--sample-type", "powder", "--no-plots", "--no-tracks",
        ]
    )
    assert rc == 0
    with h5py.File(str(tmp_path / "none" / "correlations_powder.h5"), "r") as h5:
        assert "tracks" not in h5
    summary = inspect_correlations(
        tmp_path / "none" / "correlations_powder.h5"
    )
    assert summary["ok"] and "n_tracks" not in summary

    gated = run_correlations(
        analysis, tmp_path / "gated", sample_type="powder",
        make_plots=False, track_min_similarity=1.01,
    )
    assert gated["tracks"]["n_tracks"] == 0
    assert gated["tracks"]["n_transition_candidates"] == 0
    with h5py.File(gated["correlations_h5"], "r") as h5:
        assert h5["tracks/obs/track"].shape == (0,)
        assert h5["tracks/summary/id"].shape == (0,)
    summary = inspect_correlations(gated["correlations_h5"])
    assert summary["ok"] and summary["n_tracks"] == 0
