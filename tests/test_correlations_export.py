"""CSV export contracts: every plotted number has a spreadsheet form."""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np

from seriesxrd.correlations.batch import main
from seriesxrd.correlations.export import export_matrices, export_summary_csvs
from seriesxrd.correlations.processing import run_correlations

from tests.test_correlations_processing import _write_analysis


def _read_csv(path: Path) -> "list[dict]":
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_summary_csvs_mirror_the_artifact(tmp_path):
    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder", make_plots=False,
    )
    artifact = Path(manifest["correlations_h5"])
    written = export_summary_csvs(artifact, tmp_path / "csv", top_n=3)
    names = {path.name for path in written}
    assert names == {
        "anchors_summary.csv",
        "window_across_long.csv",
        "window_within_long.csv",
        "tracks_summary.csv",
        "track_observations.csv",
        "transition_intervals.csv",
    }

    anchors = _read_csv(tmp_path / "csv" / "anchors_summary.csv")
    assert len(anchors) == manifest["n_peaks"]
    with h5py.File(str(artifact), "r") as h5:
        roi = np.asarray(h5["anchor_maps/roi_area"][:], float)
        n_frames = int(h5.attrs["n_frames"])
        n_windows = int(h5.attrs["n_windows"])
    # The reported best match is exactly the matrix row's best finite score.
    first = anchors[0]
    best = float(first["match1_score"])
    row = roi[int(first["id"])]
    assert best == np.nanmax(row)
    assert int(first["match1_anchor"]) == int(np.nanargmax(row))

    across = _read_csv(tmp_path / "csv" / "window_across_long.csv")
    assert len(across) == n_windows * n_frames * (n_frames - 1) // 2
    within = _read_csv(tmp_path / "csv" / "window_within_long.csv")
    assert len(within) == n_frames * n_windows * (n_windows - 1) // 2

    tracks = _read_csv(tmp_path / "csv" / "tracks_summary.csv")
    assert len(tracks) == manifest["tracks"]["n_tracks"]
    observations = _read_csv(tmp_path / "csv" / "track_observations.csv")
    assert len(observations) == manifest["tracks"]["n_track_obs"]
    intervals = _read_csv(tmp_path / "csv" / "transition_intervals.csv")
    assert len(intervals) == n_frames - 1
    assert set(intervals[0]) >= {
        "births", "deaths", "n_active", "transition_candidate",
    }


def test_matrix_dump_round_trips(tmp_path):
    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder",
        make_plots=False, make_tracks=False,
    )
    written = export_matrices(manifest["correlations_h5"], tmp_path / "csv")
    assert {path.name for path in written} == {"roi_area.csv", "location.csv"}
    rows = _read_csv(tmp_path / "csv" / "roi_area.csv")
    with h5py.File(manifest["correlations_h5"], "r") as h5:
        roi = np.asarray(h5["anchor_maps/roi_area"][:], float)
    assert len(rows) == roi.shape[0]
    probe = roi[0]
    finite = np.nonzero(np.isfinite(probe))[0]
    target = int(finite[0])
    assert float(rows[0][f"t{target:04d}"]) == probe[target]
    # NaN cells export as empty, never as the text "nan".
    nan_target = int(np.nonzero(~np.isfinite(probe))[0][0])
    assert rows[0][f"t{nan_target:04d}"] == ""


def test_cli_export_flag_lists_files_in_manifest(tmp_path):
    import json

    analysis = _write_analysis(tmp_path / "analysis.h5")
    rc = main(
        [
            str(analysis), "--out", str(tmp_path / "res"),
            "--sample-type", "powder", "--no-plots", "--export-csv",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "res" / "manifest_powder.json").read_text()
    )
    assert manifest["csv_files"]
    for relative in manifest["csv_files"]:
        assert (tmp_path / "res" / relative).is_file()
