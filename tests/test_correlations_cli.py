"""CLI and no-display rendering tests for the Correlations stage."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from seriesxrd.correlations.batch import main


def _write_minimal_analysis(path: Path) -> Path:
    radial = np.linspace(0.0, 6.0, 121)
    patterns = []
    centers = []
    for frame in range(3):
        center = 2.0 + 0.05 * frame
        centers.append(center)
        patterns.append(
            0.2
            + (6.0 + frame)
            * np.exp(-0.5 * ((radial - center) / 0.12) ** 2)
        )
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "2th_deg"
        h5.create_dataset("radial", data=radial)
        bg = h5.create_group("background")
        bg.create_dataset("clean", data=np.asarray(patterns))
        bg.create_dataset("baseline", data=np.zeros((3, radial.size)))
        bg.create_dataset("spot_residual", data=np.zeros((3, radial.size)))
        frames = h5.create_group("frames")
        frames.create_dataset(
            "filename",
            data=np.asarray(["a.tif", "b.tif", "c.tif"], object),
            dtype=h5py.string_dtype("utf-8"),
        )
        frames.create_dataset("pressure", data=np.asarray([1.0, 3.0, 5.0]))
        frames.create_dataset("excluded", data=np.zeros(3, bool))
        peaks = h5.create_group("peaks")
        peaks.attrs["source"] = "clean"
        peaks.create_dataset("frame", data=np.arange(3, dtype="i4"))
        peaks.create_dataset("center", data=np.asarray(centers))
        peaks.create_dataset("fwhm", data=np.full(3, 0.24))
        peaks.create_dataset("area", data=np.asarray([6.0, 7.0, 8.0]))
        peaks.create_dataset("flag", data=np.zeros(3, "i4"))
    return path


def test_cli_no_plots_really_writes_and_opens_h5(tmp_path):
    analysis = _write_minimal_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "results"
    rc = main(
        [
            str(analysis),
            "--out",
            str(out),
            "--sample-type",
            "powder",
            "--source",
            "fit",
            "--window-width",
            "3",
            "--window-step",
            "2",
            "--location-tolerance",
            "0.2",
            "--no-plots",
        ]
    )
    assert rc == 0
    with h5py.File(str(out / "correlations_powder.h5"), "r") as h5:
        assert h5["anchor_maps/roi_area"].shape == (3, 3)
        assert h5["windows/across_direct"].shape[1:] == (3, 3)
        assert h5.attrs["source_resolved"] == "fit:clean"
    manifest = json.loads((out / "manifest_powder.json").read_text())
    assert manifest["plots_written"] == 0
    assert not any(".tmp" in path.name for path in out.iterdir())


def test_cli_streams_progress_protocol(tmp_path, capsys):
    """The run emits house-style 3-token '[CORRELATIONS] done total' lines."""
    analysis = _write_minimal_analysis(tmp_path / "analysis.h5")
    rc = main(
        [
            str(analysis),
            "--out",
            str(tmp_path / "out"),
            "--sample-type",
            "powder",
            "--window-width",
            "3",
            "--window-step",
            "2",
            "--location-tolerance",
            "0.2",
            "--plots",
            "all",
            "--max-anchor-plots",
            "1",
        ]
    )
    assert rc == 0
    progress = []
    for line in capsys.readouterr().out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "[CORRELATIONS]":
            try:
                progress.append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    assert progress, "no 3-token progress lines were streamed"
    done, total = progress[-1]
    assert done == total > 0
    manifest = json.loads(
        (tmp_path / "out" / "manifest_powder.json").read_text()
    )
    assert manifest["anchor_plot_cap"] == 1
    assert len(
        [f for f in manifest["plot_files"] if "anchor_" in f]
    ) == 3


def test_bulk_rendering_is_off_by_default_and_selectable(tmp_path, capsys):
    """A plain run writes no PNGs and says so; families are selectable."""
    from seriesxrd.correlations.processing import resolve_plot_families
    from seriesxrd.correlations.plots import FAMILIES

    analysis = _write_minimal_analysis(tmp_path / "analysis.h5")
    rc = main(
        [
            str(analysis), "--out", str(tmp_path / "plain"),
            "--sample-type", "powder", "--window-width", "3",
            "--window-step", "3",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "plain" / "manifest_powder.json").read_text()
    )
    assert manifest["plots"] == [] and manifest["plots_written"] == 0
    assert not (tmp_path / "plain" / "heatmaps").exists()
    assert "no PNGs written (default)" in capsys.readouterr().out

    # One family only.
    rc = main(
        [
            str(analysis), "--out", str(tmp_path / "some"),
            "--sample-type", "powder", "--window-width", "3",
            "--window-step", "3", "--plots", "waterfall",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "some" / "manifest_powder.json").read_text()
    )
    assert manifest["plots"] == ["waterfall"]
    assert manifest["plots_written"] > 0
    assert all("waterfall/" in f for f in manifest["plot_files"])

    # Resolution rules, including the deprecated boolean spelling.
    assert resolve_plot_families(None, None) == ()
    assert resolve_plot_families(None, False) == ()
    assert resolve_plot_families(None, True) == tuple(FAMILIES)
    assert resolve_plot_families(("all",)) == tuple(FAMILIES)
    assert resolve_plot_families(("none", "tracks")) == ()
    # De-duplicated into canonical render order.
    assert resolve_plot_families(("tracks", "roi_area", "tracks")) == (
        "roi_area", "tracks",
    )


def test_cli_missing_input_returns_one(tmp_path):
    rc = main(
        [
            str(tmp_path / "missing.h5"),
            "--out",
            str(tmp_path / "out"),
            "--sample-type",
            "powder",
            "--no-plots",
        ]
    )
    assert rc == 1


def test_module_cli_renders_with_no_display(tmp_path):
    analysis = _write_minimal_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "rendered"
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "seriesxrd.correlations.batch",
            str(analysis),
            "--out",
            str(out),
            "--sample-type",
            "powder",
            "--window-width",
            "3",
            "--window-step",
            "3",
            "--plots",
            "all",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    manifest = json.loads((out / "manifest_powder.json").read_text())
    assert manifest["plots_written"] > 0
    pngs = sorted((out / "heatmaps" / "powder").rglob("*.png"))
    assert pngs
    with Image.open(pngs[0]) as image:
        image.verify()
    pressure_roi = list((out / "heatmaps" / "powder" / "roi_area").glob("pressure_*"))
    pressure_waterfall = list(
        (out / "heatmaps" / "powder" / "waterfall").glob("pressure_*")
    )
    assert pressure_roi and pressure_waterfall

    # A rerender replaces only the managed powder tree, removing stale files
    # while preserving the sibling sample type.
    stale = out / "heatmaps" / "powder" / "roi_area" / "pressure_stale" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old result")
    sibling = out / "heatmaps" / "single_crystal" / "keep.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("unrelated sample type")
    from seriesxrd.correlations.plots import render_all

    current = render_all(out / "correlations_powder.h5", out / "heatmaps")
    assert not stale.exists()
    assert sibling.read_text() == "unrelated sample type"
    assert current and all("stale" not in item for item in current)
