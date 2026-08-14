"""Checkpoint, recovery, and resumable-export contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from seriesxrd.correlations import checkpoint
from seriesxrd.correlations.plots import render_all
from seriesxrd.correlations.processing import run_correlations

from tests.test_correlations_processing import _write_analysis


def _artifact(tmp_path: Path) -> Path:
    analysis = _write_analysis(tmp_path / "analysis.h5")
    manifest = run_correlations(
        analysis, tmp_path / "res", sample_type="powder",
    )
    return Path(manifest["correlations_h5"])


def test_interrupted_render_keeps_its_staging_tree(tmp_path, monkeypatch):
    """A render that dies leaves its finished figures behind, marked."""
    artifact = _artifact(tmp_path)
    root = tmp_path / "res" / "heatmaps"

    real_save = __import__(
        "seriesxrd.correlations.plots", fromlist=["save_figure"]
    ).save_figure
    calls = {"n": 0}

    def exploding_save(fig, path):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("worker died")
        real_save(fig, path)

    monkeypatch.setattr(
        "seriesxrd.correlations.plots.save_figure", exploding_save
    )
    with pytest.raises(RuntimeError, match="worker died"):
        render_all(artifact, root)

    recoverable = checkpoint.find_recoverable(tmp_path / "res")
    assert len(recoverable) == 1
    record = recoverable[0]
    assert record["sample_type"] == "powder"
    assert record["n_figures"] == 5           # the ones that made it
    assert record["planned"] > 5
    assert record["status"] == "interrupted"
    # The half-finished tree was NOT published.
    assert not (root / "powder").exists()


def test_resume_skips_finished_figures_and_completes(tmp_path, monkeypatch):
    """Resuming re-renders only what is missing, then publishes in full."""
    artifact = _artifact(tmp_path)
    root = tmp_path / "res" / "heatmaps"
    import seriesxrd.correlations.plots as plots_module

    real_save = plots_module.save_figure
    calls = {"n": 0}

    def exploding_save(fig, path):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("worker died")
        real_save(fig, path)

    monkeypatch.setattr(plots_module, "save_figure", exploding_save)
    with pytest.raises(RuntimeError):
        render_all(artifact, root)
    monkeypatch.setattr(plots_module, "save_figure", real_save)

    built = {"n": 0}
    real_build = plots_module.build_figure

    def counting_build(ctx, spec):
        built["n"] += 1
        return real_build(ctx, spec)

    monkeypatch.setattr(plots_module, "build_figure", counting_build)
    files = render_all(artifact, root, resume=True)

    total = len(files)
    assert total > 5
    # Only the unfinished remainder was rebuilt.
    assert built["n"] == total - 5
    assert (root / "powder").is_dir()
    assert len(list((root / "powder").rglob("*.png"))) == total
    # The state file does not survive into the published tree.
    assert not (root / "powder" / checkpoint.STATE_FILENAME).exists()
    assert not checkpoint.find_recoverable(tmp_path / "res")


def test_resume_refuses_an_incompatible_checkpoint(tmp_path, monkeypatch):
    """A checkpoint for different settings is not adopted."""
    artifact = _artifact(tmp_path)
    root = tmp_path / "res" / "heatmaps"
    import seriesxrd.correlations.plots as plots_module

    real_save = plots_module.save_figure
    calls = {"n": 0}

    def exploding_save(fig, path):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("stop")
        real_save(fig, path)

    monkeypatch.setattr(plots_module, "save_figure", exploding_save)
    with pytest.raises(RuntimeError):
        render_all(artifact, root, families=("waterfall",))
    monkeypatch.setattr(plots_module, "save_figure", real_save)

    stale = checkpoint.find_staging(tmp_path / "res", "powder")
    assert len(stale) == 1

    # Different family selection: must start a fresh staging directory.
    render_all(artifact, root, families=("tracks",))
    published = list((root / "powder").rglob("*.png"))
    assert published and all("tracks" in str(p) for p in published)


def test_live_render_is_not_offered_for_recovery(tmp_path):
    """A staging tree whose writer is still alive is left alone."""
    staging = tmp_path / "res" / "heatmaps" / ".powder.tmp-live"
    staging.mkdir(parents=True)
    (staging / "a.png").write_bytes(b"x")
    checkpoint.write_state(staging, status="running", sample_type="powder")
    assert checkpoint.is_live(staging)          # our own pid
    assert not checkpoint.find_recoverable(tmp_path / "res")

    # Another process that has stopped heart-beating is recoverable.
    checkpoint.write_state(staging, status="running", pid=os.getpid() + 90000)
    old = 1.0
    os.utime(staging / "a.png", (old, old))
    os.utime(staging / checkpoint.STATE_FILENAME, (old, old))
    os.utime(staging, (old, old))
    assert not checkpoint.is_live(staging)
    assert len(checkpoint.find_recoverable(tmp_path / "res")) == 1


def test_promote_publishes_and_records_incompleteness(tmp_path):
    """Promotion swaps the tree in, marks it, and spares the sibling."""
    root = tmp_path / "res" / "heatmaps"
    staging = root / ".powder.tmp-abandoned"
    (staging / "roi_area").mkdir(parents=True)
    (staging / "roi_area" / "anchor_0000.png").write_bytes(b"png")
    checkpoint.write_state(
        staging, status="interrupted", sample_type="powder", planned=40
    )
    sibling = root / "single_crystal"
    sibling.mkdir(parents=True)
    (sibling / "keep.txt").write_text("untouched")

    destination = root / "powder"
    checkpoint.publish_staging(staging, destination)
    checkpoint.mark_incomplete(
        destination, n_figures=1, planned=40, source=staging.name,
    )

    assert not staging.exists()
    assert (destination / "roi_area" / "anchor_0000.png").is_file()
    assert (sibling / "keep.txt").read_text() == "untouched"
    marker = json.loads(
        (destination / checkpoint.INCOMPLETE_FILENAME).read_text()
    )
    assert marker["incomplete"] is True
    assert marker["n_figures"] == 1 and marker["planned"] == 40
    assert "interrupted" in marker["reason"]


def test_sigterm_stops_gracefully_and_resume_finishes(tmp_path):
    """SIGTERM stops at a figure boundary with exit 130, keeps what is done,
    and --resume completes the export without recomputing."""
    import signal
    import subprocess
    import sys
    import time

    from seriesxrd.core.processes import worker_popen

    analysis = _write_analysis(tmp_path / "analysis.h5")
    out = tmp_path / "res"
    argv = [
        sys.executable, "-m", "seriesxrd.correlations.batch", str(analysis),
        "--out", str(out), "--sample-type", "powder", "--plots", "all",
    ]
    env = dict(os.environ, MPLBACKEND="Agg")
    # Launch exactly as the GUI does: worker_popen puts the child in its own
    # process group on Windows, which is what makes CTRL_BREAK_EVENT
    # deliverable to it and to nothing else.
    proc = worker_popen(
        argv, cwd=str(Path(__file__).resolve().parents[1]), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Stop it once figures have started landing.
    staged = None
    deadline = time.time() + 90
    while time.time() < deadline:
        found = checkpoint.find_staging(out, "powder")
        if found and list(found[0].rglob("*.png")):
            staged = found[0]
            break
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    if staged is None:
        proc.kill()
        proc.wait(timeout=30)
        pytest.skip("render finished before it could be interrupted")

    # Ask for a graceful stop the way each platform can actually deliver
    # one. On Windows SIGTERM is not a signal at all: Popen.send_signal
    # maps it to TerminateProcess, which runs no handler and exits 1.
    if os.name == "nt":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGTERM)
    output = proc.communicate(timeout=60)[0]
    assert proc.returncode == 130, output
    assert "--resume" in output

    recoverable = checkpoint.find_recoverable(out)
    assert recoverable and recoverable[0]["status"] == "interrupted"
    partial = recoverable[0]["n_figures"]
    assert partial > 0
    assert not (out / "heatmaps" / "powder").exists()

    artifact_mtime = (out / "correlations_powder.h5").stat().st_mtime_ns
    finished = subprocess.run(
        argv + ["--resume"], cwd=str(Path(__file__).resolve().parents[1]),
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "exporting only" in finished.stdout
    # The numbers were not recomputed, and the tree is now complete.
    assert (
        out / "correlations_powder.h5"
    ).stat().st_mtime_ns == artifact_mtime
    published = list((out / "heatmaps" / "powder").rglob("*.png"))
    assert len(published) > partial
    assert not checkpoint.find_recoverable(out)


def test_stray_tmp_files_are_swept_on_resume(tmp_path):
    """A SIGKILL between savefig and rename cannot poison a resume."""
    artifact = _artifact(tmp_path)
    root = tmp_path / "res" / "heatmaps"
    render_all(artifact, root, families=("tracks",))
    staging = root / "powder"
    stray = staging / "tracks" / "tracks.png.tmp"
    stray.write_bytes(b"partial")

    from seriesxrd.correlations.plots import _render_into

    _render_into(artifact, staging, None, ("tracks",), resume=True)
    assert not stray.exists()
