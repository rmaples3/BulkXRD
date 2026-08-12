"""Headless tests for correlation-stage session seeding."""
from __future__ import annotations

import json
from pathlib import Path

from seriesxrd.correlations.session import (
    CONFIG_FILENAME,
    correlation_config_path,
    seed_correlation_config,
)
from seriesxrd.correlations.batch import build_parser
from seriesxrd.correlations.gui import (
    RESULT_FILTER_ALL,
    CorrelationApp,
    _classify_result_path,
    _find_result_paths,
    _load_result_pressures,
    _pressure_label,
    _result_matches,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_correlation_config_path_is_workspace_local(tmp_path):
    workspace = tmp_path / "workspace"
    assert correlation_config_path(workspace) == (
        workspace.resolve() / CONFIG_FILENAME
    )


def test_seed_discovers_completed_analysis_h5(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    analysis_h5 = workspace / "run_analysis.h5"
    analysis_h5.write_bytes(b"test fixture")
    _write_json(
        workspace / "analysis_session_config.json",
        {"analysis_h5_file": str(analysis_h5)},
    )

    path = seed_correlation_config(workspace)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert path == workspace / CONFIG_FILENAME
    assert config["analysis_h5_file"] == str(analysis_h5)
    assert config["workspace_root"] == str(workspace.resolve())
    assert config["result_root"] == str(workspace.resolve() / "correlations")
    assert config["transform"] == "log_squared"
    assert config["source"] == "fit"
    assert config["window_width"] == "5.0"
    assert config["window_step"] == "1.0"
    assert config["location_tolerance"] == "0.02"
    assert config["session_config_path"] == str(path)


def test_seed_does_not_copy_missing_analysis_output(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_json(
        workspace / "analysis_session_config.json",
        {"analysis_h5_file": str(workspace / "not-created-yet.h5")},
    )

    path = seed_correlation_config(workspace)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["analysis_h5_file"] == ""


def test_seed_preserves_existing_user_values(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    upstream = workspace / "upstream_analysis.h5"
    upstream.write_bytes(b"upstream")
    user_analysis = workspace / "chosen_analysis.h5"
    user_analysis.write_bytes(b"chosen")
    _write_json(
        workspace / "analysis_session_config.json",
        {"analysis_h5_file": str(upstream)},
    )
    _write_json(
        workspace / CONFIG_FILENAME,
        {
            "analysis_h5_file": str(user_analysis),
            "result_root": str(workspace / "custom-results"),
            "sample_type": "single_crystal",
            "window_width": "2.5",
            "future_option": {"keep": True},
        },
    )

    path = seed_correlation_config(workspace)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["analysis_h5_file"] == str(user_analysis)
    assert config["result_root"] == str(workspace / "custom-results")
    assert config["sample_type"] == "single_crystal"
    assert config["window_width"] == "2.5"
    assert config["future_option"] == {"keep": True}
    assert config["transform"] == "log_squared"


def test_seed_defaults_single_crystal_to_spots_source(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_json(
        workspace / CONFIG_FILENAME,
        {"sample_type": "single_crystal"},
    )

    path = seed_correlation_config(workspace)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["source"] == "spots"


def test_headless_gui_command_matches_supported_batch_contract(tmp_path):
    analysis_h5 = tmp_path / "analysis.h5"
    analysis_h5.touch()
    controller = CorrelationApp.__new__(CorrelationApp)
    controller.vars = {}
    controller.config = {
        "analysis_h5_file": str(analysis_h5),
        "result_root": str(tmp_path / "results"),
        "sample_type": "single_crystal",
        "source": "spots",
        "radial_min": "1.0",
        "radial_max": "8.0",
        "window_width": "2.0",
        "window_step": "0.5",
        "location_tolerance": "0.06",
    }

    command, result_root, run_dir, sample_type = (
        controller._build_batch_command()
    )
    parsed = build_parser().parse_args(command[3:])

    assert result_root == (tmp_path / "results")
    assert sample_type == "single_crystal"
    # The -m launch resolves the package because run_dir holds the checkout.
    assert (run_dir / "seriesxrd" / "correlations" / "batch.py").is_file()
    assert parsed.analysis == str(analysis_h5.resolve())
    assert parsed.out == str(result_root)
    assert parsed.sample_type == "single_crystal"
    assert parsed.source == "spots"
    assert parsed.location_tol == 0.06
    # New knobs ride along with their defaults.
    assert parsed.order_by == "frame"
    assert parsed.scale_quantile == 0.995
    assert parsed.max_anchor_plots is None
    assert not parsed.no_plots
    assert not parsed.no_tracks
    assert parsed.track_min_similarity == 0.2
    assert parsed.track_group_by == "none"


def test_headless_gui_command_honors_configured_interpreter(tmp_path):
    analysis_h5 = tmp_path / "analysis.h5"
    analysis_h5.touch()
    controller = CorrelationApp.__new__(CorrelationApp)
    controller.vars = {}
    controller.config = {
        "analysis_h5_file": str(analysis_h5),
        "result_root": str(tmp_path / "results"),
        "sample_type": "powder",
        "source": "fit",
        "window_width": "5.0",
        "window_step": "1.0",
        "location_tolerance": "0.02",
        "python_exe": "/opt/custom/python3",
        "skip_plots": True,
        "max_anchor_plots": "12",
        "order_by": "pressure",
        "make_tracks": False,
    }

    command, _root, _run_dir, _sample = controller._build_batch_command()
    assert command[0] == "/opt/custom/python3"
    parsed = build_parser().parse_args(command[3:])
    assert parsed.no_plots and parsed.no_tracks
    assert parsed.max_anchor_plots == 12
    assert parsed.order_by == "pressure"


def test_gui_sample_switch_applies_safe_profile_source_default():
    class _Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    controller = CorrelationApp.__new__(CorrelationApp)
    controller.vars = {
        "sample_type": _Var("single_crystal"),
        "source": _Var("fit"),
    }
    controller.save_config = lambda silent=False: None
    controller._update_input_status = lambda: None

    controller._sample_type_changed()
    assert controller.vars["source"].get() == "spots"

    controller.vars["sample_type"].set("powder")
    controller._sample_type_changed()
    assert controller.vars["source"].get() == "fit"


def test_successful_worker_reviews_the_result_root_snapshotted_at_launch(tmp_path):
    class _Widget:
        def configure(self, **_kwargs):
            pass

    controller = CorrelationApp.__new__(CorrelationApp)
    launched_root = tmp_path / "launched-results"
    controller._active_result_root = launched_root
    controller._run_proc = object()
    controller._cancel_requested = False
    controller.run_button = _Widget()
    controller.cancel_button = _Widget()
    controller.run_status = _Widget()
    controller._status_bar = _Widget()
    controller.log = lambda _message: None
    reviewed = []
    controller.review_results = lambda **kwargs: reviewed.append(kwargs)

    controller._handle_worker_event(("done", 0))

    assert controller._active_result_root is None
    assert controller._run_proc is None
    assert reviewed == [{"show_errors": False, "result_root": launched_root}]


def test_successful_worker_reads_manifest_and_records_artifact(tmp_path):
    class _Widget:
        def __init__(self):
            self.options = {}

        def configure(self, **kwargs):
            self.options.update(kwargs)

    launched_root = tmp_path / "results"
    launched_root.mkdir()
    manifest = {
        "correlations_h5": str(launched_root / "correlations_powder.h5"),
        "n_frames": 4,
        "n_peaks": 8,
        "n_anchors_valid": 7,
        "n_windows": 6,
        "plots_written": 30,
        "tracks": {"n_tracks": 2, "n_transition_candidates": 1},
    }
    _write_json(launched_root / "manifest_powder.json", manifest)

    controller = CorrelationApp.__new__(CorrelationApp)
    controller._active_result_root = launched_root
    controller._active_sample_type = "powder"
    controller._run_started = 0.0
    controller._run_proc = object()
    controller._cancel_requested = False
    controller._setting_widgets = []
    controller.config = {}
    controller.save_config = lambda silent=False: None
    controller.run_button = _Widget()
    controller.cancel_button = _Widget()
    controller.run_status = _Widget()
    controller._status_bar = _Widget()
    logged = []
    controller.log = lambda message, level="INFO": logged.append(message)
    controller.review_results = lambda **kwargs: None

    controller._handle_worker_event(("done", 0))

    assert controller.config["correlations_h5"] == manifest["correlations_h5"]
    summary = controller.run_status.options["text"]
    assert "4 frames" in summary
    assert "7/8 anchors" in summary
    assert "2 tracks" in summary
    assert "30 PNGs" in summary


def test_drain_queues_survives_a_failing_handler():
    """One raising event handler must not kill the poll loop."""
    import queue as queue_module

    controller = CorrelationApp.__new__(CorrelationApp)
    controller._closing = False
    controller._poll_after_id = None
    controller._log_queue = queue_module.Queue()
    controller._event_queue = queue_module.Queue()
    handled = []
    rearmed = []
    controller._schedule_queue_poll = lambda: rearmed.append(True)
    controller._insert_log_line = lambda line: handled.append(line)

    def _boom(event):
        if event[1] == "explode":
            raise RuntimeError("widget destroyed")
        handled.append(event)

    controller._handle_worker_event = _boom
    controller.log = lambda message, level="INFO": None
    controller._event_queue.put(("done", "explode"))
    controller._event_queue.put(("done", 0))
    controller._log_queue.put("a line")

    controller._drain_queues()

    assert ("done", 0) in handled       # the later event still processed
    assert "a line" in handled
    assert rearmed == [True]            # the poll loop re-armed exactly once


def test_set_run_ui_state_locks_and_restores_settings():
    class _Widget:
        def __init__(self):
            self.state = None

        def configure(self, state=None, **_kwargs):
            self.state = state

    controller = CorrelationApp.__new__(CorrelationApp)
    controller.run_button = _Widget()
    controller.cancel_button = _Widget()
    entry, combo = _Widget(), _Widget()
    controller._setting_widgets = [(entry, "normal"), (combo, "readonly")]

    controller._set_run_ui_state(True)
    assert controller.run_button.state == "disabled"
    assert controller.cancel_button.state == "normal"
    assert entry.state == "disabled" and combo.state == "disabled"

    controller._set_run_ui_state(False)
    assert controller.run_button.state == "normal"
    assert controller.cancel_button.state == "disabled"
    assert entry.state == "normal" and combo.state == "readonly"


def test_write_failure_log_persists_recent_lines(tmp_path):
    controller = CorrelationApp.__new__(CorrelationApp)
    controller.config = {"logs_root": str(tmp_path / "logs")}
    controller._log_history = [f"line {index}" for index in range(600)]

    target = controller._write_failure_log()
    assert target is not None and target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "line 599" in text
    assert "line 99" not in text        # only the recent tail is kept

    controller.config = {}
    assert controller._write_failure_log() is None


def test_missing_selected_png_clears_the_previous_preview(tmp_path):
    class _Tree:
        def selection(self):
            return ("leaf",)

    class _Widget:
        def __init__(self):
            self.options = {}

        def configure(self, **kwargs):
            self.options.update(kwargs)

    missing = tmp_path / "removed.png"
    controller = CorrelationApp.__new__(CorrelationApp)
    controller._preview_after_id = "scheduled"
    controller.results_tree = _Tree()
    controller._result_paths = {"leaf": missing}
    controller._preview_photo = object()
    controller.preview_label = _Widget()
    controller.preview_path_label = _Widget()

    controller._preview_selected()

    assert controller._preview_after_id is None
    assert controller._preview_photo is None
    assert "no longer exists" in controller.preview_label.options["text"]
    assert controller.preview_path_label.options["text"] == str(missing)


def test_result_pressure_folder_display_and_sort_value():
    assert _pressure_label("pressure_3p5_GPa") == ("3.5 GPa", 3.5)
    assert _pressure_label("pressure_m1p25_GPa") == ("-1.25 GPa", -1.25)
    label, sort_value = _pressure_label("pressure_unknown")
    assert label == "Pressure unavailable"
    assert sort_value == float("inf")


def test_result_paths_are_classified_by_diagram_and_pressure(tmp_path):
    root = tmp_path / "results"
    roi = (
        root / "heatmaps" / "powder" / "roi_area"
        / "pressure_3p5_GPa" / "anchor_0001.png"
    )
    across = (
        root / "heatmaps" / "powder" / "window_across"
        / "direct" / "window_000_2.017_7.017.png"
    )
    within = (
        root / "heatmaps" / "single_crystal" / "window_within"
        / "acf" / "frame_0012.png"
    )

    roi_entry = _classify_result_path(roi, root)
    assert roi_entry["sample_label"] == "Powder"
    assert roi_entry["category_label"] == "ROI area"
    assert roi_entry["pressure_label"] == "3.5 GPa"
    assert roi_entry["leaf_label"] == "Anchor 0001"

    across_entry = _classify_result_path(across, root)
    assert across_entry["category_label"] == "Window across frames"
    assert across_entry["pressure_label"] == "All pressures"
    assert across_entry["method_label"] == "DIRECT"
    assert across_entry["leaf_label"] == "Window 000 — 2.017–7.017"

    within_entry = _classify_result_path(
        within, root, {("single_crystal", 12): 7.6},
    )
    assert within_entry["category_label"] == "Window within frame"
    assert within_entry["pressure_label"] == "7.6 GPa"
    assert within_entry["method_label"] == "ACF"
    assert within_entry["leaf_label"] == "Frame 0012"


def test_result_search_matches_multiple_terms_and_category(tmp_path):
    root = tmp_path / "results"
    path = (
        root / "heatmaps" / "powder" / "waterfall"
        / "pressure_9p1_GPa" / "anchor_0028.png"
    )
    entry = _classify_result_path(path, root)

    assert _result_matches(entry, "waterfall 9.1 anchor 0028", RESULT_FILTER_ALL)
    assert _result_matches(entry, "powder", "Waterfall")
    assert not _result_matches(entry, "7.6", RESULT_FILTER_ALL)
    assert not _result_matches(entry, "", "Peak location")

    single_crystal = _classify_result_path(
        root / "heatmaps" / "single_crystal" / "roi_area"
        / "pressure_9p1_GPa" / "anchor_0028.png",
        root,
    )
    assert _result_matches(single_crystal, "single-crystal ROI-area", RESULT_FILTER_ALL)


def test_result_index_ignores_temporary_old_and_unrelated_pngs(tmp_path):
    root = tmp_path / "results"
    powder = (
        root / "heatmaps" / "powder" / "roi_area"
        / "pressure_3p5_GPa" / "anchor_0001.png"
    )
    crystal = (
        root / "heatmaps" / "single_crystal" / "location"
        / "pressure_7p6_GPa" / "anchor_0002.png"
    )
    stale = root / "heatmaps" / ".powder.old-123" / "roi_area" / "stale.png"
    partial = root / "heatmaps" / ".powder.tmp-123" / "roi_area" / "partial.png"
    unrelated = root / "notes" / "screenshot.png"
    for path in (powder, crystal, stale, partial, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    assert _find_result_paths(root) == [powder, crystal]


def test_window_within_pressure_lookup_uses_original_frame_index(tmp_path):
    import h5py
    import numpy as np

    root = tmp_path / "results"
    root.mkdir()
    with h5py.File(root / "correlations_powder.h5", "w") as h5:
        frames = h5.create_group("frames")
        frames.create_dataset("index", data=np.asarray([4, 9], dtype="i4"))
        frames.create_dataset("pressure", data=np.asarray([3.5, 11.5]))

    pressures = _load_result_pressures(root)

    assert pressures == {("powder", 4): 3.5, ("powder", 9): 11.5}
