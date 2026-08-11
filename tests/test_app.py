"""Headless tests for unified-application coordination."""
from __future__ import annotations

from pathlib import Path

from seriesxrd.app import (
    SeriesXRDApp,
    _workspace_display_name,
    workspace_launch_args,
)
from seriesxrd.analysis.gui import AnalysisApp
from seriesxrd.calib.gui import CalibrationApp
from seriesxrd.correlations.gui import CorrelationApp
from seriesxrd.reduce.gui import ReductionApp


class _FakePane:
    def __init__(self, allow_close: bool):
        self.allow_close = allow_close
        self.confirm_calls = 0
        self.shutdown_calls = []

    def confirm_shutdown(self) -> bool:
        self.confirm_calls += 1
        return self.allow_close

    def shutdown(self, confirm: bool = True) -> bool:
        self.shutdown_calls.append(confirm)
        return self.allow_close


class _FakeRoot:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def test_unified_close_confirmation_is_transactional():
    """A later veto must not partially shut down an earlier stage."""
    app = SeriesXRDApp.__new__(SeriesXRDApp)
    app.calib_pane = _FakePane(True)
    app.reduce_pane = _FakePane(False)
    app.analysis_pane = _FakePane(True)
    app.correlation_pane = _FakePane(True)
    app.root = _FakeRoot()

    app._on_quit()

    assert app.calib_pane.confirm_calls == 1
    assert app.reduce_pane.confirm_calls == 1
    assert app.analysis_pane.confirm_calls == 0
    assert app.correlation_pane.confirm_calls == 0
    assert app.calib_pane.shutdown_calls == []
    assert app.reduce_pane.shutdown_calls == []
    assert app.analysis_pane.shutdown_calls == []
    assert app.correlation_pane.shutdown_calls == []
    assert not app.root.destroyed


def test_unified_close_shuts_all_panes_after_all_confirm():
    app = SeriesXRDApp.__new__(SeriesXRDApp)
    app.calib_pane = _FakePane(True)
    app.reduce_pane = _FakePane(True)
    app.analysis_pane = _FakePane(True)
    app.correlation_pane = _FakePane(True)
    app.root = _FakeRoot()

    app._on_quit()

    for pane in (
        app.calib_pane,
        app.reduce_pane,
        app.analysis_pane,
        app.correlation_pane,
    ):
        assert pane.confirm_calls == 1
        assert pane.shutdown_calls == [False]
    assert app.root.destroyed


def test_workspace_launch_args_use_module_entry_point(tmp_path):
    args = workspace_launch_args(tmp_path, executable="python-test")
    assert args[:4] == ["python-test", "-m", "seriesxrd.app", "--workspace"]
    assert Path(args[4]) == tmp_path.resolve()


def test_workspace_header_uses_only_the_folder_name(tmp_path):
    workspace = tmp_path / "institution" / "researcher" / "SeriesXRD-Demo"
    assert _workspace_display_name(workspace) == "SeriesXRD-Demo"


def test_pattern_review_contamination_panel_can_be_collapsed():
    assert AnalysisApp._review_panel_layout(True, True) == (
        ("pattern", "cake", "contamination"),
        (3, 2, 1),
    )
    assert AnalysisApp._review_panel_layout(True, False) == (
        ("pattern", "cake"),
        (3, 2),
    )
    assert AnalysisApp._review_panel_layout(False, False) == (("pattern",), (3,))


def test_scientific_tools_are_exposed_by_gui_controllers():
    for name in (
        "export_refinement_clicked",
        "export_gsas_raw_clicked",
        "import_gsas_results_clicked",
        "run_microstructure_clicked",
        "run_phase_fractions_clicked",
        "run_spot_tracking_clicked",
    ):
        assert callable(getattr(AnalysisApp, name))
    assert callable(getattr(ReductionApp, "_run_texture_job"))


def test_analysis_completion_listener_hands_off_existing_file(tmp_path):
    controller = AnalysisApp.__new__(AnalysisApp)
    controller._analysis_listeners = []
    controller.log = lambda *_args, **_kwargs: None
    received = []
    analysis_h5 = tmp_path / "analysis.h5"
    analysis_h5.touch()

    controller.add_analysis_listener(received.append)
    controller._notify_analysis_ready(analysis_h5)

    assert received == [str(analysis_h5)]


def test_unified_analysis_handoff_selects_correlations_stage():
    app = SeriesXRDApp.__new__(SeriesXRDApp)
    received = []
    selected = []
    app.correlation_pane = type(
        "CorrelationPane",
        (),
        {"set_analysis": lambda _self, path: received.append(path)},
    )()
    app._select_stage = selected.append

    app._on_analysis_ready("analysis.h5")

    assert received == ["analysis.h5"]
    assert selected == [3]


def test_all_workflow_stages_expose_left_rail_navigation():
    for controller in (
        CalibrationApp,
        ReductionApp,
        AnalysisApp,
        CorrelationApp,
    ):
        assert callable(getattr(controller, "_build_navigation"))
        assert callable(getattr(controller, "select_page"))
