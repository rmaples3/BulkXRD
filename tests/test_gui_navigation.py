"""Regression tests for the unified left-rail workflow navigation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _display import tk_display_available


def test_calibration_and_reduction_use_left_rail_navigation(tmp_path):
    pytest.importorskip("tkinter")
    if not tk_display_available():
        pytest.skip("no usable Tk display")

    from seriesxrd.app import SeriesXRDApp

    app = SeriesXRDApp(tmp_path / "workspace")
    try:
        app.root.withdraw()
        app.root.update_idletasks()
        assert app._ws_label.cget("text").strip() == "workspace"
        assert app.calib_pane.config["dioptas_image_flip"] is False
        assert app.calib_pane._orientation_text().startswith("Flip OFF")

        identify = app.analysis_pane
        assert identify._show_contamination.get() is True
        identify._show_contamination.set(False)
        assert identify._show_contamination.get() is False
        identify._identify_help_var.set(True)
        identify._toggle_identify_help()
        help_row = int(identify._identify_help.grid_info()["row"])
        overlaps = [
            widget
            for widget in identify._identify_help.master.grid_slaves(row=help_row)
            if widget is not identify._identify_help
        ]
        assert overlaps == [], "Identify instructions are covered by another widget"
        identify._identify_help_var.set(False)
        identify._toggle_identify_help()

        expected_pages = {
            app.calib_pane: ("inputs", "mask", "generate", "review", "accept"),
            app.reduce_pane: (
                "calibration", "dataset", "settings", "run", "review", "gallery"
            ),
            app.correlation_pane: ("input", "settings", "run", "results"),
        }
        for pane, page_keys in expected_pages.items():
            assert tuple(pane.pages) == page_keys
            assert pane._nav_rail.winfo_manager() == "grid"
            for key in page_keys:
                pane.select_page(key)
                assert pane._nav_rail.selection() == (pane._nav_items[key],)
    finally:
        app.root.destroy()
