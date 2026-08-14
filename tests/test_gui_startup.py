"""Real-Tk GUI startup smoke test.

Constructs the unified application against a fresh workspace, pumps the
event loop once, and closes. Runs only where a display is available (CI
provides one via xvfb); the construction happens in a subprocess so a hang
or native crash fails the test instead of the test session.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _display import tk_display_available


@pytest.mark.skipif(not tk_display_available(),
                    reason="no usable Tk display (run under xvfb-run to enable)")
def test_unified_app_constructs_and_closes():
    script = (
        "import sys\n"
        "from tkinter import ttk\n"
        "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg\n"
        "from matplotlib.colors import to_hex\n"
        "from matplotlib.figure import Figure\n"
        "from seriesxrd.app import SeriesXRDApp\n"
        "from seriesxrd.guikit import theme\n"
        "app = SeriesXRDApp(sys.argv[1])\n"
        "assert 'refinement' in app.analysis_pane.pages\n"
        "assert tuple(app.correlation_pane.pages) == "
        "('input', 'settings', 'run', 'results')\n"
        "app.root.update_idletasks()\n"
        "app.root.update()\n"
        "fig = Figure()\n"
        "fig.add_subplot(111).plot([0, 1], [0, 1], color='#123456')\n"
        "canvas = FigureCanvasTkAgg(fig, master=app.root)\n"
        "canvas.get_tk_widget().pack()\n"
        "app.analysis_pane._theme_test_canvas = canvas\n"
        "theme.set_theme('latte')\n"
        "app.root.update_idletasks()\n"
        "assert app.analysis_pane.input_text.cget('background').lower() == theme.C.BG2\n"
        "assert ttk.Style(app.root).lookup('TLabel', 'foreground').lower() == theme.C.FG\n"
        "assert to_hex(fig.get_facecolor()) == theme.C.BG\n"
        "theme.set_theme('mocha')\n"
        "app.root.update_idletasks()\n"
        "assert app.analysis_pane.input_text.cget('background').lower() == theme.C.BG2\n"
        "assert to_hex(fig.get_facecolor()) == theme.C.BG\n"
        "app.root.destroy()\n"
        "print('GUI-STARTUP-OK')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "workspace"
        proc = subprocess.run([sys.executable, "-c", script, str(ws)],
                              capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"GUI startup failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "GUI-STARTUP-OK" in proc.stdout


if __name__ == "__main__":
    test_unified_app_constructs_and_closes()
    print("OK")
