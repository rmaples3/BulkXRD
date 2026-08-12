from pathlib import Path
import sys, importlib
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
mods = [
    'seriesxrd',
    'seriesxrd.core', 'seriesxrd.core.config', 'seriesxrd.core.env', 'seriesxrd.core.naming',
    'seriesxrd.core.io', 'seriesxrd.core.masks', 'seriesxrd.core.handoff', 'seriesxrd.core.inspect',
    'seriesxrd.core.uiprefs',
    'seriesxrd.guikit', 'seriesxrd.guikit.theme', 'seriesxrd.guikit.tkstyle',
    'seriesxrd.guikit.tooltip', 'seriesxrd.guikit.dpi', 'seriesxrd.guikit.dpi',
    'seriesxrd.calib', 'seriesxrd.calib.processing', 'seriesxrd.calib.dioptas',
    'seriesxrd.calib.worker', 'seriesxrd.calib.gui',
    'seriesxrd.calib.run_gui',
    'seriesxrd.reduce', 'seriesxrd.reduce.processing',
    'seriesxrd.reduce.session', 'seriesxrd.reduce.review',
    'seriesxrd.reduce.worker', 'seriesxrd.reduce.gui', 'seriesxrd.reduce.run_gui',
    'seriesxrd.app',
    'seriesxrd.analysis', 'seriesxrd.analysis.background', 'seriesxrd.analysis.peaks',
    'seriesxrd.analysis.review', 'seriesxrd.analysis.session',
    'seriesxrd.analysis.worker', 'seriesxrd.analysis.gui', 'seriesxrd.analysis.run_gui',
    'seriesxrd.analysis.phases', 'seriesxrd.analysis.refdata', 'seriesxrd.analysis.identify',
    'seriesxrd.analysis.heatmap', 'seriesxrd.analysis.mldata',
    'seriesxrd.analysis.residual', 'seriesxrd.analysis.frame_metadata',
    'seriesxrd.analysis.ml_features', 'seriesxrd.analysis.ml_simulate',
    'seriesxrd.analysis.ml_rank', 'seriesxrd.analysis.ml_scorer',
    'seriesxrd.analysis.ml_train',
    'seriesxrd.analysis.parallel', 'seriesxrd.analysis.batch',
    'seriesxrd.analysis.benchmark', 'seriesxrd.analysis.corpus',
    'seriesxrd.analysis.unknowns', 'seriesxrd.analysis.series',
    'seriesxrd.analysis.microstructure',
    'seriesxrd.analysis.fractions', 'seriesxrd.analysis.refine_export',
    'seriesxrd.analysis.refine_import',
    'seriesxrd.analysis.spots',
    'seriesxrd.correlations', 'seriesxrd.correlations.processing',
    'seriesxrd.correlations.tracks', 'seriesxrd.correlations.export',
    'seriesxrd.correlations.plots', 'seriesxrd.correlations.review',
    'seriesxrd.correlations.session', 'seriesxrd.correlations.batch',
    'seriesxrd.correlations.gui', 'seriesxrd.correlations.run_gui',
    'seriesxrd.reduce.straighten', 'seriesxrd.reduce.texture',
    'seriesxrd.reduce.watch',
]
for m in mods:
    importlib.import_module(m)
    print('IMPORT OK:', m)


def test_cp1252_stdio_does_not_crash():
    """A non-ASCII log line must not abort on a legacy (cp1252) console once
    make_stdio_robust() has run — Windows would otherwise raise UnicodeEncodeError."""
    import io
    from seriesxrd.core.config import make_stdio_robust, print_status
    saved = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")  # strict
        make_stdio_robust()
        print("[IDENTIFY] window 2.0*sigma σ → ok", flush=True)  # σ, →
        print_status("pressure prior σ set", "INFO")
        sys.stdout.flush()
    finally:
        sys.stdout = saved
    # print_status must also survive even without reconfigure (its own fallback).
    saved = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        print_status("raw σ fallback", "INFO")   # no make_stdio_robust here
        sys.stdout.flush()
    finally:
        sys.stdout = saved


test_cp1252_stdio_does_not_crash()
print('ALL IMPORTS OK')
