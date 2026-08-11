"""Tabbed interface for configuring, running, and reviewing analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import queue
import subprocess
import sys
import threading

from ..core.config import (
    TOOL_NAME, read_json, write_json, ensure_dir,
    now_iso, now_timestamp, output_base,
)
from ..core.naming import next_available_path
from ..core.processes import terminate_process_tree, worker_popen
from ..guikit import theme
from ..guikit.tkstyle import apply_theme
from ..guikit.tooltip import ToolTip as _ToolTip
from ..guikit.mpl_embed import embed_figure
from ..guikit.labels import (unit_label, AZIMUTH_LABEL, INTENSITY_LABEL,
                              INTENSITY_ARB_LABEL, D_SPACING_LABEL,
                              PRESSURE_LABEL, FRAME_LABEL, CONTAMINATION_LABEL)


def _tk_imports():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


HELP: Dict[str, str] = {
    # Input / output
    "reduced_h5_file": (
        "A reduced_*.h5 from the reduction stage. Step 1 needs its "
        "intensity_robust channel; if it is missing, re-run reduction with "
        "the robust pattern enabled."
    ),
    "analysis_h5_file": (
        "Output analysis HDF5. Blank = <stem>_analysis.h5 beside the reduced file."
    ),
    # Run scope
    "run_step1": "Run Step 1: SNIP baseline + spot-residual separation.",
    "run_step2": "Run Step 2: pseudo-Voigt peak fitting on the selected Peak source.",
    "run_step3": (
        "Run Step 3a: match fitted peaks against the candidate phases via each "
        "phase's EOS. Gives per-frame phase confidence and, for pressure series, "
        "a per-frame pressure estimate."
    ),
    # Step 3a
    "identify_all_phases": (
        "Score every phase in the reference library (bundled + yours) against "
        "each frame instead of only the Phases-tab selection. Slower and more "
        "prone to spurious matches; use it when you don't know what is in the "
        "sample. This searches your library, not all of ICSD/MP."
    ),
    "p_min": "Lower bound (GPa) of the EOS pressure search. 0 for ambient work.",
    "p_max": (
        "Upper bound (GPa) of the EOS pressure search. Phases with an EOS "
        "validity ceiling (p_max in their entry) are capped there regardless."
    ),
    "rel_tol": (
        "Peak-match tolerance as a fraction of d-spacing (0.01 = 1%). Raise it "
        "if real lines just miss their match; too loose lets wrong phases match."
    ),
    "seen_conf": (
        "Confidence (0-1) above which a phase counts as present in a frame. "
        "Present phases are subtracted in the residual step. Default 0.5."
    ),
    "identify_wavelength": (
        "X-ray wavelength (Å). Only needed for 2θ data; blank = read from the "
        "reduced file's PONI. q-axis data never needs it."
    ),
    "intensity_k": (
        "Weight (0-1) of the intensity-agreement factor in the confidence. "
        "0 = match on positions only. Keep it low (default 0.3): texture and "
        "spotty rings distort measured intensities."
    ),
    "use_frame_temperature": (
        "Apply each frame's temperature (Frame metadata tab) to the predicted "
        "d-spacings of phases that define thermal expansion. Uncheck to treat "
        "all frames as ambient temperature."
    ),
    "unknown_tracking_axis": (
        "Axis used to link residual peaks into unknown tracks. same follows the "
        "Peaks tab's seed order; frame preserves collection order; "
        "pressure/temperature/time sort frames by that metadata and allow smooth "
        "peak drift along the chosen axis."
    ),
    "unknown_group_by": (
        "Keep independent series separate while tracking unknowns. same follows "
        "the Peaks tab's seed grouping. Use scan for datasets named with "
        "scan001/scan034-style tokens; use folder when each scan lives in its "
        "own directory."
    ),
    "unknown_axis_predictor": (
        "For pressure/temperature/time tracking, extrapolate the next peak center "
        "from the track's recent slope. Keep on for pressure-shifting unknowns."
    ),
    "unknown_link_tol_fwhm": (
        "Linking tolerance in fitted-peak widths. Raise if a real unknown line "
        "splits into short tracks; lower if nearby unrelated peaks are merging."
    ),
    "unknown_max_gap": (
        "How many missing ordered samples a track may skip before it is closed. "
        "With pressure tracking, samples are pressure-sorted frames."
    ),
    "unknown_max_axis_gap": (
        "Optional maximum physical-axis jump between linked observations: GPa for "
        "pressure, K for temperature, seconds for time. Blank = no physical cap."
    ),
    "unknown_min_frames": "Minimum distinct observations required to keep an unknown track.",
    "unknown_jaccard": (
        "Co-occurrence threshold for merging tracks into one unknown cluster. "
        "Higher = stricter clustering."
    ),
    # Step 1
    "max_half_window": (
        "Widest feature (bins) SNIP treats as background. Set to 1.5-2x the "
        "half-width of your broadest real peak. Too wide flattens broad peaks "
        "into the baseline. Default 40."
    ),
    "n_passes": "SNIP passes. 1 is enough in practice. Default 1.",
    "use_lls": (
        "Compress dynamic range (log-log-sqrt) before SNIP. Keep it on: without "
        "it the baseline overshoots under intense sharp peaks."
    ),
    "contamination_threshold": (
        "Flag frames whose spot-contamination score (sum of positive "
        "spot_residual) exceeds this value. Blank = no flagging."
    ),
    "robust_source": (
        "Spot-suppressed channel Step 1 builds on.\n"
        "robust = azimuthal median (default).\n"
        "straightened = cake de-waved median (patterns/intensity_straightened_"
        "robust). Run the reduce stage's Review → 'Write straightened 1D' first "
        "(needs saved cakes). Use it when the sample sat off the calibrant "
        "position and rings arrive as double-horned peaks; cake-less frames fall "
        "back to the ordinary median automatically."
    ),
    # Step 2
    "peak_source": (
        "Signal the peaks are fit on. auto (default) = the reduce-side sigmaclip "
        "channel if present, else hybrid. clean = azimuthal median minus "
        "baseline: cleanest, but drops intensity on spotty/textured/incomplete "
        "rings. hybrid = clean plus the broad part of (mean − median), rejecting "
        "narrow single-crystal spikes. sigmaclip = the reduce-side trimmed mean "
        "(best; enable it in reduction). mean keeps everything including "
        "diamond spots — diagnostic only. spots = fit (mean − median) itself: "
        "the SINGLE-CRYSTAL SAMPLE channel — a crystal's sparse reflections are "
        "rejected by every median-based channel exactly like diamond spots, and "
        "this is where they end up. If peaks you can see in the pattern are "
        "missing from the fit, try hybrid or mean; if the sample is a crystal, "
        "run a spots pass."
    ),
    "sensitivity": (
        "Preset for the detection knobs below (any left blank). conservative = "
        "fewer, cleaner peaks; sensitive = catches weak shoulders but more noise "
        "hits. Explicit values below override the preset. Default normal."
    ),
    "auto_range": (
        "When Fit min/max are blank, skip the beamstop ramp and the dead "
        "detector tail automatically (trims at most the outer ~15%). Uncheck to "
        "fit the full axis."
    ),
    "hybrid_spike_bins": (
        "Hybrid source only: mean-excess narrower than this many bins is "
        "removed as a single-crystal spike; broader excess is kept as texture. "
        "Default 5."
    ),
    "min_snr": (
        "Peak height threshold in noise-floor units. Lower = more peaks, more "
        "noise hits. Blank = preset (normal 5)."
    ),
    "min_prominence_snr": (
        "Peak prominence threshold in noise-floor units; controls whether a "
        "shoulder on a stronger peak counts as its own peak. Blank = preset "
        "(normal 2)."
    ),
    "window_factor": (
        "Fit-window half-width as a multiple of the estimated FWHM. Default 3."
    ),
    "max_chi2": (
        "Reduced χ² over a peak's own span, above which its fit may be flagged "
        "bad. Default 25. This is the test that governs weak, noise-limited "
        "peaks. Judged together with Max rel. misfit — a peak has to fail both."
    ),
    "max_rel_misfit": (
        "Rms fit residual as a fraction of the peak's own height, above which "
        "the fit is bad. Default 0.05. This is the test that governs bright "
        "peaks: χ² is measured against the background noise, so a bright peak "
        "fails it for being well measured. Both measures are per peak, not per "
        "fitted group. Lower it (0.03–0.04) to be stricter about profile shape."
    ),
    "fit_min": (
        "Lower fit bound (q or 2θ). Set just above the beamstop onset — the "
        "low-angle ramp inflates the noise floor and hides weak peaks. "
        "Blank = auto range."
    ),
    "fit_max": (
        "Upper fit bound (q or 2θ). Set below the noisy detector tail. "
        "Blank = auto range."
    ),
    "edge_bins": (
        "Drop peaks within this many bins of either end of the pattern (edge "
        "artefacts). Blank = preset (normal 5)."
    ),
    "min_fwhm_bins": (
        "Reject peaks narrower than this many bins — a real peak spans several; "
        "1-bin spikes are noise. If real peaks trip this, the pattern is "
        "under-sampled: re-reduce with more bins (see the run log's npt "
        "recommendation). Blank = preset (normal 2)."
    ),
    "detrend_bins": (
        "Detection-only local-baseline window (bins): removes broad background "
        "SNIP left behind so weak peaks clear the noise threshold. Fitting "
        "still uses the un-detrended signal. Size it a few peak widths. "
        "0 = off. Default 81."
    ),
    "propagate_seeds": (
        "Seed each frame's detection with the previous frame's good peak "
        "centers, so a reflection keeps its identity as it drifts through the "
        "series (compression, heating). Keep on for series data."
    ),
    "seed_tracking_axis": (
        "Order used for peak-seed propagation (Step 2). frame uses collection "
        "order; pressure/temperature/time sort frames by metadata so seeds move "
        "along the physical scan. Step 3c's Unknown tracking can mirror this "
        "(its 'same')."
    ),
    "seed_group_by": (
        "Keep seed propagation inside independent series (Step 2). Use scan for "
        "scan001/scan034-style names, or folder when each scan lives in its own "
        "directory. Step 3c's Unknown grouping can mirror this (its 'same')."
    ),
    "seed_axis_predictor": (
        "For pressure/temperature/time seed order, shift seed centers by their "
        "recent drift before fitting the next frame. Keep on for pressure scans."
    ),
    "seed_max_axis_gap": (
        "Optional physical-axis jump that resets seed memory: GPa for pressure, "
        "K for temperature, seconds for time. Blank = no cap."
    ),
    # Step 3a metadata-prior knobs
    "use_pressure_prior": (
        "Confine each phase's pressure fit to the frame's metadata pressure "
        "± window instead of the full p_min-p_max search. This is the main "
        "accuracy control for pressure series: without it, a wrong phase can "
        "slide along pressure until a few lines coincide. Needs frame "
        "pressures (Frame metadata tab)."
    ),
    "pressure_window": (
        "Half-width (GPa) of the prior window when a frame has no per-frame "
        "uncertainty. 0.5-2 GPa is typical. Default 2."
    ),
    "pressure_sigma_k": (
        "When a frame has a pressure uncertainty (CSV import), the window is "
        "k·σ instead of the fixed value. Default 2."
    ),
    "marker_prior": (
        "No metadata pressures? Fit the marker-category phases first and reuse "
        "the best marker's per-frame pressure as the prior for everything else."
    ),
    "min_matched": (
        "Reflections a phase must match (one-to-one) to count as present. "
        "Guards against 1-2 line coincidences. Default 3."
    ),
    "allow_sparse": (
        "Let phases below Min matched still be subtracted in the residual "
        "(e.g. sparse pressure markers). Off by default."
    ),
    # Series plots / grid map
    "map_value": (
        "Per-frame scalar shown on the grid: integrated or max intensity of "
        "the fit source (optionally within the ROI below), contamination "
        "score, peak count, P, T, or one phase's matched-reflection intensity."
    ),
    "map_layout": (
        "How frames are placed on the grid. 'scan lines' uses the collection "
        "order plus the controls to the right (frames per line, direction, "
        "serpentine). 'coordinates' places each frame by its stage position "
        "(/frames/pos_x, pos_y — import them on the Frame meta tab via CSV "
        "or the frame headers); no other input needed."
    ),
    "map_line_len": (
        "Frames per scan line — how many frames the stage collected before "
        "turning (horizontal) or how many rows tall a column is (vertical). "
        "Not needed with the 'coordinates' layout."
    ),
    "map_order": (
        "horizontal = scan lines are rows of the map; vertical = scan lines "
        "are columns."
    ),
    "map_serpentine": (
        "Checked = boustrophedon (stage reverses direction every line). "
        "Unchecked = unidirectional raster (every line scans the same way)."
    ),
    "map_roi_min": "ROI lower bound on the radial axis for intensity values. Blank = full axis.",
    "map_roi_max": "ROI upper bound on the radial axis for intensity values. Blank = full axis.",
}


class AnalysisApp:
    def __init__(self, config_path: "str | Path", parent=None):
        tk, ttk, filedialog, messagebox = _tk_imports()
        self.tk, self.ttk, self.filedialog, self.messagebox = (
            tk, ttk, filedialog, messagebox
        )
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: Dict[str, Any] = read_json(self.config_path)
        self.config.setdefault("session_config_path", str(self.config_path))
        if parent is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.root.title(f"{TOOL_NAME} Analysis")
            self.root.geometry("1180x780")
            self.root.minsize(960, 640)
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self._owns_root = False
            self.root = parent.winfo_toplevel()
        self._embed_parent = parent  # None when standalone, ttk.Frame when embedded

        self.vars: Dict[str, Any] = {}
        self._run_proc: "subprocess.Popen | None" = None
        # Host applications can subscribe to a completed Analysis HDF5 without
        # coupling the analysis worker to the next workflow stage.
        self._analysis_listeners: "list[Any]" = []
        # Thread-safe logging: worker threads push lines here; a main-thread
        # poller drains them into the Text widget.
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        # Thread-safe run events: the worker thread pushes ("progress"|"done"|
        # "error", ...) tuples here and the main-thread poller dispatches them.
        # Tkinter is not thread-safe, so the worker must NEVER touch widgets (or
        # even root.after) directly — that can deadlock the event loop.
        self._event_queue: "queue.Queue[tuple]" = queue.Queue()
        # History buffer so lines aren't lost before the console window opens.
        self._log_history: "list[str]" = []

        # State for status bar
        self._frame_count: int = 0
        self._worker_status: str = "idle"

        # Review tab state
        self._review_nframes: int = 0
        self._review_contamination = None  # numpy array or None
        self._review_after: "int | None" = None  # debounce scheduler id

        self._build_gui()
        theme.register_widget_tree(self._embed_parent or self.root)
        theme.register_restyle(self._restyle_theme)
        self._drain_log_queue()
        self.log("GUI initialized")
        self.save_config(silent=True)
        self._update_status_bar()

        # Auto-inspect on startup if the reduced file already exists.
        _h5 = self.config.get("reduced_h5_file", "")
        if _h5 and Path(_h5).is_file():
            self.root.after(300, self.inspect_input_clicked)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _restyle_theme(self):
        """Repaint this live pane without touching worker or analysis state."""
        apply_theme(self.root, self.ttk)
        theme.register_widget_tree(self._embed_parent or self.root)
        theme.restyle_widgets()
        for attr, tags in (
            ("phases_tree", {"user": theme.C.ACCENT}),
            ("_fm_table", {"user": theme.C.ACCENT}),
        ):
            tree = getattr(self, attr, None)
            if tree is not None:
                for tag, color in tags.items():
                    tree.tag_configure(tag, foreground=color)
        table = getattr(self, "_identify_table", None)
        if table is not None:
            table.tag_configure("present", foreground=theme.C.ACCENT2)
            table.tag_configure("absent", foreground=theme.C.MUTED)
        theme.restyle_owner_figures(self)

    def _build_gui(self):
        tk, ttk = self.tk, self.ttk
        if self._owns_root:
            apply_theme(self.root, ttk)
        _container = self._embed_parent if self._embed_parent is not None else self.root
        outer = ttk.Frame(_container, padding=6)
        outer.pack(fill="both", expand=True)

        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 6))
        ttk.Label(
            topbar, text="Analysis",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side="left")
        ttk.Button(
            topbar, text="View log", command=self.open_console_logs,
        ).pack(side="right", padx=4)

        # Status bar carved out before the notebook so it's always visible.
        self._status_bar_frame = ttk.Frame(outer, relief="sunken")
        self._status_bar_frame.pack(side="bottom", fill="x", pady=(2, 0))

        self._build_navigation(outer)

        self._build_status_bar()

    def _build_navigation(self, outer):
        """Hierarchical navigation: a left rail (Configure → Run → Review →
        Export, with sub-pages) and a raised-frame content area.

        Replaces the previous single row of 12 numbered tabs, which hid later
        tabs at small window widths and 200% display scaling. Page builders
        are unchanged — each old tab body is now a page.
        """
        tk, ttk = self.tk, self.ttk

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        rail = ttk.Treeview(body, show="tree", selectmode="browse")
        rail.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        rail.column("#0", width=170, stretch=False)
        self._nav_rail = rail

        content = ttk.Frame(body)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        sections = [
            ("Configure", [
                ("data",       "Data",           self._tab_input),
                ("background", "Background",     self._tab_background),
                ("peaks",      "Peaks",          self._tab_peaks),
                ("phases",     "Phases",         self._tab_phases),
                ("metadata",   "Frame metadata", self._tab_frame_metadata),
                ("identify",   "Identification", self._tab_identify),
            ]),
            ("Run", [
                ("run",        "Run analysis",   self._tab_run),
            ]),
            ("Review", [
                ("pattern",    "Pattern review", self._tab_review),
                ("peakmap",    "Peak map",       self._tab_heatmap),
                ("phasemap",   "Phase map",      self._tab_patternmap),
                ("unknowns",   "Unknowns",       self._tab_unknowns),
                ("spatial",    "Spatial map",    self._tab_gridmap),
            ]),
            ("Refinement", [
                ("refinement", "GSAS-II round trip", self._tab_refinement),
            ]),
            ("Export", [
                ("export",     "Export",         self._tab_export),
            ]),
        ]

        self.pages: Dict[str, Any] = {}
        self._nav_items: Dict[str, str] = {}      # page key -> tree item id
        self._nav_first_child: Dict[str, str] = {}  # section item id -> page key
        for section_label, pages in sections:
            sec_id = rail.insert("", "end", text=section_label, open=True)
            for key, label, builder in pages:
                frame = ttk.Frame(content, padding=10)
                builder(frame)
                frame.grid(row=0, column=0, sticky="nsew")
                self.pages[key] = frame
                item = rail.insert(sec_id, "end", text=label,
                                   values=(key,), tags=(key,))
                self._nav_items[key] = item
                self._nav_first_child.setdefault(sec_id, key)

        # Legacy aliases so older call sites / muscle memory keep working.
        self.tabs = {
            "1 Data": self.pages["data"], "2 Background": self.pages["background"],
            "3 Peaks": self.pages["peaks"], "4 Phases": self.pages["phases"],
            "5 Metadata": self.pages["metadata"],
            "6 Identification": self.pages["identify"],
            "7 Run": self.pages["run"], "8 Pattern review": self.pages["pattern"],
            "9 Peak map": self.pages["peakmap"],
            "10 Phase map": self.pages["phasemap"],
            "11 Unknowns": self.pages["unknowns"],
            "12 Spatial map": self.pages["spatial"],
            "13 Refinement": self.pages["refinement"],
        }

        def _on_select(_event=None):
            sel = rail.selection()
            if not sel:
                return
            item = sel[0]
            vals = rail.item(item, "values")
            if vals:
                self.pages[vals[0]].tkraise()
                if vals[0] == "run":
                    self._update_preflight()
            else:
                # A section header: forward to its first page.
                first = self._nav_first_child.get(item)
                if first:
                    rail.selection_set(self._nav_items[first])
        rail.bind("<<TreeviewSelect>>", _on_select)
        self.select_page("data")

    def select_page(self, key: str) -> None:
        """Raise a navigation page by key (e.g. 'data', 'run', 'pattern')."""
        item = self._nav_items.get(key)
        if item is None:
            return
        try:
            self._nav_rail.selection_set(item)
            self._nav_rail.see(item)
        except Exception:
            pass
        self.pages[key].tkraise()
        if key == "run":
            self._update_preflight()

    def _tab_refinement(self, frame):
        """Explain and expose the external GSAS-II refinement round trip."""
        ttk = self.ttk
        intro = ttk.Label(
            frame, style="Muted.TLabel", justify="left", wraplength=760,
            text=(
                "SeriesXRD first finds peaks and proposes which phases are present. "
                "It then exports those phase models and the measured patterns to "
                "GSAS-II. GSAS-II is a separate program that fits every phase "
                "against each complete diffraction pattern and calculates refined "
                "weight fractions, unit cells, uncertainties, and fit quality. "
                "Importing the sequential results brings those final numbers back "
                "onto the original frame series; it does not run GSAS-II inside "
                "SeriesXRD."
            ),
        )
        intro.pack(anchor="w", fill="x", pady=(0, 10))
        self.autowrap(intro)

        flow = ttk.LabelFrame(frame, text="Refinement round trip", padding=10)
        flow.pack(fill="x", anchor="n")
        ttk.Label(flow, text="1  Prepare patterns, phase CIFs, instrument file, and frame map").grid(
            row=0, column=0, sticky="w", padx=4, pady=5)
        ttk.Button(flow, text="Export refinement bundle…",
                   command=self.export_refinement_clicked).grid(
            row=0, column=1, sticky="w", padx=8, pady=5)
        ttk.Label(
            flow,
            text=("2  Open the bundle in GSAS-II and perform a sequential "
                  "Rietveld refinement using the included README."),
            style="Muted.TLabel", justify="left", wraplength=500,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=5)
        ttk.Label(flow, text="3  Return WgtFrac, uncertainties, cells, Rwp/GOF, and convergence").grid(
            row=2, column=0, sticky="w", padx=4, pady=5)
        ttk.Button(flow, text="Import sequential results…",
                   command=self.import_gsas_results_clicked).grid(
            row=2, column=1, sticky="w", padx=8, pady=5)

        raw = ttk.LabelFrame(frame, text="Alternative pattern preparation", padding=10)
        raw.pack(fill="x", anchor="n", pady=(12, 0))
        ttk.Button(raw, text="Export GSAS-ready raw patterns…",
                   command=self.export_gsas_raw_clicked).pack(
            side="left", padx=4)
        ttk.Label(
            raw, style="Muted.TLabel", justify="left", wraplength=560,
            text=("Re-integrates original detector frames with counting "
                  "uncertainties, optionally summing frames by pressure. Use this "
                  "when the normal analysis-pattern export is not suitable for "
                  "your refinement."),
        ).pack(side="left", padx=10)

        result = ttk.Label(
            frame, style="Muted.TLabel", justify="left", wraplength=760,
            text=("Imported results are stored under /refinement in the analysis "
                  "HDF5. Existing /fractions screening estimates are preserved so "
                  "you can compare the automated estimate with the Rietveld result."),
        )
        result.pack(anchor="w", fill="x", pady=(12, 0))
        self.autowrap(result)

    def _tab_export(self, frame):
        """Point to context-specific exports; refinement has its own page."""
        ttk = self.ttk
        note = ttk.Label(
            frame, style="Muted.TLabel", justify="left", wraplength=680,
            text=("Context-bound exports live with their results:\n"
                  "  • Pattern review — per-frame .xy patterns and stacked "
                  "figures\n"
                  "  • Unknowns — cluster diagrams, spot tracks, spot masks\n"
                  "  • Frame metadata — selected-frame metadata CSV"))
        note.pack(anchor="w", pady=(10, 0))

    def _build_status_bar(self):
        ttk = self.ttk
        bar = self._status_bar_frame
        self.status_session = ttk.Label(bar, text="", style="Muted.TLabel", anchor="w")
        self.status_session.pack(side="left", padx=(6, 12))
        self.status_input = ttk.Label(bar, text="input: none", style="Muted.TLabel", anchor="w")
        self.status_input.pack(side="left", padx=(0, 12))
        self.status_frames = ttk.Label(bar, text="frames: —", style="Muted.TLabel", anchor="w")
        self.status_frames.pack(side="left", padx=(0, 12))
        self.status_worker = ttk.Label(bar, text="idle", style="Muted.TLabel", anchor="e")
        self.status_worker.pack(side="right", padx=6)
        # Transient non-modal notifications (successful saves/exports) land
        # here instead of interrupting with a dialog.
        self.status_notify = ttk.Label(bar, text="", style="Ok.TLabel", anchor="w")
        self.status_notify.pack(side="left", padx=(0, 12))
        self._notify_after_id = None

    def notify(self, text: str, *, level: str = "INFO", seconds: int = 8):
        """Non-modal notification: one line in the status bar plus the log.

        For successful saves/loads/exports — outcomes the user should see
        without having to dismiss anything. Errors stay modal.
        """
        one_line = " ".join(str(text).split())
        self.log(one_line, level)
        lbl = getattr(self, "status_notify", None)
        if lbl is None:
            return
        if self._notify_after_id is not None:
            try:
                lbl.after_cancel(self._notify_after_id)
            except Exception:
                pass
        lbl.configure(text=one_line[:160])
        self._notify_after_id = lbl.after(
            int(seconds * 1000), lambda: lbl.configure(text=""))

    def _update_status_bar(self):
        try:
            session = self.config.get("session_name", "")
            if hasattr(self, "status_session"):
                self.status_session.configure(
                    text=f"session: {session}" if session else "session: (unnamed)")
            if hasattr(self, "status_input"):
                h5 = self.config.get("reduced_h5_file", "")
                bname = Path(h5).name if h5 else "none"
                self.status_input.configure(text=f"input: {bname}")
            if hasattr(self, "status_frames"):
                fc = self._frame_count
                self.status_frames.configure(
                    text=f"frames: {fc}" if fc > 0 else "frames: —")
            if hasattr(self, "status_worker"):
                self.status_worker.configure(text=self._worker_status)
        except Exception:
            pass

    # -- shared small widgets -----------------------------------------------

    def field(self, parent, key, label, browse=None, row=None, width=80, col=0):
        """Entry field bound to a config key, with optional Browse button.

        ``col`` places the pair in a second (third, ...) column group so dense
        tabs can lay parameters out side by side instead of one tall stack."""
        tk, ttk = self.tk, self.ttk
        var = tk.StringVar(value=str(self.config.get(key, "")))
        self.vars[key] = var
        base = int(col) * 3
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=base, sticky="w", padx=4, pady=3)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=base + 1, sticky="we", padx=4, pady=3)
        if not hasattr(self, "entry_widgets"):
            self.entry_widgets: Dict[str, Any] = {}
        self.entry_widgets[key] = entry
        if browse:
            ttk.Button(
                parent, text="Browse",
                command=lambda: self.browse_into(key, browse),
            ).grid(row=row, column=base + 2, padx=4)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(entry, txt)
        parent.columnconfigure(base + 1, weight=1)

    def autowrap(self, label, pad=28):
        """Keep a long explanatory label's wraplength tracking its parent's
        width, so text reflows instead of running off the tab on narrow
        windows (fixed wraplengths clipped on small screens)."""
        def _fit(event, lbl=label):
            try:
                w = max(240, int(event.width) - pad)
                lbl.configure(wraplength=w)
            except Exception:
                pass
        label.master.bind("<Configure>", _fit, add="+")

    def checkbox(self, parent, key, label, row=None, col=0):
        """Checkbox bound to a boolean config key (``col`` = column group)."""
        tk, ttk = self.tk, self.ttk
        var = tk.BooleanVar(value=bool(self.config.get(key, False)))
        self.vars[key] = var
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=int(col) * 3, columnspan=2, sticky="w", padx=4, pady=3)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(cb, txt)

    def combo(self, parent, key, label, values, row=None, width=16, default=""):
        """Read-only combobox bound to a config key."""
        tk, ttk = self.tk, self.ttk
        cur = str(self.config.get(key, default) or (default or (values[0] if values else "")))
        var = tk.StringVar(value=cur)
        self.vars[key] = var
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)
        cb = ttk.Combobox(parent, textvariable=var, values=list(values),
                          state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="w", padx=4, pady=3)
        txt = HELP.get(key, "")
        if txt:
            _ToolTip(lbl, txt)
            _ToolTip(cb, txt)

    def browse_into(self, key, mode):
        if mode == "dir":
            value = self.filedialog.askdirectory(title=f"Select {key}")
        else:
            value = self.filedialog.askopenfilename(title=f"Select {key}")
        if value:
            self.vars[key].set(value)
            self.save_config(silent=True)

    def pull_vars(self):
        for key, var in self.vars.items():
            self.config[key] = var.get()

    def save_config(self, silent=False):
        self.pull_vars()
        self.config["updated_at"] = now_iso()
        write_json(self.config_path, self.config)
        if not silent:
            self.log(f"Config saved: {self.config_path}")

    # ------------------------------------------------------------------
    # Thread-safe logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        line = f"[{now_iso()}] [{level}] {message}"
        print(line, flush=True)
        if threading.current_thread() is not threading.main_thread():
            self._log_queue.put(line)
            return
        self._insert_log_line(line)

    def _insert_log_line(self, line: str):
        self._log_history.append(line)
        if len(self._log_history) > 5000:
            self._log_history = self._log_history[-5000:]
        if hasattr(self, "log_text"):
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except self.tk.TclError:
                pass
        if hasattr(self, "run_log_text"):
            try:
                self.run_log_text.configure(state="normal")
                self.run_log_text.insert("end", line + "\n")
                self.run_log_text.see("end")
                self.run_log_text.configure(state="disabled")
            except self.tk.TclError:
                pass

    def _drain_log_queue(self):
        """Recurring main-thread poller: flush queued lines into the widget."""
        if getattr(self, "_closing", False):
            return
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._insert_log_line(line)
        except queue.Empty:
            pass
        # Dispatch run events on the main thread.
        try:
            while True:
                evt = self._event_queue.get_nowait()
                self._dispatch_run_event(evt)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _dispatch_run_event(self, evt: tuple):
        """Handle a worker run event on the main thread (see _worker_thread)."""
        kind = evt[0]
        try:
            if kind == "progress":
                self._update_progress(evt[1], evt[2], evt[3])
            elif kind == "done":
                self._run_done(evt[1], evt[2])
            elif kind == "error":
                self._run_error(evt[1])
        except Exception as e:  # never let a dispatch error wedge the poller
            self.log(f"run-event handler failed ({kind}): {e!r}", "WARN")

    # ------------------------------------------------------------------
    # Console log window
    # ------------------------------------------------------------------

    def open_console_logs(self):
        tk, ttk = self.tk, self.ttk
        try:
            if getattr(self, "_log_window", None) and self._log_window.winfo_exists():
                self._log_window.deiconify()
                self._log_window.lift()
                self._log_window.focus_set()
                return
        except self.tk.TclError:
            pass
        self._log_window = tk.Toplevel(self.root)
        self._log_window.title("SeriesXRD — Analysis log")
        self._log_window.geometry("900x420")
        self._log_window.configure(bg=theme.C.BG)
        self.log_text = tk.Text(
            self._log_window, wrap="word", state="disabled",
            font=("TkFixedFont", 10), bg=theme.C.BG2, fg=theme.C.FG,
            insertbackground=theme.C.FG, selectbackground=theme.C.ACCENT,
        )
        scroll = ttk.Scrollbar(
            self._log_window, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.configure(state="normal")
        if self._log_history:
            self.log_text.insert("end", "\n".join(self._log_history) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self._log_window.protocol("WM_DELETE_WINDOW", self._hide_console_logs)

    def _hide_console_logs(self):
        if getattr(self, "_log_window", None):
            try:
                self._log_window.withdraw()
            except self.tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Tab 1 — Input
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_analysis_output(reduced: str) -> str:
        """Default analysis output path beside the reduced input
        (``<stem>_analysis.h5``) — mirrors background.run_background_separation."""
        r = str(reduced or "").strip()
        if not r:
            return ""
        p = Path(r)
        return str(p.with_name(p.stem + "_analysis.h5"))

    def _autofill_analysis_output(self, *_):
        """Keep the output Analysis HDF5 in step with the reduced input.

        Fills it from the input's default when the field is blank or still holds
        the value we last auto-derived (so a path the user typed is never
        clobbered)."""
        if "reduced_h5_file" not in self.vars or "analysis_h5_file" not in self.vars:
            return
        reduced = str(self.vars["reduced_h5_file"].get() or "").strip()
        derived = self._derive_analysis_output(reduced)
        if not derived:
            return
        current = str(self.vars["analysis_h5_file"].get() or "").strip()
        if current == derived:
            self._auto_out_value = derived   # already matches → adopt as managed
            return
        if current and current != getattr(self, "_auto_out_value", ""):
            return  # user-customized — leave it alone
        self.vars["analysis_h5_file"].set(derived)
        self.config["analysis_h5_file"] = derived
        self._auto_out_value = derived

    def _tab_input(self, frame):
        tk, ttk = self.tk, self.ttk
        self.field(frame, "reduced_h5_file", "Reduced data", browse="file", row=0, width=64)
        self.field(frame, "analysis_h5_file", "Analysis results", browse="file",
                   row=1, width=64)
        # Auto-derive the output path whenever the reduced input changes (typed,
        # browsed, or handed off from the reduction stage).
        self.vars["reduced_h5_file"].trace_add("write", self._autofill_analysis_output)
        self._autofill_analysis_output()
        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=3, sticky="w", padx=2, pady=6)
        ttk.Button(
            btns, text="Inspect input", command=self.inspect_input_clicked,
        ).pack(side="left", padx=2)
        # The raw HDF5 tree and attribute dumps are implementation detail —
        # off by default, one click away when needed.
        self._inspect_advanced = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            btns, text="Advanced details", variable=self._inspect_advanced,
            command=self._render_input_report,
        ).pack(side="left", padx=8)

        # Warn if robust pattern is missing — analysis requires it.
        self._robust_warn_label = ttk.Label(btns, text="", style="Warn.TLabel")
        self._robust_warn_label.pack(side="left", padx=12)

        self.input_text = tk.Text(
            frame, height=20, bg=theme.C.BG2, fg=theme.C.FG, insertbackground=theme.C.FG,
            relief="flat", state="disabled", font=("TkFixedFont", 10),
        )
        self.input_text.grid(
            row=3, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(1, weight=1)

    def inspect_input_clicked(self):
        """Inspect the reduced HDF5 (and analysis HDF5 if present) and show reports."""
        self.pull_vars()
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            self.messagebox.showerror(
                "Input", "Select a reduced .h5 file first (1 Data tab).")
            return
        from ..reduce.review import inspect_reduction
        self.log(f"Inspecting reduced HDF5: {reduced}")
        try:
            review_r = inspect_reduction(reduced)
        except Exception as e:
            self.messagebox.showerror("Inspect failed", repr(e))
            return
        self._last_review_reduced = review_r

        # Warn about missing robust pattern.
        robust_ok = bool(review_r.get("robust_present"))
        if hasattr(self, "_robust_warn_label"):
            if not robust_ok:
                self._robust_warn_label.configure(
                    text="WARNING: robust pattern missing — re-run reduction with "
                         "robust_1d=True before analysis.")
            else:
                self._robust_warn_label.configure(text="")

        # Update frame count for status bar.
        nf = review_r.get("n_frames", 0)
        if nf:
            self._frame_count = int(nf)

        # If an analysis HDF5 already exists, summarize it too.
        self._last_review_analysis = None
        self._last_review_analysis_error = ""
        analysis = str(self.config.get("analysis_h5_file", "") or "").strip()
        if analysis and Path(analysis).is_file():
            from .review import inspect_analysis
            self.log(f"Inspecting analysis HDF5: {analysis}")
            try:
                review_a = inspect_analysis(analysis)
                self._last_review_analysis = review_a
                if review_a.get("n_frames"):
                    self._frame_count = int(review_a["n_frames"])
            except Exception as e:
                self._last_review_analysis_error = repr(e)

        self._render_input_report()
        self._update_status_bar()
        self.save_config(silent=True)

    def _render_input_report(self):
        """Render the cached inspection results into the Data page, honoring
        the Advanced-details toggle (no file re-read on toggle)."""
        review_r = getattr(self, "_last_review_reduced", None)
        if review_r is None:
            return
        from ..reduce.review import structure_report as reduce_report
        from .review import structure_report as analysis_report
        advanced = bool(self._inspect_advanced.get())
        lines = ["=== Reduced HDF5 ===", reduce_report(review_r, advanced=advanced)]
        review_a = getattr(self, "_last_review_analysis", None)
        if review_a is not None:
            lines += ["", "=== Analysis HDF5 ===",
                      analysis_report(review_a, advanced=advanced)]
        elif getattr(self, "_last_review_analysis_error", ""):
            lines += ["", f"[Could not inspect analysis HDF5: "
                          f"{self._last_review_analysis_error}]"]
        if not advanced:
            lines += ["", "(Enable “Advanced details” for the raw HDF5 tree "
                          "and all attributes.)"]
        self.input_text.configure(state="normal")
        self.input_text.delete("1.0", "end")
        self.input_text.insert("end", "\n".join(lines) + "\n")
        self.input_text.configure(state="disabled")

    def set_reduced(self, path: "str | Path") -> None:
        """Called by the host app to wire in the reduced file after reduction.

        Sets reduced_h5_file in config + its StringVar, saves, switches to the
        Input tab, and refreshes the input summary.
        """
        p = str(path or "").strip()
        if not p:
            return
        self.config["reduced_h5_file"] = p
        if "reduced_h5_file" in self.vars:
            self.vars["reduced_h5_file"].set(p)
        self.save_config(silent=True)
        self.log(f"Reduced HDF5 received: {p}")
        try:
            self.select_page("data")
        except Exception:
            pass
        if Path(p).is_file():
            self.root.after(100, self.inspect_input_clicked)

    def add_analysis_listener(self, fn) -> None:
        """Register a callback invoked after a successful analysis run."""
        if not callable(fn):
            raise TypeError("analysis listener must be callable")
        self._analysis_listeners.append(fn)

    def _notify_analysis_ready(self, path: "str | Path") -> None:
        """Notify downstream stages while keeping listener failures isolated."""
        p = str(path or "").strip()
        if not p or not Path(p).is_file():
            return
        for listener in tuple(self._analysis_listeners):
            try:
                listener(p)
            except Exception as exc:
                self.log(f"Analysis handoff listener failed: {exc!r}", "WARN")

    # ------------------------------------------------------------------
    # Tab 2 — Background
    # ------------------------------------------------------------------

    def _tab_background(self, frame):
        ttk = self.ttk
        self.checkbox(frame, "run_step1", "Run Step 1 — background separation", row=0)
        self.field(frame, "max_half_window", "Max half-window (bins)", row=2, width=14)
        self.field(frame, "n_passes", "SNIP passes", row=3, width=14)
        self.checkbox(frame, "use_lls", "Use LLS transform (Log-Log-Sqrt compression)", row=4)
        self.field(frame, "contamination_threshold",
                   "Contamination threshold (blank = off)", row=6, width=14)
        self.combo(frame, "robust_source", "Background source",
                   ["robust", "straightened"], row=7, width=14, default="robust")
        _bg_help = ttk.Label(
            frame,
            text=(
                "Step 1 splits each pattern into background, sample signal, and\n"
                "single-crystal contamination:\n\n"
                "  spot_residual = mean − median   (spots hit the mean, not the median)\n"
                "  baseline = SNIP(robust)         (iterative peak-clipping estimate)\n"
                "  clean = robust − baseline       (what the later steps build on)\n\n"
                "If broad peaks lose height after Step 1, the SNIP window is too\n"
                "wide — reduce Max half-window. The stored channels let every later\n"
                "step rebuild its own fit source, so nothing is lost."
            ),
            style="Muted.TLabel", justify="left", wraplength=640,
        )
        _bg_help.grid(row=8, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))
        self.autowrap(_bg_help)

    # ------------------------------------------------------------------
    # Tab 3 — Peaks
    # ------------------------------------------------------------------

    def _tab_peaks(self, frame):
        tk, ttk = self.tk, self.ttk
        self.checkbox(frame, "run_step2", "Run Step 2 — pseudo-Voigt peak fitting", row=0)
        # Primary controls: fit source + sensitivity preset + auto range.
        self.combo(frame, "peak_source", "Peak source",
                   ["auto", "hybrid", "sigmaclip", "clean", "mean", "spots"],
                   row=1, default="auto")
        self.combo(frame, "sensitivity", "Sensitivity",
                   ["conservative", "normal", "sensitive"], row=2, default="normal")
        self.checkbox(frame, "auto_range", "Auto valid q/2θ range (blank Fit min/max)", row=3)
        ttk.Label(frame, text="Advanced (blank = follow the Sensitivity preset):",
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=6, sticky="w",
                                         padx=4, pady=(10, 0))
        # Two column-groups so the tab fits on ~700px-tall screens.
        self.field(frame, "min_snr", "Min SNR (height)", row=5, width=12)
        self.field(frame, "min_prominence_snr", "Min prominence SNR", row=6, width=12)
        self.field(frame, "window_factor", "Window factor (× FWHM)", row=7, width=12)
        self.field(frame, "max_chi2", "Max reduced χ²", row=8, width=12)
        self.field(frame, "max_rel_misfit", "Max rel. misfit", row=9, width=12)
        self.field(frame, "fit_min", "Fit 2θ/q min (blank=auto)", row=10, width=12)
        self.field(frame, "fit_max", "Fit 2θ/q max (blank=auto)", row=5, width=12, col=1)
        self.field(frame, "edge_bins", "Edge guard (bins)", row=6, width=12, col=1)
        self.field(frame, "min_fwhm_bins", "Min FWHM (bins)", row=7, width=12, col=1)
        self.field(frame, "hybrid_spike_bins", "Hybrid spike width (bins)", row=8, width=12, col=1)
        self.field(frame, "detrend_bins", "Detrend window (bins, 0=off)", row=9, width=12, col=1)
        self.checkbox(frame, "propagate_seeds",
                      "Propagate peak seeds frame-to-frame", row=11)
        seedrow = ttk.Frame(frame)
        seedrow.grid(row=12, column=0, columnspan=6, sticky="w", padx=4, pady=3)
        ttk.Label(seedrow, text="Seed order", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        self.vars["seed_tracking_axis"] = tk.StringVar(
            value=str(self.config.get("seed_tracking_axis", "frame") or "frame"))
        _seed_axis = ttk.Combobox(
            seedrow, textvariable=self.vars["seed_tracking_axis"],
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        _seed_axis.pack(side="left")
        _ToolTip(_seed_axis, HELP["seed_tracking_axis"])
        ttk.Label(seedrow, text="group by", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["seed_group_by"] = tk.StringVar(
            value=str(self.config.get("seed_group_by", "none") or "none"))
        _seed_group = ttk.Combobox(
            seedrow, textvariable=self.vars["seed_group_by"],
            values=["none", "scan", "folder"], state="readonly", width=8)
        _seed_group.pack(side="left")
        _ToolTip(_seed_group, HELP["seed_group_by"])
        self.vars["seed_axis_predictor"] = tk.BooleanVar(
            value=bool(self.config.get("seed_axis_predictor", True)))
        _seed_pred = ttk.Checkbutton(
            seedrow, text="predict drift", variable=self.vars["seed_axis_predictor"])
        _seed_pred.pack(side="left", padx=(10, 2))
        _ToolTip(_seed_pred, HELP["seed_axis_predictor"])
        ttk.Label(seedrow, text="axis gap", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["seed_max_axis_gap"] = tk.StringVar(
            value=str(self.config.get("seed_max_axis_gap", "")))
        _seed_gap = ttk.Entry(seedrow, textvariable=self.vars["seed_max_axis_gap"], width=7)
        _seed_gap.pack(side="left")
        _ToolTip(_seed_gap, HELP["seed_max_axis_gap"])
        _pk_help = ttk.Label(
            frame,
            text=(
                "Step 2 fits pseudo-Voigt profiles A·(η·Lorentzian + (1−η)·Gaussian) "
                "to the selected source. Start with Peak source, Sensitivity, and "
                "Auto range; leave the advanced fields blank unless a specific "
                "problem points at one (each tooltip says which). Common cases: "
                "visible peaks missing from the fit → source 'hybrid' or 'mean'; "
                "weak shoulders not detected → Sensitivity 'sensitive'; noise fitted "
                "as peaks → 'conservative'; stepped/blocky patterns → too few bins, "
                "re-reduce (see run log). Rejection flags: LOW_AMP=1, BAD_CHI2=2, "
                "CENTER_DRIFT=4, WIDTH_BOUND=8, NO_CONVERGE=16."
            ),
            style="Muted.TLabel", justify="left", wraplength=640,
        )
        _pk_help.grid(row=13, column=0, columnspan=6, sticky="w", padx=6, pady=(12, 4))
        self.autowrap(_pk_help)

    # ------------------------------------------------------------------
    # Tab 4 — Run
    # ------------------------------------------------------------------

    def _tab_run(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x")
        self.run_btn = ttk.Button(top, text="Run analysis", command=self.run_analysis)
        self.run_btn.pack(side="left", padx=4, pady=4)
        self.cancel_btn = ttk.Button(
            top, text="Cancel", command=self.cancel_analysis, state="disabled")
        self.cancel_btn.pack(side="left", padx=4, pady=4)
        ttk.Label(top, text="Workers:", style="Muted.TLabel").pack(side="left", padx=(16, 2))
        _w_var = tk.StringVar(value=str(self.config.get("num_workers", "0")))
        self.vars["num_workers"] = _w_var
        _w_entry = ttk.Entry(top, textvariable=_w_var, width=5)
        _w_entry.pack(side="left", padx=2)
        _ToolTip(_w_entry, "Parallel worker processes for all steps. "
                           "0 = auto (CPU count − 1), 1 = serial.")

        # Preflight: what will run, on what input, into what output — visible
        # before committing to a run instead of discovered from error dialogs.
        pf = ttk.LabelFrame(frame, text="Preflight", padding=(8, 4))
        pf.pack(fill="x", padx=4, pady=(6, 0))
        self._preflight_label = ttk.Label(pf, text="", justify="left",
                                          style="Muted.TLabel")
        self._preflight_label.pack(anchor="w")
        self._preflight_warn = ttk.Label(pf, text="", justify="left",
                                         style="Warn.TLabel")
        self._preflight_warn.pack(anchor="w")

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=4, pady=6)
        self.progress_label = ttk.Label(frame, text="Idle", style="Muted.TLabel")
        self.progress_label.pack(anchor="w", padx=6)

        # Completion summary: filled by _run_done on success, with direct
        # follow-up actions instead of a modal popup.
        done = ttk.Frame(frame)
        done.pack(fill="x", padx=4)
        self._completion_label = ttk.Label(done, text="", justify="left",
                                           style="Ok.TLabel")
        self._completion_label.pack(side="left", anchor="w")
        self._completion_review_btn = ttk.Button(
            done, text="Review results",
            command=lambda: (self.select_page("pattern"), self.load_review()))
        self._completion_open_btn = ttk.Button(
            done, text="Open output folder", command=self._open_output_folder)
        # (buttons pack on completion)

        self.run_log_text = tk.Text(
            frame, bg=theme.C.BG2, fg=theme.C.FG, insertbackground=theme.C.FG, relief="flat",
            state="disabled", font=("TkFixedFont", 9),
        )
        self.run_log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._update_preflight()

    def _update_preflight(self):
        """Refresh the Run page's preflight block from the current config."""
        if not hasattr(self, "_preflight_label"):
            return
        try:
            self.pull_vars()
        except Exception:
            pass
        cfg = self.config
        reduced = str(cfg.get("reduced_h5_file", "") or "").strip()
        out_h5 = str(cfg.get("analysis_h5_file", "") or "").strip()
        steps = []
        if bool(cfg.get("run_step1", True)):
            steps.append("1 background")
        if bool(cfg.get("run_step2", True)):
            steps.append("2 peaks")
        if bool(cfg.get("run_ml_rank", False)):
            steps.append("3b ML rank")
        if bool(cfg.get("run_step3", False)):
            steps.append("3a identify + residual + unknowns")
        frames = getattr(self, "_frame_count", 0)
        lines = [
            f"Input:   {Path(reduced).name if reduced else '(none)'}"
            + (f"    frames: {frames}" if frames else ""),
            f"Steps:   {', '.join(steps) if steps else '(none enabled)'}",
            f"Output:  {out_h5 or '(derived from input)'}",
        ]
        self._preflight_label.configure(text="\n".join(lines))

        warns = []
        if not reduced or not Path(reduced).is_file():
            warns.append("No reduced input file — set it on Configure → Data.")
        rev = getattr(self, "_last_review_reduced", None)
        if rev is not None and not rev.get("robust_present", True):
            warns.append("Robust pattern missing — re-run reduction with "
                         "robust_1d=True before Step 1.")
        if bool(cfg.get("run_step3", False)):
            cands = cfg.get("candidate_phases") or []
            if not cands and not bool(cfg.get("identify_all_phases", False)) \
                    and not bool(cfg.get("run_ml_rank", False)):
                warns.append("Step 3a is enabled with no candidate phases — "
                             "pick phases (Configure → Phases), enable ML "
                             "ranking, or set identify-all.")
        if not steps:
            warns.append("Nothing to run — enable at least one step.")
        self._preflight_warn.configure(text="\n".join(warns))

    def _open_output_folder(self):
        out_h5 = str(self.config.get("analysis_h5_file", "") or "").strip()
        folder = Path(out_h5).parent if out_h5 else None
        if not folder or not folder.is_dir():
            self.log("No output folder to open yet.", "WARN")
            return
        try:
            if sys.platform.startswith("win"):
                import os as _os
                _os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            self.log(f"Could not open {folder}: {exc!r}", "WARN")

    def run_analysis(self):
        if self._run_proc is not None:
            self.messagebox.showinfo("Busy", "An analysis is already running.")
            return
        self.save_config(silent=True)
        self.pull_vars()

        run_step1 = bool(self.config.get("run_step1", True))
        run_step2 = bool(self.config.get("run_step2", True))
        run_step3 = bool(self.config.get("run_step3", False))
        if not run_step1 and not run_step2 and not run_step3:
            self.messagebox.showerror(
                "Nothing to run",
                "Enable at least one of 'Run Step 1', 'Run Step 2', or 'Run Step 3a' "
                "on the Background / Peaks / Identify tabs.")
            return

        if run_step1:
            reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
            if not reduced or not Path(reduced).is_file():
                self.messagebox.showerror(
                    "Input missing",
                    "Step 1 requires a reduced HDF5.\n"
                    "Set it on the '1 Data' tab first.")
                return

        backend_dir = self.config.get(
            "backend_dir", str(Path(__file__).resolve().parents[1]))
        python_exe = Path(self.config.get("python_exe", sys.executable))
        logs_root = (
            self.config.get("logs_root", "")
            or str(output_base(self.config) / "logs")
        )
        ensure_dir(Path(logs_root))
        out_json = str(
            next_available_path(Path(logs_root) / f"analysis_{now_timestamp()}.json"))
        worker_script = str(Path(backend_dir) / "analysis" / "worker.py")
        if not Path(worker_script).is_file():
            self.messagebox.showerror(
                "Worker not found",
                f"Analysis worker script not found:\n{worker_script}\n\n"
                "Check 'backend_dir' in the session config.")
            return

        cmd = [
            str(python_exe), worker_script,
            "--config", str(self.config_path),
            "--output-json", out_json,
        ]
        self.log("Worker command: " + " ".join(cmd))
        self._update_preflight()
        self._clear_completion()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0)
        # Fresh run: clear any leftover cancelled/failed/done styling and state.
        self._cancel_requested = False
        self.progress_label.configure(text="Starting worker ...", style="Muted.TLabel")
        self._worker_status = "running"
        self._update_status_bar()

        def _worker_thread():
            try:
                proc = worker_popen(
                    cmd, cwd=backend_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                self._run_proc = proc
                assert proc.stdout is not None
                # Watchdog: once the process exits, close stdout so the reader
                # below can't hang on a pipe still held open by forked pool
                # workers / the multiprocessing resource tracker. Without this
                # the loop never sees EOF, _run_done never fires, the run stays
                # stuck at "running", and closing the app always warns.
                def _watch(p=proc):
                    p.wait()
                    try:
                        if p.stdout is not None:
                            p.stdout.close()
                    except Exception:
                        pass
                threading.Thread(target=_watch, daemon=True).start()
                try:
                    for line in proc.stdout:
                        line = line.rstrip()
                        parts = line.split()
                        if len(parts) == 3 and parts[0] in ("[ANALYSIS]", "[PEAKS]", "[IDENTIFY]"):
                            try:
                                done = int(parts[1])
                                total = int(parts[2])
                                _phase_labels = {
                                    "[ANALYSIS]": "Background",
                                    "[PEAKS]": "Peaks",
                                    "[IDENTIFY]": "Identify",
                                }
                                phase = _phase_labels.get(parts[0], parts[0])
                                self._event_queue.put(
                                    ("progress", phase, done, total))
                                continue
                            except ValueError:
                                pass
                        self.log(line)
                except (ValueError, OSError):
                    pass  # stdout closed by the watchdog once the process exited
                rc = int(proc.wait())
                self._event_queue.put(("done", rc, out_json))
            except Exception as e:
                self._event_queue.put(("error", repr(e)))

        threading.Thread(target=_worker_thread, daemon=True).start()

    def _update_progress(self, phase: str, done: int, total: int):
        self.progress.configure(maximum=max(total, 1), value=done)
        self.progress_label.configure(text=f"{phase}: {done} / {total} frames")

    def _clear_completion(self):
        if not hasattr(self, "_completion_label"):
            return
        self._completion_label.configure(text="")
        self._completion_review_btn.pack_forget()
        self._completion_open_btn.pack_forget()

    def _show_completion(self, manifest: dict):
        """Completion summary with direct follow-up actions (non-modal)."""
        if not hasattr(self, "_completion_label"):
            return
        lines = []
        s1 = manifest.get("step1") or {}
        if s1:
            n_flag = s1.get("n_flagged")
            extra = f", {n_flag} contamination-flagged" if n_flag else ""
            lines.append(f"Step 1: {s1.get('n_frames', '?')} frames"
                         f" ({s1.get('n_excluded', 0)} excluded{extra})")
            if s1.get("spotty_sample"):
                lines.append("  ⚠ spotty/coarse-grained sample diagnosed — "
                             "peak fitting used the mean channel")
        s2 = manifest.get("step2") or {}
        if s2:
            lines.append(f"Step 2: {s2.get('n_good', '?')} good / "
                         f"{s2.get('n_peaks', '?')} fitted peaks")
            if s2.get("npt_recommended"):
                lines.append(f"  ⚠ peaks are under-sampled — re-reduce with "
                             f"npt_1d ≈ {s2['npt_recommended']}")
        s3 = manifest.get("step3") or {}
        if s3:
            seen = [n for n, d in (s3.get("summary") or {}).items()
                    if d.get("n_frames_seen")]
            lines.append("Step 3a: " + (", ".join(seen) if seen
                                        else "no phase cleared the gate"))
        self._completion_label.configure(
            text="\n".join(lines) if lines else "Run finished.")
        self._completion_review_btn.pack(side="right", padx=4)
        self._completion_open_btn.pack(side="right", padx=4)

    def _run_done(self, returncode: int, out_json: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if returncode != 0:
            # A user-requested cancel also exits nonzero — don't call it a
            # failure or pop an error dialog for a deliberate act. Say what
            # state things are in and what Run will do next.
            if getattr(self, "_cancel_requested", False):
                self._cancel_requested = False
                self._worker_status = "cancelled"
                self._update_status_bar()
                self.progress_label.configure(
                    text="Cancelled — completed steps were saved to the analysis "
                         "file; interrupted steps left no partial output (atomic "
                         "writes). Run analysis re-runs every enabled step.",
                    style="Muted.TLabel")
                self.log("Run cancelled by user.", "WARN")
                return
            self._worker_status = "failed"
            self._update_status_bar()
            self.progress_label.configure(
                text=f"Failed (return code {returncode})", style="Warn.TLabel")
            self.messagebox.showerror(
                "Analysis failed",
                f"Worker return code {returncode}\nSee the Run tab log.")
            return
        try:
            manifest = read_json(out_json)
        except Exception as e:
            self.log(f"Could not read manifest: {e!r}", "WARN")
            manifest = {}
        h5 = manifest.get("analysis_h5_file", "")
        steps = manifest.get("steps", [])
        self._worker_status = "done"
        self._update_status_bar()
        self.progress_label.configure(
            text=f"Done: {', '.join(steps)} -> {h5}", style="Ok.TLabel")
        self.log(f"Analysis complete: {h5}")
        self._show_completion(manifest)
        if h5:
            self.config["analysis_h5_file"] = h5
            if "analysis_h5_file" in self.vars:
                self.vars["analysis_h5_file"].set(h5)
            self.save_config(silent=True)
            self._notify_analysis_ready(h5)
            # Log peak summary if Step 2 ran.
            s2 = manifest.get("step2", {})
            if s2:
                n_peaks = s2.get("n_peaks", "?")
                n_good = s2.get("n_good", "?")
                self.log(f"Peak fitting: {n_good} good / {n_peaks} total peaks")
            # Log Step 3a summary if it ran.
            s3 = manifest.get("step3", {})
            if s3:
                for name, d in s3.get("summary", {}).items():
                    try:
                        pm = d.get("pressure_median")
                        p_txt = f"{pm:.1f} GPa" if pm is not None and pm == pm else "n/a"
                        self.log(
                            f"  {name}: seen in {d['n_frames_seen']} frame(s) "
                            f"(conf>{d.get('seen_conf', 0.5):.2f}); "
                            f"best recall {d.get('max_recall', 0.0):.2f}, "
                            f"best precision {d.get('max_precision', 0.0):.2f}, "
                            f"up to {d.get('max_matched', 0)} line(s) matched, "
                            f"median P={p_txt}"
                        )
                    except Exception:
                        pass
        # Refresh the views off the new results. Some loads are heavy (the
        # pattern map runs pymatgen reflection-track simulation), so stagger
        # them via after() rather than running all in one synchronous blast —
        # that kept the event loop from pumping and showed "Not responding"
        # right after a phase match. Each step yields to the loop between runs.
        loaders = [
            ("inspect", self.inspect_input_clicked),
            ("review", self.load_review),
            ("peak map", self.load_heatmap),
            ("identify", self.load_identify),
            ("pattern map", self.load_pattern_map),
            ("unknowns", self.load_unknowns),
        ]

        def _run_loader(i=0):
            if i >= len(loaders):
                return
            name, fn = loaders[i]
            try:
                fn()
            except Exception as e:
                import traceback
                self.log(f"Auto {name} load failed: {e!r}", "WARN")
                self.log(traceback.format_exc(), "WARN")
            self.root.after(20, lambda: _run_loader(i + 1))

        self.root.after(20, _run_loader)

    def _run_error(self, err: str):
        self._run_proc = None
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self._worker_status = "failed"
        self._update_status_bar()
        self.progress_label.configure(text="Launch error", style="Warn.TLabel")
        self.messagebox.showerror("Worker launch error", err)

    def cancel_analysis(self):
        proc = self._run_proc
        if proc is not None and proc.poll() is None:
            self._cancel_requested = True
            self.cancel_btn.configure(state="disabled")
            self.progress_label.configure(text="Cancelling ...", style="Muted.TLabel")
            terminate_process_tree(proc)
            self.log("Cancel requested — stopped worker process tree", "WARN")

    # ------------------------------------------------------------------
    # Tab 5 — Review (single-frame QC)
    # ------------------------------------------------------------------

    def _tab_review(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))

        ttk.Button(ctrl, text="Load review", command=self.load_review).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_review_fig", None), "Review")
                   ).pack(side="left", padx=4)
        _rev_exp = ttk.Button(ctrl, text="Export frame…",
                              command=self.review_export_frame_clicked)
        _rev_exp.pack(side="left", padx=4)
        _ToolTip(_rev_exp, (
            "Export the frame shown here: its pattern as a two-column .xy "
            "(channel of your choice) and optionally its fitted peaks as "
            "peaks.csv. Select several frames on the Frame meta tab to "
            "export a batch."))
        _rev_ringless = ttk.Button(ctrl, text="Ringless cake…",
                                   command=self.review_export_ringless_clicked)
        _rev_ringless.pack(side="left", padx=4)
        _ToolTip(_rev_ringless, (
            "Export this frame's cake with the powder rings removed (each "
            "radial column minus its azimuthal median — the W/marker rings "
            "cancel, isolated crystallite spots survive). Writes a "
            "quick-look PNG plus the raw .npy array, exactly the image the "
            "spot tracker detects on. Needs cakes saved in the reduction."))

        # Trace toggles live on their own row: one long row of controls used
        # to run wider than the window and clip on the right.
        togglerow = ttk.Frame(frame)
        togglerow.pack(fill="x", pady=(0, 4))

        ttk.Label(ctrl, text="Frame:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._review_idx_var = tk.IntVar(value=0)
        # NOTE: the Scale is deliberately NOT linked to _review_idx_var. When the
        # Scale and the Spinbox shared the variable, the Scale's callback echoed a
        # var.set() back into the Spinbox mid-arrow-press, and the press applied
        # its increment twice (one click advanced two frames). The slider now
        # drives the var only through _on_review_slider (change-guarded), and the
        # spinbox/render paths sync the slider explicitly.
        self._review_scale = ttk.Scale(
            ctrl, from_=0, to=0, orient="horizontal", length=200,
            command=self._on_review_slider,
        )
        self._review_scale.pack(side="left", padx=2)
        self._review_spinbox = ttk.Spinbox(
            ctrl, from_=0, to=0, width=6,
            textvariable=self._review_idx_var,
            command=self._on_review_spinbox,
        )
        self._review_spinbox.pack(side="left", padx=2)

        # Trace overlays
        self._show_mean = tk.BooleanVar(value=True)
        self._show_robust = tk.BooleanVar(value=True)
        self._show_baseline = tk.BooleanVar(value=True)
        self._show_clean = tk.BooleanVar(value=True)
        self._show_spot = tk.BooleanVar(value=False)
        self._show_residual = tk.BooleanVar(value=False)
        self._show_peaks = tk.BooleanVar(value=True)
        self._show_residual_peaks = tk.BooleanVar(value=False)
        self._show_unknowns = tk.BooleanVar(value=False)
        self._show_cake = tk.BooleanVar(value=False)
        self._show_contamination = tk.BooleanVar(value=True)
        for var, label in [
            (self._show_mean, "mean"),
            (self._show_robust, "robust"),
            (self._show_baseline, "baseline"),
            (self._show_clean, "clean"),
            (self._show_spot, "spot_residual"),
            (self._show_residual, "residual"),
            (self._show_peaks, "fitted peaks"),
            (self._show_residual_peaks, "residual peaks"),
            (self._show_unknowns, "unknown peaks"),
            (self._show_cake, "cake (2D)"),
            (self._show_contamination, "contamination"),
        ]:
            cb = ttk.Checkbutton(togglerow, text=label, variable=var,
                                 command=self._schedule_review_render)
            cb.pack(side="left", padx=2)
            if label == "residual":
                _ToolTip(cb, "Step-3a removal result: /residual/clean.")
            elif label == "residual peaks":
                _ToolTip(cb, "Peaks re-fitted on /residual/clean.")
            elif label == "unknown peaks":
                _ToolTip(cb, "Step-3c unknown-track observations for this frame.")
            elif label == "contamination":
                _ToolTip(cb, "Show or collapse the frame-series contamination score plot.")
        self._review_source_reduced = ""

        # Matplotlib area
        self.review_plot_frame = ttk.Frame(frame)
        self.review_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.review_plot_frame,
            text="Load the analysis HDF5 to plot per-frame traces.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

    def load_review(self):
        """Load metadata from the analysis HDF5 and render the current frame."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            # Silently skip (called on auto-refresh); show error only if user triggered.
            return
        from .review import inspect_analysis
        try:
            info = inspect_analysis(path)
        except Exception as e:
            self.log(f"Review load failed: {e!r}", "WARN")
            return
        if not info.get("ok_to_read"):
            self.log("Analysis HDF5 not readable for review.", "WARN")
            return
        nf = int(info.get("n_frames", 0))
        self._review_nframes = nf
        self._review_source_reduced = str(info.get("source_reduced", "") or "")
        contam = info.get("contamination")
        self._review_contamination = contam
        if nf > 0:
            self._frame_count = nf
            self._review_scale.configure(to=max(nf - 1, 0))
            self._review_spinbox.configure(to=max(nf - 1, 0))
        self._update_status_bar()
        self._render_review(int(self._review_idx_var.get()))

    def _on_review_slider(self, value):
        """Called on every slider tick — snap to an int frame and debounce.

        Change-guarded: writing the var only when the frame actually changes is
        what keeps _sync_review_scale() below loop-free (scale.set fires this
        callback once, sees no change, stops)."""
        try:
            idx = int(round(float(value)))
        except (ValueError, TypeError):
            return
        try:
            if int(self._review_idx_var.get() or 0) == idx:
                return
            self._review_idx_var.set(idx)
        except Exception:
            return
        self._schedule_review_render()

    def _sync_review_scale(self):
        """Move the slider to the spinbox's frame (guarded, see slider callback)."""
        try:
            self._review_scale.set(int(self._review_idx_var.get() or 0))
        except Exception:
            pass

    def _on_review_spinbox(self):
        self._sync_review_scale()
        self._schedule_review_render()

    def _schedule_review_render(self):
        """Debounce rapid slider drags: cancel any pending render and re-schedule."""
        if self._review_after is not None:
            try:
                self.root.after_cancel(self._review_after)
            except Exception:
                pass
        self._review_after = self.root.after(
            120, self._fire_review_render)

    def _fire_review_render(self):
        self._review_after = None
        try:
            idx = int(self._review_idx_var.get())
        except (ValueError, TypeError):
            idx = 0
        self._sync_review_scale()   # typed entry / programmatic changes move the slider too
        self._render_review(idx)

    @staticmethod
    def _review_panel_layout(show_cake: bool, show_contamination: bool):
        """Return ordered review-panel names and their relative heights."""
        panels = ["pattern"]
        ratios = [3]
        if show_cake:
            panels.append("cake")
            ratios.append(2)
        if show_contamination:
            panels.append("contamination")
            ratios.append(1)
        return tuple(panels), tuple(ratios)

    def _render_review(self, frame_index: int):
        """Render the configurable review figure for one frame."""
        # Close previous figure to avoid leaks.
        prev = getattr(self, "_review_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._review_fig = None

        for w in self.review_plot_frame.winfo_children():
            w.destroy()

        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.ttk.Label(
                self.review_plot_frame,
                text="No analysis HDF5 loaded — run analysis or set path on Input tab.",
                style="Muted.TLabel",
            ).pack(anchor="center", expand=True)
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.review_plot_frame,
                text=f"matplotlib unavailable: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import frame_data
        try:
            fd = frame_data(path, frame_index)
        except Exception as e:
            self.ttk.Label(
                self.review_plot_frame,
                text=f"frame_data error: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        if not fd.get("ok"):
            self.ttk.Label(
                self.review_plot_frame,
                text=f"frame_data: {fd.get('error', 'unknown error')}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        radial = fd.get("radial")
        unit = fd.get("unit") or "radial bin"
        x = np.asarray(radial) if radial is not None else None

        # constrained layout recomputes margins on every resize (one-shot
        # tight_layout leaves labels clipped/overlapping when the pane resizes).
        show_cake = bool(self._show_cake.get())
        show_contamination = bool(self._show_contamination.get())
        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._review_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        panels, ratios = self._review_panel_layout(show_cake, show_contamination)
        gs = fig.add_gridspec(len(panels), 1, height_ratios=ratios)
        axes = {name: fig.add_subplot(gs[index]) for index, name in enumerate(panels)}
        ax1 = axes["pattern"]
        ax_cake = axes.get("cake")
        ax2 = axes.get("contamination")

        def _plot(ax, arr, label, color, lw=0.9, alpha=0.85):
            if arr is None:
                return
            y = np.asarray(arr, dtype=float)
            if x is not None and x.shape == y.shape:
                ax.plot(x, y, lw=lw, alpha=alpha, label=label, color=color)
            else:
                ax.plot(y, lw=lw, alpha=alpha, label=label, color=color)

        if self._show_mean.get():
            _plot(ax1, fd.get("mean"), "mean", theme.C.CLR_RAW)
        if self._show_robust.get():
            _plot(ax1, fd.get("robust"), "robust", theme.C.CLR_MSKD)
        if self._show_baseline.get():
            _plot(ax1, fd.get("baseline"), "baseline", theme.C.CLR_SMTH)
        if self._show_clean.get():
            _plot(ax1, fd.get("clean"), "clean", theme.C.ACCENT2)
        if self._show_spot.get():
            _plot(ax1, fd.get("spot_residual"), "spot_residual", theme.C.CLR_DIFF)
        if self._show_residual.get():
            _plot(ax1, fd.get("residual"), "residual", theme.C.CLR_REF, lw=1.0, alpha=0.9)

        # Overlay fitted peaks if requested.
        peaks = fd.get("peaks", [])
        if self._show_peaks.get() and peaks:
            good_centers = [
                p["center"] for p in peaks if p.get("flag", 0) == 0
            ]
            bad_centers = [
                p["center"] for p in peaks if p.get("flag", 0) != 0
            ]
            # Retrieve clean for amplitude reference.
            clean_arr = fd.get("clean")
            y_ref = np.asarray(clean_arr, dtype=float) if clean_arr is not None else None
            for c in good_centers:
                ax1.axvline(c, color=theme.C.ACCENT2, lw=0.7, alpha=0.6)
            for c in bad_centers:
                ax1.axvline(c, color=theme.C.WARN, lw=0.7, alpha=0.5)

        residual_peaks = fd.get("residual_peaks", [])
        if self._show_residual_peaks.get() and residual_peaks:
            for k, p in enumerate(residual_peaks):
                ax1.axvline(
                    p["center"], color=theme.C.CLR_REF, lw=0.9, alpha=0.75,
                    linestyle="--",
                    label="residual peaks" if k == 0 else None,
                )

        unknown_obs = fd.get("unknown_obs", [])
        if self._show_unknowns.get() and unknown_obs:
            for k, p in enumerate(unknown_obs):
                ax1.axvline(
                    p["center"], color=theme.C.WARN, lw=1.0, alpha=0.85,
                    linestyle=":",
                    label="unknown peaks" if k == 0 else None,
                )

        fname = Path(fd.get("filename", "")).name or f"frame {frame_index}"
        ax1.set_title(f"{fname}  [frame {frame_index}]", color=theme.C.FG)
        ax1.set_xlabel(unit_label(unit))
        ax1.set_ylabel(INTENSITY_LABEL)
        if ax1.get_legend_handles_labels()[1]:
            ax1.legend(fontsize=7, framealpha=0.4)
        self._style_ax(ax1)

        # Optional middle axis: the 2D cake (azimuth × radial) for this frame,
        # read from the source reduced file (cakes don't live in the analysis file).
        if ax_cake is not None:
            from .review import cake_for_frame
            ck = cake_for_frame(self._review_source_reduced, frame_index)
            if ck.get("ok") and ck.get("cake") is not None:
                cake = np.asarray(ck["cake"], dtype=float)
                cr, caz = ck.get("radial"), ck.get("azimuthal")
                extent = None
                if cr is not None and caz is not None and cr.size and caz.size:
                    extent = [float(np.min(cr)), float(np.max(cr)),
                              float(np.min(caz)), float(np.max(caz))]
                cc = cake.copy()
                cc[~np.isfinite(cc) | (cc <= 0)] = np.nan
                finite = cc[np.isfinite(cc)]
                vmin = float(np.percentile(finite, 5)) if finite.size else None
                vmax = float(np.percentile(finite, 99)) if finite.size else None
                ax_cake.imshow(cc, aspect="auto", origin="lower", cmap="magma",
                               extent=extent, vmin=vmin, vmax=vmax)
                ax_cake.set_xlabel(unit_label(ck.get("unit") or unit))
                ax_cake.set_ylabel(AZIMUTH_LABEL)
                ax_cake.set_title("Cake (2D)", color=theme.C.FG)
            else:
                ax_cake.set_title(f"Cake: {ck.get('error', 'unavailable')}", color=theme.C.WARN)
                ax_cake.set_xticks([]); ax_cake.set_yticks([])
            self._style_ax(ax_cake)

        # Bottom axis: contamination across series.
        if ax2 is not None:
            contam = self._review_contamination
            if contam is not None and len(contam):
                c_arr = np.asarray(contam, dtype=float)
                ax2.plot(c_arr, lw=0.8, color=theme.C.CLR_DIFF, alpha=0.85)
                ax2.axvline(frame_index, color=theme.C.ACCENT, lw=1.2, alpha=0.8)
                ax2.set_xlabel(FRAME_LABEL)
                ax2.set_ylabel(CONTAMINATION_LABEL)
                ax2.set_title("Contamination vs frame", color=theme.C.FG)
            else:
                ax2.set_title("Contamination (not available)", color=theme.C.FG)
                ax2.set_xlabel(FRAME_LABEL)
            self._style_ax(ax2)

        self._review_canvas = self._embed_figure(self.review_plot_frame, fig)

    def _style_ax(self, ax):
        ax.set_facecolor(theme.C.BG2)
        ax.tick_params(colors=theme.C.FG, which="both")
        ax.xaxis.label.set_color(theme.C.FG)
        ax.yaxis.label.set_color(theme.C.FG)
        ax.title.set_color(theme.C.FG)
        for s in ax.spines.values():
            s.set_edgecolor(theme.C.FG)

    def _toggle_identify_help(self):
        """Show/hide the Identify instructions so the plot can use the space."""
        if self._identify_help_var.get():
            self._identify_help.grid()
        else:
            self._identify_help.grid_remove()

    def _open_plot_window(self, fig, title="Plot"):
        """Open ``fig`` in a separate, large, resizable window with its own
        toolbar. The figure is shared with the inline canvas (these plots are
        static, rendered once per load), so the pop-out is just a bigger view;
        reloading the tab refreshes both."""
        if fig is None:
            self.messagebox.showinfo("Open in window", "Load a plot first.")
            return
        try:
            win = self.tk.Toplevel(self.root)
            win.title(f"SeriesXRD — {title}")
            try:
                win.configure(bg=theme.C.BG)
                win.geometry("1100x800")
            except Exception:
                pass
            embed_figure(
                win,
                fig,
                self.root,
                toolbar_factory=self._add_nav_toolbar,
            )
        except Exception as e:
            self.messagebox.showerror("Open in window", f"Could not open: {e!r}")

    def _attach_hover(self, canvas, status_label):
        """Live cursor read-out into ``status_label``, restoring its text on
        leave. Call AFTER the label's base text is set. The x value is shown
        as-is (frame index, pressure, ... — whatever the plot's x-axis is)."""
        if canvas is None or status_label is None:
            return
        base = {"text": status_label.cget("text")}
        def _move(event):
            if event.inaxes is not None and event.xdata is not None:
                if event.ydata is not None:
                    status_label.configure(
                        text=f"{base['text']}   |   x={event.xdata:.6g}, y={event.ydata:.4g}")
                else:
                    status_label.configure(text=f"{base['text']}   |   x={event.xdata:.6g}")
        def _leave(event):
            status_label.configure(text=base["text"])
        try:
            canvas.mpl_connect("motion_notify_event", _move)
            canvas.mpl_connect("axes_leave_event", _leave)
        except Exception:
            pass

    def _style_colorbar(self, cb):
        """Recolour a colorbar to the dark palette — its label, ticks, and
        outline default to black and vanish against the figure background."""
        try:
            cb.ax.tick_params(colors=theme.C.FG, which="both")
            cb.ax.yaxis.label.set_color(theme.C.FG)
            cb.ax.xaxis.label.set_color(theme.C.FG)
            if cb.outline is not None:
                cb.outline.set_edgecolor(theme.C.FG)
        except Exception:
            pass

    def _embed_figure(self, parent, fig, toolbar=True):
        """Embed a matplotlib figure so it tracks the pane size instead of forcing it.

        A ttk.Notebook sizes itself to its largest tab, so a fixed-size canvas
        (figsize×dpi ≈ 700–800 px) would pin the whole window to at least that
        size — the plot then loads larger than the GUI and can't shrink. Giving
        the canvas widget a tiny *requested* size removes that floor; fill+expand
        grows it to fill the pane. The shared embedding helper waits until the
        page is mapped, then fits and draws at the allocated size; constrained
        layout re-flows the margins on each later resize.

        A navigation toolbar (home / pan / box-zoom / save) is packed beneath the
        canvas so dense patterns can be zoomed into without resizing the window.
        Returns the canvas.
        """
        toolbar_factory = self._add_nav_toolbar if toolbar else None
        return embed_figure(
            parent,
            fig,
            self.root,
            toolbar_factory=toolbar_factory,
        )

    def _add_nav_toolbar(self, canvas, parent):
        """Add a dark-styled matplotlib navigation toolbar below an embedded
        canvas (pan / box-zoom / home / save). Degrades silently if the toolbar
        backend is unavailable."""
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            tb.update()
            # matplotlib's toolbar glyphs are dark, so painting the buttons the
            # window background (near-black) hides them. Give the buttons a light
            # fill so the icons read clearly; keep the frame + coordinate label
            # on the dark palette.
            try:
                tb.configure(background=theme.C.BG)
                for child in tb.winfo_children():
                    cls = child.winfo_class()
                    try:
                        if cls in ("Button", "Checkbutton", "Radiobutton"):
                            child.configure(background=theme.C.FG, activebackground=theme.C.ACCENT,
                                            highlightbackground=theme.C.BG, relief="flat")
                        elif cls == "Label":
                            child.configure(background=theme.C.BG, foreground=theme.C.FG)
                        else:
                            child.configure(background=theme.C.BG)
                    except Exception:
                        pass
            except Exception:
                pass
            tb.pack(side="bottom", fill="x")
            return tb
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Tab 9 — Peak map (fitted peak positions across the series)
    # ------------------------------------------------------------------

    def _tab_heatmap(self, frame):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="Load peak map", command=self.load_heatmap).pack(
            side="left", padx=4)
        self._heatmap_good_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="Good peaks only", variable=self._heatmap_good_only,
            command=self.load_heatmap,
        ).pack(side="left", padx=8)
        ttk.Label(top, text="Color by:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._heatmap_color_by = tk.StringVar(value="area")
        ttk.Combobox(
            top, textvariable=self._heatmap_color_by,
            values=["area", "amplitude", "fwhm"],
            width=10, state="readonly",
        ).pack(side="left", padx=2)
        ttk.Label(top, text="X axis:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._heatmap_xaxis = ttk.Combobox(
            top, values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        self._heatmap_xaxis.set("frame")
        self._heatmap_xaxis.pack(side="left", padx=2)
        self._heatmap_xaxis.bind("<<ComboboxSelected>>",
                                 lambda e: self.load_heatmap())
        ttk.Button(top, text="Refresh", command=self.load_heatmap).pack(
            side="left", padx=4)
        ttk.Button(top, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_heatmap_fig", None), "Peak map")
                   ).pack(side="left", padx=4)

        self.heatmap_status = ttk.Label(top, text="", style="Muted.TLabel")
        self.heatmap_status.pack(side="left", padx=12)

        tools = ttk.Frame(frame)
        tools.pack(fill="x", pady=(0, 4))
        ttk.Button(
            tools, text="Calculate crystallite size / strain…",
            command=self.run_microstructure_clicked,
        ).pack(side="left", padx=4)
        ttk.Label(
            tools,
            text="Williamson–Hall analysis of accepted peak widths",
            style="Muted.TLabel",
        ).pack(side="left", padx=6)

        self.heatmap_plot_frame = ttk.Frame(frame)
        self.heatmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.heatmap_plot_frame,
            text="Load the analysis HDF5 to display the peak map.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

    def run_microstructure_clicked(self):
        """Run a Williamson–Hall analysis with an optional instrument width."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror(
                "Size / strain", "Run peak fitting before calculating size and strain.")
            return
        from tkinter import simpledialog
        raw = simpledialog.askstring(
            "Instrument broadening",
            "Instrument FWHM in q (Å⁻¹). Leave blank for an exploratory, "
            "uncorrected estimate:",
            parent=self.root,
        )
        if raw is None:
            return
        instrument_width = None
        if raw.strip():
            try:
                instrument_width = float(raw)
                if instrument_width <= 0:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror(
                    "Size / strain", "Instrument FWHM must be a positive number or blank.")
                return
        elif not self.messagebox.askyesno(
                "Uncorrected estimate",
                "Without an instrument profile, size is a lower bound and strain is "
                "an upper bound. Continue for exploratory use?"):
            return
        try:
            from .microstructure import williamson_hall
            result = williamson_hall(path, instrument_fwhm_q=instrument_width)
        except Exception as exc:
            self.messagebox.showerror("Size / strain", str(exc))
            return
        fitted = sum(int(n) >= 5 for n in result.get("n_peaks", []))
        qualifier = "instrument-corrected" if result.get("instrument_corrected") else "uncorrected"
        self.log(f"Williamson–Hall analysis saved: {fitted} frames ({qualifier})")
        self.notify(f"Size/strain: saved {qualifier} estimates for {fitted} "
                    f"frame(s) to /microstructure.")

    def load_heatmap(self):
        """Render the peak-map scatter plot from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        prev = getattr(self, "_heatmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._heatmap_fig = None

        for w in self.heatmap_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.colors as mcolors
        except Exception as e:
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"matplotlib unavailable: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import peak_map
        good_only = bool(self._heatmap_good_only.get())
        try:
            pm = peak_map(path, good_only=good_only)
        except Exception as e:
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"peak_map error: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        if not pm.get("ok"):
            err = pm.get("error", "unknown error")
            self.ttk.Label(
                self.heatmap_plot_frame,
                text=f"Peak map: {err}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            if hasattr(self, "heatmap_status"):
                self.heatmap_status.configure(text=err)
            return

        frame_arr = np.asarray(pm["frame"], dtype=float)
        center_arr = np.asarray(pm["center"], dtype=float)
        color_by = str(self._heatmap_color_by.get())
        c_arr = np.asarray(pm.get(color_by, pm["area"]), dtype=float)

        # Map the frame index onto the chosen independent variable.
        x_kind = getattr(self._heatmap_xaxis, "get", lambda: "frame")() or "frame"
        x_arr, x_label = frame_arr, "Frame index"
        if x_kind != "frame":
            from .heatmap import series_axis
            sx = series_axis(path, x_kind)
            if not sx["ok"]:
                self.ttk.Label(self.heatmap_plot_frame, text=sx["error"],
                               style="Warn.TLabel").pack(anchor="center", expand=True)
                if hasattr(self, "heatmap_status"):
                    self.heatmap_status.configure(text=sx["error"])
                return
            xv = np.asarray(sx["x"], dtype=float)
            idx = frame_arr.astype(int)
            ok = (idx >= 0) & (idx < xv.size)
            x_arr = np.full(frame_arr.size, np.nan)
            x_arr[ok] = xv[idx[ok]]
            x_label = sx["label"]
            keep = np.isfinite(x_arr)
            x_arr, center_arr, c_arr = x_arr[keep], center_arr[keep], c_arr[keep]

        n_pts = int(center_arr.size)
        if hasattr(self, "heatmap_status"):
            self.heatmap_status.configure(
                text=f"{n_pts} peaks plotted" + (" (good only)" if good_only else ""))

        unit = pm.get("unit") or "radial"

        fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self._heatmap_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if n_pts == 0:
            ax.set_title("No peaks to display", color=theme.C.FG)
        else:
            # Log-safe normalisation for area / amplitude; linear for fwhm.
            if color_by in ("area", "amplitude"):
                pos = c_arr[c_arr > 0]
                if pos.size > 0:
                    vmin = float(pos.min())
                    vmax = float(c_arr.max())
                    norm = mcolors.LogNorm(vmin=max(vmin, 1e-9), vmax=max(vmax, vmin + 1e-9))
                else:
                    norm = None
            else:
                norm = None

            # Larger markers with a light edge so even dark-coloured (low-value)
            # points read against the dark axes background.
            sc = ax.scatter(
                x_arr, center_arr, c=c_arr,
                cmap="viridis", s=28, alpha=0.9, norm=norm,
                edgecolors=theme.C.FG, linewidths=0.4,
            )
            try:
                cb = fig.colorbar(sc, ax=ax, label=color_by)
                self._style_colorbar(cb)
            except Exception:
                pass
            ax.set_xlabel(x_label, color=theme.C.FG)
            ax.set_ylabel(f"Peak center — {unit_label(unit)}", color=theme.C.FG)
            ax.set_title(f"Peak map — {n_pts} peaks", color=theme.C.FG)

        self._heatmap_canvas = self._embed_figure(self.heatmap_plot_frame, fig)
        self._attach_hover(self._heatmap_canvas, self.heatmap_status)

    # ------------------------------------------------------------------
    # Tab 7 — Phases (reference-phase library)
    # ------------------------------------------------------------------

    def _phases_workspace(self) -> Path:
        """Return the workspace dir for the user phase library."""
        ws = self.config.get("workspace_root")
        if ws:
            return Path(ws)
        return self.config_path.parent

    def _tab_phases(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Button(ctrl, text="Import CIF…", command=self.import_cif_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Add phase…",
                   command=lambda: self._phase_dialog(None)).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Edit…", command=self.edit_phase_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Remove", command=self.remove_phase_clicked).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="Refresh", command=self.load_phases_table).pack(
            side="left", padx=4)
        self._phases_status = ttk.Label(ctrl, text="", style="Muted.TLabel")
        self._phases_status.pack(side="right", padx=8)

        # pymatgen availability hint
        self._phases_pymatgen_label = ttk.Label(frame, text="", style="Muted.TLabel",
                                                wraplength=800, justify="left")
        self._phases_pymatgen_label.pack(fill="x", padx=4, pady=(0, 2))

        # Treeview with scrollbar
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        cols = ("enabled", "name", "category", "spacegroup", "K0", "K0p", "source")
        self.phases_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                        selectmode="browse")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.phases_tree.yview)
        self.phases_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.phases_tree.pack(side="left", fill="both", expand=True)

        self.phases_tree.heading("enabled",     text="✓")
        self.phases_tree.heading("name",        text="Name")
        self.phases_tree.heading("category",    text="Category")
        self.phases_tree.heading("spacegroup",  text="Space group")
        self.phases_tree.heading("K0",          text="K0 (GPa)")
        self.phases_tree.heading("K0p",         text="K0'")
        self.phases_tree.heading("source",      text="Source/Origin")

        self.phases_tree.column("enabled",    width=36,  minwidth=32,  anchor="center", stretch=False)
        self.phases_tree.column("name",       width=140, minwidth=80)
        self.phases_tree.column("category",   width=80,  minwidth=60)
        self.phases_tree.column("spacegroup", width=100, minwidth=60)
        self.phases_tree.column("K0",         width=80,  minwidth=50,  anchor="center")
        self.phases_tree.column("K0p",        width=60,  minwidth=40,  anchor="center")
        self.phases_tree.column("source",     width=260, minwidth=100)

        self.phases_tree.tag_configure("user", foreground=theme.C.ACCENT)

        self.phases_tree.bind("<Button-1>", self._phases_tree_click)
        self.phases_tree.bind("<Double-1>", self.edit_phase_clicked)

        self._phases_by_name: "Dict[str, Any]" = {}

        self.load_phases_table()

    def load_phases_table(self):
        from .phases import list_phases, pymatgen_available
        ws = self._phases_workspace()
        phases = list_phases(ws)
        self._phases_by_name = {p.name: p for p in phases}

        enabled_set = set(str(n) for n in self.config.get("candidate_phases", []))

        # Clear and repopulate
        self.phases_tree.delete(*self.phases_tree.get_children())
        for p in phases:
            eos = p.eos or {}
            k0_val = eos.get("K0")
            k0p_val = eos.get("K0p")
            k0_str = f"{k0_val:g}" if k0_val is not None else "—"
            k0p_str = f"{k0p_val:g}" if k0p_val is not None else "—"
            if p.builtin:
                origin = p.source or "bundled"
            else:
                origin = "(user)" if not p.source else f"(user) {p.source}"
            enabled_mark = "✓" if p.name in enabled_set else ""
            tags = () if p.builtin else ("user",)
            self.phases_tree.insert(
                "", "end", iid=p.name,
                values=(enabled_mark, p.name, p.category,
                        p.space_group or "—", k0_str, k0p_str, origin),
                tags=tags,
            )

        n_total = len(phases)
        n_user = sum(1 for p in phases if not p.builtin)
        n_builtin = n_total - n_user
        n_enabled = len([n for n in enabled_set if n in self._phases_by_name])
        if hasattr(self, "_phases_status"):
            self._phases_status.configure(
                text=(f"{n_total} phases ({n_user} user, {n_builtin} bundled)"
                      f"  ·  {n_enabled} enabled"))

        if hasattr(self, "_phases_pymatgen_label"):
            if pymatgen_available():
                self._phases_pymatgen_label.configure(
                    text="pymatgen available — CIF auto-parsing and pattern simulation enabled.",
                    style="Muted.TLabel")
            else:
                self._phases_pymatgen_label.configure(
                    text=("pymatgen not installed — CIF auto-parsing & pattern simulation "
                          "disabled (pip install pymatgen). You can still add phases manually."),
                    style="Warn.TLabel")
        self._refresh_gridmap_values()

    def _phases_tree_click(self, event):
        col = self.phases_tree.identify_column(event.x)
        if col == "#1":
            row = self.phases_tree.identify_row(event.y)
            if row:
                self._toggle_phase_enabled(row)
        # Otherwise let normal selection happen (no return / no break needed)

    def _toggle_phase_enabled(self, name: str):
        enabled = list(self.config.get("candidate_phases", []))
        enabled_strs = [str(n) for n in enabled]
        if name in enabled_strs:
            enabled_strs.remove(name)
        else:
            enabled_strs.append(name)
        self.config["candidate_phases"] = sorted(set(enabled_strs))
        self.save_config(silent=True)
        # Update just this row's enabled cell
        mark = "✓" if name in self.config["candidate_phases"] else ""
        try:
            self.phases_tree.set(name, "enabled", mark)
        except Exception:
            pass
        # Refresh status count
        enabled_set = set(self.config["candidate_phases"])
        n_enabled = len([n for n in enabled_set if n in self._phases_by_name])
        n_total = len(self._phases_by_name)
        n_user = sum(1 for p in self._phases_by_name.values() if not p.builtin)
        n_builtin = n_total - n_user
        if hasattr(self, "_phases_status"):
            self._phases_status.configure(
                text=(f"{n_total} phases ({n_user} user, {n_builtin} bundled)"
                      f"  ·  {n_enabled} enabled"))
        self.log(f"Phase '{name}' {'enabled' if mark else 'disabled'} as candidate.")
        self._refresh_gridmap_values()

    def _phase_dialog(self, existing):
        """Add (existing=None) or Edit (existing=Phase) a user phase."""
        from .phases import Phase, upsert_user_phase, CATEGORIES
        tk, ttk = self.tk, self.ttk

        title = "Edit phase" if existing else "Add phase"
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=theme.C.BG)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        def _f(s):
            """Parse float leniently; return None on blank/invalid."""
            try:
                v = str(s).strip()
                return float(v) if v else None
            except (ValueError, TypeError):
                return None

        pad = {"padx": 6, "pady": 3}

        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)

        row = 0
        # Name
        ttk.Label(content, text="Name").grid(row=row, column=0, sticky="w", **pad)
        v_name = tk.StringVar(value=existing.name if existing else "")
        ttk.Entry(content, textvariable=v_name, width=36).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Formula
        ttk.Label(content, text="Formula").grid(row=row, column=0, sticky="w", **pad)
        v_formula = tk.StringVar(value=existing.formula if existing else "")
        ttk.Entry(content, textvariable=v_formula, width=36).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Category
        ttk.Label(content, text="Category").grid(row=row, column=0, sticky="w", **pad)
        v_category = tk.StringVar(
            value=(existing.category if existing else "marker"))
        ttk.Combobox(content, textvariable=v_category,
                     values=list(CATEGORIES), state="readonly", width=16).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        # Space group
        ttk.Label(content, text="Space group").grid(row=row, column=0, sticky="w", **pad)
        v_sg = tk.StringVar(value=existing.space_group if existing else "")
        ttk.Entry(content, textvariable=v_sg, width=20).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        # Lattice — six small entries in one row
        ttk.Label(content, text="Lattice (Å, °)").grid(
            row=row, column=0, sticky="w", **pad)
        lat_frame = ttk.Frame(content)
        lat_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        lat_keys = ("a", "b", "c", "alpha", "beta", "gamma")
        lat_defaults = {"alpha": "90", "beta": "90", "gamma": "90"}
        lat_vars: "Dict[str, tk.StringVar]" = {}
        for i, k in enumerate(lat_keys):
            ex_val = ""
            if existing and existing.lattice:
                v_raw = existing.lattice.get(k)
                ex_val = f"{v_raw:g}" if v_raw is not None else ""
            if not ex_val:
                ex_val = lat_defaults.get(k, "")
            ttk.Label(lat_frame, text=k).grid(row=0, column=i * 2, sticky="e",
                                               padx=(6 if i else 0, 1))
            sv = tk.StringVar(value=ex_val)
            ttk.Entry(lat_frame, textvariable=sv, width=8).grid(
                row=0, column=i * 2 + 1, padx=(0, 4))
            lat_vars[k] = sv
        row += 1

        # EOS
        from .phases import EOS_TYPES, _eos_norm_type
        ttk.Label(content, text="EOS").grid(row=row, column=0, sticky="w", **pad)
        eos_frame = ttk.Frame(content)
        eos_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        ex_eos = (existing.eos or {}) if existing else {}
        ttk.Label(eos_frame, text="type").grid(row=0, column=0, sticky="e", padx=(0, 1))
        v_eos_type = tk.StringVar(value=_eos_norm_type(ex_eos) if ex_eos else "BM3")
        ttk.Combobox(eos_frame, textvariable=v_eos_type, values=list(EOS_TYPES),
                     state="readonly", width=10).grid(row=0, column=1, padx=(0, 4))
        # K0'' only used by BM4; left blank otherwise.
        eos_keys = ("V0", "K0", "K0p", "K0pp")
        eos_labels = ("V0 (Å³)", "K0 (GPa)", "K0'", "K0'' (1/GPa, BM4)")
        eos_vars: "Dict[str, tk.StringVar]" = {}
        for i, (k, lbl) in enumerate(zip(eos_keys, eos_labels)):
            v_raw = ex_eos.get(k)
            ex_val = f"{v_raw:g}" if v_raw is not None else ""
            ttk.Label(eos_frame, text=lbl).grid(row=1, column=i * 2,
                                                sticky="e", padx=(6 if i else 0, 1))
            sv = tk.StringVar(value=ex_val)
            ttk.Entry(eos_frame, textvariable=sv, width=10).grid(
                row=1, column=i * 2 + 1, padx=(0, 4))
            eos_vars[k] = sv
        ttk.Label(eos_frame,
                  text="V0 optional (cancels in scaling); only K0 is required. "
                       "Forms: BM2/BM3/BM4, Vinet, Murnaghan.",
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=8, sticky="w", pady=(2, 0))
        row += 1

        # Axial (anisotropic) EOS — optional, per-axis K0/K0' for non-cubic phases.
        ttk.Label(content, text="Axial EOS").grid(row=row, column=0, sticky="nw", **pad)
        ax_frame = ttk.Frame(content)
        ax_frame.grid(row=row, column=1, columnspan=3, sticky="w", **pad)
        ex_ax = (existing.axial_eos or {}) if existing else {}
        ttk.Label(ax_frame, text="axis").grid(row=0, column=0, padx=(0, 4))
        ttk.Label(ax_frame, text="K0 (GPa)").grid(row=0, column=1, padx=2)
        ttk.Label(ax_frame, text="K0'").grid(row=0, column=2, padx=2)
        axial_vars: "Dict[str, tuple]" = {}
        for i, axis in enumerate(("a", "b", "c")):
            e = ex_ax.get(axis) if isinstance(ex_ax.get(axis), dict) else {}
            k0 = e.get("K0"); kp = e.get("K0p")
            ttk.Label(ax_frame, text=axis).grid(row=i + 1, column=0, sticky="e", padx=(0, 4))
            v_k0 = tk.StringVar(value=f"{k0:g}" if k0 is not None else "")
            v_kp = tk.StringVar(value=f"{kp:g}" if kp is not None else "")
            ttk.Entry(ax_frame, textvariable=v_k0, width=10).grid(row=i + 1, column=1, padx=2)
            ttk.Entry(ax_frame, textvariable=v_kp, width=8).grid(row=i + 1, column=2, padx=2)
            axial_vars[axis] = (v_k0, v_kp)
        ttk.Label(ax_frame,
                  text="Optional. Fill per-axis K0 (on the cubed axis length, "
                       "PASCal/EosFit convention) for anisotropic compression; "
                       "blank axes fall back to the volume EOS. b inherits a if equal.",
                  style="Muted.TLabel", wraplength=420, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))
        row += 1

        # Source
        ttk.Label(content, text="Source").grid(row=row, column=0, sticky="w", **pad)
        v_source = tk.StringVar(value=existing.source if existing else "")
        ttk.Entry(content, textvariable=v_source, width=50).grid(
            row=row, column=1, columnspan=3, sticky="we", **pad)
        row += 1

        # Notes
        ttk.Label(content, text="Notes").grid(row=row, column=0, sticky="nw", **pad)
        notes_text = tk.Text(content, width=50, height=3, bg=theme.C.BG2, fg=theme.C.FG,
                             insertbackground=theme.C.FG, relief="flat", wrap="word")
        notes_text.grid(row=row, column=1, columnspan=3, sticky="we", **pad)
        if existing and existing.notes:
            notes_text.insert("1.0", existing.notes)
        row += 1

        content.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=row, column=0, columnspan=4, sticky="e", pady=(8, 0))

        def _save():
            name = v_name.get().strip()
            if not name:
                self.messagebox.showerror("Validation", "Name is required.",
                                          parent=dlg)
                return

            lattice = {}
            for k in lat_keys:
                fv = _f(lat_vars[k].get())
                if fv is not None:
                    lattice[k] = fv

            eos: "Dict[str, Any]" = {"type": v_eos_type.get() or "BM3"}
            for k in eos_keys:
                fv = _f(eos_vars[k].get())
                if fv is not None:
                    eos[k] = fv

            # Per-axis EOS (only axes with a K0 entered); same form as the main EOS.
            axial_eos: "Dict[str, Any]" = {}
            for axis, (v_k0, v_kp) in axial_vars.items():
                k0 = _f(v_k0.get())
                if k0 is not None:
                    ae = {"type": v_eos_type.get() or "BM3", "K0": k0}
                    kp = _f(v_kp.get())
                    if kp is not None:
                        ae["K0p"] = kp
                    axial_eos[axis] = ae

            notes = notes_text.get("1.0", "end-1c")

            phase = Phase(
                name=name,
                formula=v_formula.get().strip(),
                category=v_category.get(),
                space_group=v_sg.get().strip(),
                lattice=lattice,
                atoms=(existing.atoms if existing else []),
                eos=eos,
                axial_eos=axial_eos,
                amorphous=(existing.amorphous if existing else False),
                cif_path=(existing.cif_path if existing else ""),
                source=v_source.get().strip(),
                notes=notes,
            )
            try:
                upsert_user_phase(self._phases_workspace(), phase)
            except Exception as e:
                self.messagebox.showerror("Save failed", repr(e), parent=dlg)
                return
            dlg.destroy()
            self.load_phases_table()
            action = "updated" if existing else "added"
            self.log(f"Phase '{name}' {action}.")

        ttk.Button(btn_frame, text="Save", command=_save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=dlg.destroy).pack(side="left", padx=4)

    def edit_phase_clicked(self, event=None):
        sel = self.phases_tree.selection()
        if not sel:
            self.messagebox.showinfo("Edit phase", "Select a phase to edit.")
            return
        name = sel[0]
        phase = self._phases_by_name.get(name)
        if phase is None:
            self.messagebox.showinfo("Edit phase", f"Phase '{name}' not found.")
            return
        self._phase_dialog(phase)

    def remove_phase_clicked(self):
        from .phases import remove_user_phase
        sel = self.phases_tree.selection()
        if not sel:
            self.messagebox.showinfo("Remove phase", "Select a phase to remove.")
            return
        name = sel[0]
        if not self.messagebox.askyesno(
                "Remove phase",
                f"Remove user phase '{name}'? This cannot be undone."):
            return
        ws = self._phases_workspace()
        removed = remove_user_phase(ws, name)
        if not removed:
            self.messagebox.showinfo(
                "Remove phase",
                f"'{name}' is a bundled phase and cannot be deleted.\n"
                "Use Edit to create a user override.")
            return
        # Drop from candidate_phases if present
        enabled = list(self.config.get("candidate_phases", []))
        if name in enabled:
            enabled.remove(name)
            self.config["candidate_phases"] = sorted(set(enabled))
            self.save_config(silent=True)
        self.log(f"Phase '{name}' removed.")
        self.load_phases_table()

    def import_cif_clicked(self):
        from .phases import import_cif, pymatgen_available
        path = self.filedialog.askopenfilename(
            title="Import CIF",
            filetypes=[("CIF", "*.cif"), ("All files", "*.*")],
        )
        if not path:
            return
        ws = self._phases_workspace()
        try:
            phase = import_cif(ws, path)
        except Exception as e:
            self.messagebox.showerror("Import CIF failed", repr(e))
            return
        self.log(f"CIF imported: {path} -> phase '{phase.name}'")
        self.load_phases_table()
        if not pymatgen_available():
            self.messagebox.showinfo(
                "CIF imported",
                "The CIF was stored but could not be auto-parsed (pymatgen is not "
                "installed). Install pymatgen for automatic lattice/structure parsing, "
                "or fill in the lattice and EOS fields manually below.")
        # Always open the edit dialog so the user can fill in / verify the EOS
        self._phase_dialog(self._phases_by_name.get(phase.name, phase))

    # ------------------------------------------------------------------
    # Tab 8 — Frame metadata (pressure prior)
    # ------------------------------------------------------------------

    def _tab_frame_metadata(self, frame):
        ttk = self.ttk

        ttk.Label(
            frame, text="Frame metadata — series conditions (P, T)",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w", padx=6, pady=(4, 0))
        _fm_sub = ttk.Label(
            frame,
            text=(
                "Per-frame pressure and temperature feed the Step-3 prior and "
                "the series plots. Populate them here (filenames, CSV, or by hand)."
            ),
            style="Muted.TLabel", justify="left", wraplength=760,
        )
        _fm_sub.pack(anchor="w", padx=6, pady=(0, 6))
        self.autowrap(_fm_sub)

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Button(ctrl, text="Extract from filenames",
                   command=self.extract_pressures_clicked).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Import CSV…",
                   command=self.import_pressure_csv_clicked).pack(side="left", padx=4)
        _hdr_btn = ttk.Button(ctrl, text="Read X/Y from headers…",
                              command=self.import_positions_clicked)
        _hdr_btn.pack(side="left", padx=4)
        _ToolTip(_hdr_btn, (
            "Mapping scans: read per-frame stage positions from the raw frame "
            "files' headers (EDF/CBF motor entries) into /frames/pos_x, pos_y — "
            "the Grid map's 'coordinates' layout then places frames "
            "automatically."))
        ttk.Button(ctrl, text="Preview pressure vs frame",
                   command=self.preview_pressure_clicked).pack(side="left", padx=4)

        self._fm_status = ttk.Label(frame, text="", style="Muted.TLabel")
        self._fm_status.pack(anchor="w", padx=6, pady=(0, 4))

        # Editable per-frame metadata table
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="x", padx=6, pady=(0, 4))

        fm_cols = ("frame", "file", "pressure", "sigma", "temp", "src")
        self._fm_table = ttk.Treeview(tree_frame, columns=fm_cols, show="headings",
                                      height=10, selectmode="extended")
        fm_vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                               command=self._fm_table.yview)
        self._fm_table.configure(yscrollcommand=fm_vsb.set)
        fm_vsb.pack(side="right", fill="y")
        self._fm_table.pack(side="left", fill="both", expand=True)

        for c, txt, w, anc, stretch in (
            ("frame", "Frame", 60, "center", False),
            ("file", "Filename", 220, "w", True),
            ("pressure", "P (GPa)", 80, "center", False),
            ("sigma", "σ (GPa)", 80, "center", False),
            ("temp", "T (K)", 80, "center", False),
            ("src", "Src", 50, "center", False),
        ):
            self._fm_table.heading(c, text=txt)
            self._fm_table.column(c, width=w, minwidth=40, anchor=anc, stretch=stretch)
        self._fm_table.tag_configure("user", foreground=theme.C.ACCENT)

        # Editor row
        editor = ttk.Frame(frame)
        editor.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(editor, text="P (GPa):", style="Muted.TLabel").pack(side="left", padx=(0, 2))
        self._fm_edit_p = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_p, width=10).pack(side="left", padx=(0, 8))
        ttk.Label(editor, text="σ:", style="Muted.TLabel").pack(side="left", padx=(0, 2))
        self._fm_edit_sig = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_sig, width=10).pack(side="left", padx=(0, 8))
        ttk.Label(editor, text="T (K):", style="Muted.TLabel").pack(side="left", padx=(0, 2))
        self._fm_edit_t = self.tk.StringVar(value="")
        ttk.Entry(editor, textvariable=self._fm_edit_t, width=10).pack(side="left", padx=(0, 8))
        ttk.Button(editor, text="Apply to selected",
                   command=self.fm_apply_selected_clicked).pack(side="left", padx=4)
        ttk.Button(editor, text="Refresh table",
                   command=self.fm_refresh_table_clicked).pack(side="left", padx=4)
        _exp_btn = ttk.Button(editor, text="Export selected…",
                              command=self.fm_export_selected_clicked)
        _exp_btn.pack(side="left", padx=(12, 4))
        _ToolTip(_exp_btn, (
            "Export the selected frame(s): the chosen reduction/fit channel "
            "as two-column .xy patterns (native axis always; 2θ too when the "
            "wavelength is known) and, optionally, every fitted peak of those "
            "frames as one peaks.csv (center/amplitude/fwhm ± esd, eta, area, "
            "chi2, flag, phase). Rietveld hand-off is the separate "
            "seriesxrd-export-refinement bundle."))

        _fm_hint = ttk.Label(
            frame,
            text=("Select frame(s), enter values (blank = leave unchanged), Apply. "
                  "Applied values are marked 'user' — filename re-parsing and "
                  "Step-1 re-runs will not overwrite them."),
            style="Muted.TLabel", wraplength=760, justify="left",
        )
        _fm_hint.pack(anchor="w", padx=6, pady=(0, 4))
        self.autowrap(_fm_hint)

        self.fm_plot_frame = ttk.Frame(frame)
        self.fm_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.fm_plot_frame,
            text="Extract from filenames or Import CSV to populate frame pressures.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

        _fm_csv = ttk.Label(
            frame,
            text=(
                "CSV columns: `frame` (0-based) or `filename`, plus any of "
                "`pressure_gpa`, `pressure_sigma_gpa`, `temperature_K`, "
                "`pos_x_mm`, `pos_y_mm`. Step 1 also auto-parses pressures from "
                "filenames (e.g. sample-1p5GPa → 1.5). Use this tab to override "
                "those or to enter gauge readings (ruby, membrane, thermocouple)."
            ),
            style="Muted.TLabel", justify="left", wraplength=700,
        )
        _fm_csv.pack(anchor="w", padx=6, pady=(4, 4))
        self.autowrap(_fm_csv)

    def extract_pressures_clicked(self):
        from .frame_metadata import extract_to_analysis, import_csv_to_analysis, read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        try:
            result = extract_to_analysis(path)
            summary = result.get("summary", {})
            n_parsed = summary.get("n_parsed", 0)
            n_frames = summary.get("n_frames", 0)
            p_min = summary.get("p_min")
            p_max = summary.get("p_max")
            p_range = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None
                else "P unknown"
            )
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=(
                        f"Parsed {n_parsed}/{n_frames} frames from filenames "
                        f"({p_range})."
                    )
                )
            self._draw_pressure_preview(path)
            try:
                self.fm_refresh_table_clicked()
            except Exception:
                pass
        except Exception as e:
            self.log(f"extract_to_analysis failed: {e!r}", "WARN")
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))

    def import_pressure_csv_clicked(self):
        from .frame_metadata import extract_to_analysis, import_csv_to_analysis, read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        csv_path = self.filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if not csv_path:
            return
        try:
            result = import_csv_to_analysis(path, csv_path)
            summary = result.get("summary", {})
            csv_info = result.get("csv", {})
            n_parsed = summary.get("n_parsed", 0)
            n_frames = summary.get("n_frames", 0)
            p_min = summary.get("p_min")
            p_max = summary.get("p_max")
            p_range = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None
                else "P unknown"
            )
            cols = csv_info.get("columns", [])
            n_rows = csv_info.get("n_rows", 0)
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=(
                        f"Imported {n_rows}-row CSV ({', '.join(cols)}): "
                        f"{n_parsed}/{n_frames} frames have pressure ({p_range})."
                    )
                )
            self._draw_pressure_preview(path)
            try:
                self.fm_refresh_table_clicked()
            except Exception:
                pass
        except Exception as e:
            self.log(f"import_csv_to_analysis failed: {e!r}", "WARN")
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))

    def _do_export_frames(self, indices, out_dir, *, source="fit", peaks=True,
                          residual_unknowns=True, stack=False,
                          stack_style="panels", exclude_d=None,
                          fig_preset="screen", fig_format="",
                          status_label=None):
        """Run the frame export (patterns + optional peaks.csv + optional
        stacked figure). Returns the manifest, or None on failure."""
        status = status_label or getattr(self, "_fm_status", None)
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            if status is not None:
                status.configure(text="Run Step 1 first (no analysis file yet).")
            return None
        from .refine_export import export_frames
        try:
            man = export_frames(path, out_dir, frames=indices,
                                source=source, peaks=peaks,
                                residual_peaks=residual_unknowns,
                                unknowns=residual_unknowns,
                                exclude_d=exclude_d)
        except Exception as e:
            self.log(f"Frame export failed: {e!r}", "WARN")
            if status is not None:
                status.configure(text=f"Export failed: {e}")
            return None
        msg = (f"Exported {man['n_frames']} frame(s) ({man['source']}) "
               f"+ {man['n_peaks']} peak row(s)")
        extra = []
        if man.get("n_residual_peaks"):
            extra.append(f"{man['n_residual_peaks']} residual peak row(s)")
        if man.get("n_unknown_obs"):
            extra.append(f"{man['n_unknown_obs']} unknown row(s)")
        if extra:
            msg += " + " + " + ".join(extra)
        if stack:
            try:
                from .stackplot import stack_figure, FIGURE_PRESETS
                preset = FIGURE_PRESETS.get(fig_preset,
                                            FIGURE_PRESETS["screen"])
                fmt = (fig_format or preset["format"]).lstrip(".")
                out_name = f"stack.{fmt}"
                sman = stack_figure(path, Path(out_dir) / out_name,
                                    source=source, frames=indices,
                                    style=stack_style, exclude_d=exclude_d,
                                    dpi=int(preset["dpi"]),
                                    palette=preset.get("palette"),
                                    background=preset.get("background"))
                extra_msg = (f" + {out_name} ({sman['n_panels']} "
                             f"{sman['style']} panels, {fig_preset} preset)")
            except Exception as e:
                self.log(f"Stacked figure failed: {e!r}", "WARN")
                extra_msg = " (stacked figure FAILED, see log)"
            msg += extra_msg
        # Provenance sidecar: what produced this export, from which file, with
        # which settings — so a figure folder found months later explains itself.
        try:
            from ..core.provenance import provenance_report
            sidecar = Path(out_dir) / "export_provenance.txt"
            settings = (f"source={source}  frames={list(indices)}\n"
                        f"exclude_d={exclude_d}  stack={stack} "
                        f"style={stack_style} preset={fig_preset}\n")
            sidecar.write_text(provenance_report(path) + "\n\nExport settings:\n"
                               + settings, encoding="utf-8")
        except Exception as e:
            self.log(f"Provenance sidecar failed: {e!r}", "WARN")
        msg += f" -> {out_dir}"
        if status is not None:
            status.configure(text=msg)
        self.log(msg)
        return man

    def _export_frames_dialog(self, indices):
        """Small options dialog (channel, peaks.csv, destination), then export."""
        tk, ttk = self.tk, self.ttk
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._fm_status.configure(text="Run Step 1 first (no analysis file yet).")
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Export {len(indices)} frame(s)")
        dlg.configure(bg=theme.C.BG)
        dlg.transient(self.root)
        dlg.grab_set()
        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=(
            "Writes each frame as a two-column .xy pattern (native axis "
            "always; 2θ too when the wavelength is known) and optional CSVs "
            "for fitted, residual, and unknown peaks."),
            style="Muted.TLabel", wraplength=430, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(content, text="Pattern channel").grid(row=1, column=0,
                                                        sticky="w", pady=2)
        v_src = tk.StringVar(value="fit")
        _src = ttk.Combobox(content, textvariable=v_src, state="readonly",
                            width=12,
                            values=["fit", "clean", "mean", "hybrid",
                                    "sigmaclip", "spots", "robust",
                                    "residual"])
        _src.grid(row=1, column=1, sticky="w")
        _ToolTip(_src, "fit = the channel Step 2 actually fitted (default). "
                       "residual = /residual/clean after phase subtraction. "
                       "spots = spot_residual alone (mean − median): rings "
                       "and smooth background cancel, leaving the coarse-"
                       "grain/single-crystal sample signal. "
                       "The other entries are reduction-side channels "
                       "reconstructed exactly as the pipeline does.")
        v_peaks = tk.BooleanVar(value=True)
        ttk.Checkbutton(content, text="Include fitted peaks (peaks.csv)",
                        variable=v_peaks).grid(row=2, column=0, columnspan=2,
                                               sticky="w", pady=2)
        v_resunk = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            content,
            text="Include residual/unknown peaks when available",
            variable=v_resunk,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(content, text="Exclude d (Å)").grid(row=5, column=0,
                                                      sticky="w", pady=2)
        v_exd = tk.StringVar(value=str(self.config.get("export_exclude_d", "")))
        _exd = ttk.Entry(content, textvariable=v_exd, width=36)
        _exd.grid(row=5, column=1, sticky="we")
        _ToolTip(_exd, "Optional comma-separated d-spacings (Å) whose "
                       "windows (±2.8%) are zeroed in every exported "
                       "pattern AND the stacked figure — known contaminant "
                       "lines (gasket W, diamond). Blank = no exclusion.")
        v_stack = tk.BooleanVar(value=False)
        _stack_row = ttk.Frame(content)
        _stack_row.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        _stack_cb = ttk.Checkbutton(
            _stack_row,
            text="Also write stacked figure (stack.png)  style:",
            variable=v_stack,
        )
        _stack_cb.pack(side="left")
        v_stack_style = tk.StringVar(
            value=str(self.config.get("stack_style", "panels")))
        _stack_style = ttk.Combobox(_stack_row, textvariable=v_stack_style,
                                    state="readonly", width=9,
                                    values=["panels", "waterfall"])
        _stack_style.pack(side="left", padx=(4, 0))
        _ToolTip(_stack_cb, "Figure of the exported frames ordered by "
                            "/frames/pressure, sequential color = pressure, "
                            "same channel as the export. panels = touching "
                            "subplots (journal layout); waterfall = offset "
                            "traces on one axes (best for tracking peak "
                            "drift across many frames).")
        from .stackplot import FIGURE_PRESETS, FIGURE_FORMATS
        ttk.Label(_stack_row, text="  preset:").pack(side="left")
        v_fig_preset = tk.StringVar(
            value=str(self.config.get("fig_preset", "screen")))
        _fig_preset = ttk.Combobox(_stack_row, textvariable=v_fig_preset,
                                   state="readonly", width=12,
                                   values=list(FIGURE_PRESETS))
        _fig_preset.pack(side="left", padx=(2, 0))
        _ToolTip(_fig_preset, "screen 110 dpi PNG · presentation 200 dpi PNG "
                              "· publication 600 dpi, vector by default. "
                              "Every export writes an export_provenance.txt "
                              "sidecar recording versions and settings.")
        ttk.Label(_stack_row, text="format:").pack(side="left", padx=(6, 0))
        v_fig_format = tk.StringVar(
            value=str(self.config.get("fig_format", "")))
        _fig_format = ttk.Combobox(_stack_row, textvariable=v_fig_format,
                                   state="readonly", width=5,
                                   values=[""] + list(FIGURE_FORMATS))
        _fig_format.pack(side="left", padx=(2, 0))
        _ToolTip(_fig_format, "Blank = the preset's default format.")
        ttk.Label(content, text="Destination").grid(row=6, column=0,
                                                    sticky="w", pady=2)
        v_dir = tk.StringVar(value=str(self.config.get("export_frames_dir", "")))
        ttk.Entry(content, textvariable=v_dir, width=36).grid(
            row=6, column=1, sticky="we")

        def _browse():
            d = self.filedialog.askdirectory(title="Export destination folder")
            if d:
                v_dir.set(d)
        ttk.Button(content, text="Browse", command=_browse).grid(
            row=6, column=2, padx=4)

        def _go():
            dest = v_dir.get().strip()
            if not dest:
                self.messagebox.showerror("Export frames",
                                          "Pick a destination folder.",
                                          parent=dlg)
                return
            try:
                exd = [float(s) for s in v_exd.get().split(",") if s.strip()]
            except ValueError:
                self.messagebox.showerror(
                    "Export frames",
                    "Exclude d must be a comma-separated list of numbers.",
                    parent=dlg)
                return
            self.config["export_frames_dir"] = dest
            self.config["export_exclude_d"] = v_exd.get().strip()
            self.config["stack_style"] = v_stack_style.get()
            self.config["fig_preset"] = v_fig_preset.get()
            self.config["fig_format"] = v_fig_format.get()
            self.save_config(silent=True)
            dlg.destroy()
            self._do_export_frames(indices, dest, source=v_src.get(),
                                   peaks=bool(v_peaks.get()),
                                   residual_unknowns=bool(v_resunk.get()),
                                   stack=bool(v_stack.get()),
                                   stack_style=v_stack_style.get(),
                                   exclude_d=exd or None,
                                   fig_preset=v_fig_preset.get(),
                                   fig_format=v_fig_format.get())

        btns = ttk.Frame(content)
        btns.grid(row=7, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Export", command=_go).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left",
                                                                  padx=4)
        content.columnconfigure(1, weight=1)

    def review_export_frame_clicked(self):
        """Export the frame currently shown on the Review tab."""
        try:
            idx = int(self._review_idx_var.get())
        except (ValueError, TypeError):
            idx = 0
        self._export_frames_dialog([idx])

    def review_export_ringless_clicked(self):
        """Export the shown frame's ring-removed cake (PNG + .npy)."""
        self.pull_vars()
        try:
            idx = int(self._review_idx_var.get())
        except (ValueError, TypeError):
            idx = 0
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            self.log("Ringless export: no reduced HDF5 loaded.", "WARN")
            return
        default = str(self.config.get("export_frames_dir", "")
                      or Path(reduced).parent)
        dest = self.filedialog.askdirectory(
            title=f"Export ring-removed cake of frame {idx}",
            initialdir=default,
        )
        if not dest:
            return
        try:
            from .spots import export_ring_removed_cakes
            man = export_ring_removed_cakes(reduced, dest, [idx])
        except Exception as e:
            self.log(f"Ringless cake export failed: {e!r}", "WARN")
            return
        self.config["export_frames_dir"] = dest
        self.save_config(silent=True)
        if man["files"]:
            self.log(f"Ringless cake of frame {idx} -> {dest} "
                     f"({', '.join(man['files'])})")
        else:
            self.log(f"Frame {idx}: no cake saved in the reduction — "
                     f"nothing exported.", "WARN")

    def fm_export_selected_clicked(self):
        """Export the frames selected in the Frame meta table."""
        self.pull_vars()
        tbl = getattr(self, "_fm_table", None)
        sel = list(tbl.selection()) if tbl is not None else []
        if not sel:
            self._fm_status.configure(
                text="Select one or more frames in the table first.")
            return
        try:
            indices = sorted(int(iid) for iid in sel)
        except ValueError:
            return
        self._export_frames_dialog(indices)

    def import_positions_clicked(self):
        """Dialog: read per-frame stage positions from the raw frames' headers."""
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            self._fm_status.configure(text="Run Step 1 first (no analysis file yet).")
            return
        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("Read X/Y from frame headers")
        dlg.configure(bg=theme.C.BG)
        dlg.transient(self.root)
        dlg.grab_set()
        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=(
            "Reads each frame's raw image header (via fabio) and stores the "
            "two motor values as /frames/pos_x and pos_y. Key names are "
            "case-insensitive; the motor_mne/motor_pos pair convention is "
            "understood. 'List keys' shows what the first frame's header "
            "offers."), style="Muted.TLabel", wraplength=460, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        v_kx = tk.StringVar(value=str(self.config.get("pos_header_x", "")))
        v_ky = tk.StringVar(value=str(self.config.get("pos_header_y", "")))
        v_dir = tk.StringVar(value=str(self.config.get("pos_header_dir", "")))
        ttk.Label(content, text="X header key").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_kx, width=20).grid(row=1, column=1, sticky="w")
        ttk.Label(content, text="Y header key").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_ky, width=20).grid(row=2, column=1, sticky="w")
        ttk.Label(content, text="Frames folder").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(content, textvariable=v_dir, width=36).grid(row=3, column=1, sticky="we")

        def _browse():
            d = self.filedialog.askdirectory(title="Folder holding the raw frames")
            if d:
                v_dir.set(d)
        ttk.Button(content, text="Browse", command=_browse).grid(row=3, column=2, padx=4)
        ttk.Label(content, text=(
            "Folder is only needed when /frames/filename holds bare names "
            "instead of full paths."), style="Muted.TLabel", wraplength=460,
            justify="left").grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 8))

        def _list_keys():
            from .frame_metadata import frame_header_keys
            probe = frame_header_keys(path, search_dir=v_dir.get().strip() or None)
            if probe.get("ok"):
                self.messagebox.showinfo(
                    "Header keys",
                    f"{Path(probe['path']).name}:\n\n" + ", ".join(probe["keys"]),
                    parent=dlg)
            else:
                self.messagebox.showerror("Header keys", probe.get("error", "?"),
                                          parent=dlg)

        def _go():
            kx, ky = v_kx.get().strip(), v_ky.get().strip()
            if not kx or not ky:
                self.messagebox.showerror("Read positions",
                                          "Enter both header keys.", parent=dlg)
                return
            from .frame_metadata import import_positions_from_headers
            try:
                man = import_positions_from_headers(
                    path, kx, ky, search_dir=v_dir.get().strip() or None)
            except Exception as e:
                self.messagebox.showerror("Read positions failed", str(e), parent=dlg)
                return
            self.config["pos_header_x"] = kx
            self.config["pos_header_y"] = ky
            self.config["pos_header_dir"] = v_dir.get().strip()
            self.save_config(silent=True)
            dlg.destroy()
            msg = f"Positions read for {man['n_mapped']} frame(s)."
            if man.get("n_missing_file"):
                msg += f" {man['n_missing_file']} frame file(s) not found."
            self._fm_status.configure(text=msg)
            self.log(msg)
            self.fm_refresh_table_clicked()

        btns = ttk.Frame(content)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="List keys", command=_list_keys).pack(side="left", padx=4)
        ttk.Button(btns, text="Read positions", command=_go).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=4)
        content.columnconfigure(1, weight=1)

    def preview_pressure_clicked(self):
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        self._draw_pressure_preview(path)

    def fm_refresh_table_clicked(self):
        from .frame_metadata import read_frame_metadata
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        tbl = getattr(self, "_fm_table", None)
        if tbl is None:
            return
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        tbl.delete(*tbl.get_children())
        meta = read_frame_metadata(path)
        if not meta.get("ok"):
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=meta.get("error", "Failed to read metadata."))
            return
        names = meta.get("filename") or []
        pressure = meta.get("pressure")
        sigma = meta.get("pressure_sigma")
        temp = meta.get("temperature")
        user = meta.get("user_edited")
        n = int(meta.get("n_frames", 0) or 0)

        def _fmt(arr, i):
            if arr is None or i >= len(arr):
                return "—"
            v = float(arr[i])
            return f"{v:.3g}" if v == v else "—"

        for i in range(n):
            fname = names[i] if i < len(names) else ""
            base = fname.rsplit("/", 1)[-1] if fname else ""
            is_user = bool(user[i]) if user is not None and i < len(user) else False
            tbl.insert("", "end", iid=str(i), values=(
                i, base, _fmt(pressure, i), _fmt(sigma, i), _fmt(temp, i),
                "user" if is_user else "auto"),
                tags=(("user",) if is_user else ()))

    def fm_apply_selected_clicked(self):
        from .frame_metadata import read_frame_metadata, apply_to_analysis
        import numpy as np
        self.pull_vars()
        path = self.config.get("analysis_h5_file", "")
        if not path or not Path(path).is_file():
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text="Run Step 1 first (no analysis file yet).")
            return
        tbl = getattr(self, "_fm_table", None)
        sel = list(tbl.selection()) if tbl is not None else []
        if not sel:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="Select one or more frames first.")
            return
        try:
            indices = [int(iid) for iid in sel]
        except ValueError:
            indices = []

        def _parse(var):
            raw = (var.get() if var is not None else "").strip()
            if not raw:
                return None
            return float(raw)

        try:
            p_val = _parse(getattr(self, "_fm_edit_p", None))
            sig_val = _parse(getattr(self, "_fm_edit_sig", None))
            t_val = _parse(getattr(self, "_fm_edit_t", None))
        except ValueError:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="P/σ/T must be a number (or blank).")
            return

        if p_val is None and sig_val is None and t_val is None:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text="Enter at least one value.")
            return

        try:
            meta = read_frame_metadata(path)
            if not meta.get("ok"):
                raise RuntimeError(meta.get("error", "Failed to read metadata."))
            pressure = np.array(meta.get("pressure"), dtype=float, copy=True)
            sigma = np.array(meta.get("pressure_sigma"), dtype=float, copy=True)
            temperature = np.array(meta.get("temperature"), dtype=float, copy=True)

            parts = []
            kwargs = {}
            if p_val is not None:
                pressure[indices] = p_val
                kwargs["pressure"] = pressure
                parts.append("P")
            if sig_val is not None:
                sigma[indices] = sig_val
                kwargs["pressure_sigma"] = sigma
                parts.append("σ")
            if t_val is not None:
                temperature[indices] = t_val
                kwargs["temperature"] = temperature
                parts.append("T")

            apply_to_analysis(path, user_frames=indices, **kwargs)
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(
                    text=f"Set {', '.join(parts)} on {len(sel)} frame(s) "
                         "(marked as user edits — they survive re-runs).")
            self.fm_refresh_table_clicked()
            self._draw_pressure_preview(path)
        except Exception as e:
            if hasattr(self, "_fm_status"):
                self._fm_status.configure(text=str(e))
            self.log(f"fm_apply_selected_clicked failed: {e!r}", "WARN")

    def _draw_pressure_preview(self, path):
        from .frame_metadata import read_frame_metadata
        import numpy as np

        meta = read_frame_metadata(path)

        for w in self.fm_plot_frame.winfo_children():
            w.destroy()

        # Close previous figure if any
        prev = getattr(self, "_fm_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._fm_fig = None

        pressure = meta.get("pressure")
        if pressure is None:
            pressure = np.array([])
        pressure = np.asarray(pressure, dtype=float)

        if pressure.size == 0 or not np.any(np.isfinite(pressure)):
            self.ttk.Label(
                self.fm_plot_frame,
                text="No pressures yet — Extract from filenames or Import CSV.",
                style="Muted.TLabel",
            ).pack(anchor="center", expand=True)
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            # matplotlib unavailable — show text summary instead
            n_frames = int(meta.get("n_frames", 0))
            from .frame_metadata import summarize_pressures
            try:
                summ = summarize_pressures(pressure)
            except Exception:
                summ = {}
            n_parsed = int(summ.get("n_parsed", 0)) if summ else 0
            p_min = summ.get("p_min")
            p_max = summ.get("p_max")
            p_txt = (
                f"P {p_min:.2f}–{p_max:.2f} GPa"
                if p_min is not None and p_max is not None else "P unknown"
            )
            self.ttk.Label(
                self.fm_plot_frame,
                text=(
                    f"matplotlib unavailable: {e}\n\n"
                    f"{n_parsed}/{n_frames} frames have pressure ({p_txt})."
                ),
                style="Muted.TLabel", justify="left",
            ).pack(anchor="center", expand=True)
            return

        fig = Figure(figsize=(7, 4), dpi=100, layout="constrained")
        self._fm_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(pressure.size, dtype=float)
        ax.plot(x, pressure, marker=".", markersize=3, linewidth=0.8, color=theme.C.ACCENT2)
        ax.set_xlabel(FRAME_LABEL)
        ax.set_ylabel(PRESSURE_LABEL)
        ax.set_title("Frame pressure", color=theme.C.FG)
        self._style_ax(ax)

        self._fm_canvas = self._embed_figure(self.fm_plot_frame, fig)

    # ------------------------------------------------------------------
    # Tab 9 — Identify (Step 3a: deterministic EOS phase matching)
    # ------------------------------------------------------------------

    def _tab_identify(self, frame):
        tk, ttk = self.tk, self.ttk

        # -- params area --------------------------------------------------
        title_row = ttk.Frame(frame)
        title_row.grid(row=0, column=0, columnspan=6, sticky="we", padx=6, pady=(0, 2))
        ttk.Label(
            title_row, text="Phase identification (EOS matching)",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side="left")
        self._identify_help_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(title_row, text="Show instructions",
                        variable=self._identify_help_var,
                        command=self._toggle_identify_help).pack(side="left", padx=12)

        self.checkbox(frame, "run_step3",
                      "Enable phase identification in the next run", row=1)
        self.checkbox(frame, "identify_all_phases",
                      "Search entire library (identify without pre-selecting candidates)",
                      row=2)
        # Two column-groups keep the tab short enough that the results area
        # below stays fully visible on ~700px-tall screens.
        self.field(frame, "p_min", "Pressure min (GPa)", row=3, width=10)
        self.field(frame, "p_max", "Pressure max (GPa)", row=4, width=10)
        self.field(frame, "rel_tol", "Match tolerance (Δd/d)", row=5, width=10)
        self.field(frame, "seen_conf", "Present-in-frame confidence", row=6, width=10)
        self.field(frame, "identify_wavelength",
                   "Wavelength (Å, blank=auto)", row=7, width=10)
        self.field(frame, "pressure_window",
                   "Pressure window ± (GPa)", row=3, width=10, col=1)
        self.field(frame, "pressure_sigma_k",
                   "Window = k·σ (when σ known)", row=4, width=10, col=1)
        self.field(frame, "min_matched",
                   "Min matched reflections", row=5, width=10, col=1)
        self.field(frame, "intensity_k",
                   "Intensity weight (0 = positions only)", row=6, width=10, col=1)
        self.checkbox(frame, "use_pressure_prior",
                      "Use frame-pressure prior (confine fit to ±window)", row=8)
        self.checkbox(frame, "marker_prior",
                      "Estimate pressure from marker phases first", row=8, col=1)
        self.checkbox(frame, "allow_sparse",
                      "Allow sparse/marker-only matches in residual", row=9)
        self.checkbox(frame, "use_frame_temperature",
                      "Apply frame temperatures (thermal expansion)", row=9, col=1)

        # -- Step 3b proposer: ML candidate ranking -----------------------
        mlrow = ttk.Frame(frame)
        mlrow.grid(row=10, column=0, columnspan=6, sticky="w", pady=(6, 2))
        self.vars["run_ml_rank"] = tk.BooleanVar(value=bool(self.config.get("run_ml_rank", False)))
        _mlcb = ttk.Checkbutton(
            mlrow, text="ML candidate ranking (top-K from library → Step 3a verifies)",
            variable=self.vars["run_ml_rank"])
        _mlcb.pack(side="left")
        _ToolTip(_mlcb, (
            "Before the deterministic match, rank the WHOLE library against each frame "
            "(cosine of the measured residual/fit pattern vs each phase simulated at the "
            "frame's pressure) and verify only the top-K with Step 3a. 'ML proposes, "
            "physics verifies.' Deterministic (no torch); needs pymatgen to simulate."))
        ttk.Label(mlrow, text="top-K:", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["ml_rank_top_k"] = tk.StringVar(value=str(self.config.get("ml_rank_top_k", "5")))
        ttk.Entry(mlrow, textvariable=self.vars["ml_rank_top_k"], width=5).pack(side="left")
        ttk.Label(mlrow, text="rank vs:", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["ml_rank_source"] = tk.StringVar(value=str(self.config.get("ml_rank_source", "auto")))
        ttk.Combobox(mlrow, textvariable=self.vars["ml_rank_source"],
                     values=["auto", "residual", "fit"], state="readonly", width=8).pack(side="left")
        ttk.Label(mlrow, text="scorer:", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["ml_scorer"] = tk.StringVar(value=str(self.config.get("ml_scorer", "")))
        _mlsc = ttk.Entry(mlrow, textvariable=self.vars["ml_scorer"], width=22)
        _mlsc.pack(side="left")
        _ToolTip(_mlsc, (
            "Similarity scorer for the ranking. Blank/'cosine' = the deterministic "
            "baseline. 'torch:<path to scorer.pt>' = a trained seriesxrd-ml-train "
            "export (needs seriesxrd[ml]; see docs/ml-training.md). Whatever the "
            "scorer proposes, Step 3a still verifies."))

        # -- Step 3c unknown tracking ------------------------------------
        unkrow = ttk.Frame(frame)
        unkrow.grid(row=11, column=0, columnspan=6, sticky="w", pady=(4, 2))
        ttk.Label(unkrow, text="Unknown tracking:", style="Muted.TLabel").pack(
            side="left", padx=(0, 6))
        ttk.Label(unkrow, text="track by", style="Muted.TLabel").pack(side="left", padx=(4, 2))
        self.vars["unknown_tracking_axis"] = tk.StringVar(
            value=str(self.config.get("unknown_tracking_axis", "same") or "same"))
        _ut_axis = ttk.Combobox(
            unkrow, textvariable=self.vars["unknown_tracking_axis"],
            values=["same", "frame", "pressure", "temperature", "time"],
            state="readonly", width=11)
        _ut_axis.pack(side="left")
        _ToolTip(_ut_axis, HELP["unknown_tracking_axis"])
        ttk.Label(unkrow, text="group by", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_group_by"] = tk.StringVar(
            value=str(self.config.get("unknown_group_by", "same") or "same"))
        _ut_group = ttk.Combobox(
            unkrow, textvariable=self.vars["unknown_group_by"],
            values=["same", "none", "scan", "folder"], state="readonly", width=8)
        _ut_group.pack(side="left")
        _ToolTip(_ut_group, HELP["unknown_group_by"])
        ttk.Label(unkrow, text="tol×FWHM", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_link_tol_fwhm"] = tk.StringVar(
            value=str(self.config.get("unknown_link_tol_fwhm", "1.5")))
        _ut_tol = ttk.Entry(unkrow, textvariable=self.vars["unknown_link_tol_fwhm"], width=5)
        _ut_tol.pack(side="left")
        _ToolTip(_ut_tol, HELP["unknown_link_tol_fwhm"])
        ttk.Label(unkrow, text="missing", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_max_gap"] = tk.StringVar(
            value=str(self.config.get("unknown_max_gap", "2")))
        _ut_gap = ttk.Entry(unkrow, textvariable=self.vars["unknown_max_gap"], width=5)
        _ut_gap.pack(side="left")
        _ToolTip(_ut_gap, HELP["unknown_max_gap"])
        ttk.Label(unkrow, text="axis gap", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_max_axis_gap"] = tk.StringVar(
            value=str(self.config.get("unknown_max_axis_gap", "")))
        _ut_agap = ttk.Entry(unkrow, textvariable=self.vars["unknown_max_axis_gap"], width=7)
        _ut_agap.pack(side="left")
        _ToolTip(_ut_agap, HELP["unknown_max_axis_gap"])
        ttk.Label(unkrow, text="min frames", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_min_frames"] = tk.StringVar(
            value=str(self.config.get("unknown_min_frames", "3")))
        _ut_min = ttk.Entry(unkrow, textvariable=self.vars["unknown_min_frames"], width=5)
        _ut_min.pack(side="left")
        _ToolTip(_ut_min, HELP["unknown_min_frames"])
        ttk.Label(unkrow, text="Jaccard", style="Muted.TLabel").pack(side="left", padx=(10, 2))
        self.vars["unknown_jaccard"] = tk.StringVar(
            value=str(self.config.get("unknown_jaccard", "0.6")))
        _ut_j = ttk.Entry(unkrow, textvariable=self.vars["unknown_jaccard"], width=5)
        _ut_j.pack(side="left")
        _ToolTip(_ut_j, HELP["unknown_jaccard"])
        self.vars["unknown_axis_predictor"] = tk.BooleanVar(
            value=bool(self.config.get("unknown_axis_predictor", True)))
        _ut_pred = ttk.Checkbutton(
            unkrow, text="predict drift", variable=self.vars["unknown_axis_predictor"])
        _ut_pred.pack(side="left", padx=(10, 2))
        _ToolTip(_ut_pred, HELP["unknown_axis_predictor"])

        self._identify_help = ttk.Label(
            frame,
            text=(
                "Step 3a fits each candidate phase's equation of state to every "
                "frame's peak list and reports a match confidence (and, for "
                "pressure series, a best-fit pressure). Needs pymatgen.\n\n"
                "To run it:\n"
                "  1. Configure → Phases — enable the candidate phases (or tick Search "
                "entire library here).\n"
                "  2. Tick 'Enable phase identification' above.\n"
                "  3. Run → Run analysis — start the analysis.\n"
                "  4. Click 'Load identification' to see the results.\n"
                "The residual step then subtracts confirmed phases and re-fits, "
                "and the unknowns step clusters whatever is left."
            ),
            style="Muted.TLabel", justify="left", wraplength=640,
        )
        self._identify_help.grid(row=13, column=0, columnspan=6, sticky="w",
                                 padx=6, pady=(8, 4))
        self._identify_help.grid_remove()   # hidden until the checkbox reveals it
        self.autowrap(self._identify_help)

        # -- controls row -------------------------------------------------
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=12, column=0, columnspan=6, sticky="w", pady=(4, 2))

        ttk.Button(ctrl, text="Load identification",
                   command=self.load_identify).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_identify_fig", None), "Phase identification")
                   ).pack(side="left", padx=4)

        ttk.Label(ctrl, text="Min confidence:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self._identify_conf_var = tk.StringVar(value="0.5")
        _conf_entry = ttk.Entry(ctrl, textvariable=self._identify_conf_var, width=6)
        _conf_entry.pack(side="left", padx=2)
        _conf_entry.bind("<Return>", lambda e: self.load_identify())

        ttk.Button(ctrl, text="Redraw",
                   command=self.load_identify).pack(side="left", padx=4)
        ttk.Button(
            ctrl, text="Estimate phase fractions",
            command=self.run_phase_fractions_clicked,
        ).pack(side="left", padx=(10, 4))

        self._identify_status = ttk.Label(ctrl, text="", style="Muted.TLabel")
        self._identify_status.pack(side="left", padx=12)

        # -- body: per-frame phase table (left) + plot (right) ------------
        # Keep the body below the optional help row. When help is hidden, row
        # 13 collapses automatically; when shown, the results move down.
        body = ttk.Frame(frame)
        body.grid(row=14, column=0, columnspan=6, sticky="nsew")
        frame.rowconfigure(14, weight=1)
        frame.columnconfigure(0, weight=1)

        # Left: two browse modes in a sub-notebook. Stacking the per-frame
        # table, the materials summary, AND the frames list vertically used to
        # need ~470px and pushed the tab bottom off short screens.
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 6))
        lnb = ttk.Notebook(left)
        lnb.pack(fill="both", expand=True)
        page_frame = ttk.Frame(lnb, padding=4)
        page_mat = ttk.Frame(lnb, padding=4)
        lnb.add(page_frame, text="This frame")
        lnb.add(page_mat, text="Materials")

        # Page 1 — a frame selector and the ranked phase table for that frame.
        sel = ttk.Frame(page_frame)
        sel.pack(fill="x", pady=(0, 4))
        ttk.Label(sel, text="Frame", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        ttk.Button(sel, text="◀", width=2,
                   command=lambda: self._step_identify_frame(-1)).pack(side="left")
        self._identify_frame_var = tk.StringVar(value="0")
        self._identify_frame_spin = ttk.Spinbox(
            sel, from_=0, to=0, width=6, textvariable=self._identify_frame_var,
            command=self._update_identify_table)
        self._identify_frame_spin.pack(side="left", padx=2)
        self._identify_frame_spin.bind("<Return>", lambda e: self._update_identify_table())
        ttk.Button(sel, text="▶", width=2,
                   command=lambda: self._step_identify_frame(1)).pack(side="left")

        tbl_frame = ttk.Frame(page_frame)
        tbl_frame.pack(fill="both", expand=True)
        cols = ("phase", "model", "conf", "recall", "prec", "pressure", "lines")
        tbl = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=6,
                           selectmode="browse")
        tbl_vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tbl.yview)
        tbl.configure(yscrollcommand=tbl_vsb.set)
        for c, txt, w, anc in (("phase", "Phase", 140, "w"), ("model", "P-model", 78, "center"),
                               ("conf", "Conf", 52, "center"),
                               ("recall", "Recall", 52, "center"), ("prec", "Prec", 52, "center"),
                               ("pressure", "P (GPa)", 60, "center"), ("lines", "#", 36, "center")):
            tbl.heading(c, text=txt)
            tbl.column(c, width=w, minwidth=34, anchor=anc, stretch=(c == "phase"))
        tbl.tag_configure("present", foreground=theme.C.ACCENT2)
        tbl.tag_configure("absent", foreground=theme.C.MUTED)
        tbl_vsb.pack(side="right", fill="y")
        tbl.pack(side="left", fill="both", expand=True)
        self._identify_table = tbl

        # Page 2 — materials-found summary + frames-by-material browser.
        ttk.Label(page_mat, text="Materials found (click → frames containing it):",
                 style="Muted.TLabel").pack(anchor="w", pady=(0, 2))

        summary_cols = ("phase", "frames", "medP")
        summary = ttk.Treeview(page_mat, columns=summary_cols, show="headings", height=5,
                               selectmode="browse")
        for c, txt, w in (("phase", "Material", 140), ("frames", "Frames", 60),
                          ("medP", "med P", 70)):
            summary.heading(c, text=txt)
            summary.column(c, width=w, minwidth=34, anchor="center" if c != "phase" else "w",
                           stretch=(c == "phase"))
        summary.pack(fill="x", expand=False)
        summary.bind("<<TreeviewSelect>>", self._on_phase_summary_select)
        self._identify_phase_summary = summary

        ttk.Label(page_mat, text="Frames with selected material (double-click to view):",
                 style="Muted.TLabel").pack(anchor="w", pady=(6, 2))

        frames_list_frame = ttk.Frame(page_mat)
        frames_list_frame.pack(fill="both", expand=True)
        frames_vsb = ttk.Scrollbar(frames_list_frame, orient="vertical")
        listbox = tk.Listbox(
            frames_list_frame, height=5, bg=theme.C.BG2, fg=theme.C.FG,
            selectbackground=theme.C.ACCENT2, yscrollcommand=frames_vsb.set,
            exportselection=False,
        )
        frames_vsb.configure(command=listbox.yview)
        frames_vsb.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)
        listbox.bind("<Double-Button-1>", self._on_phase_frame_activate)
        self._identify_frames_list = listbox
        self._phase_frames: Dict[str, Any] = {}
        self._phase_frame_indices = []

        # Right: the (decluttered) confidence/pressure plot.
        self.identify_plot_frame = ttk.Frame(body)
        self.identify_plot_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(
            self.identify_plot_frame,
            text="Enable phase identification and Run, or click \"Load "
                 "identification\" to view per-frame phases + confidence. "
                 "Tick \"Show instructions\" (top) for the step-by-step workflow.",
            style="Muted.TLabel", wraplength=380, justify="left",
        ).pack(anchor="center", expand=True)

    def run_phase_fractions_clicked(self):
        """Calculate and store semi-quantitative phase intensity shares."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror(
                "Phase fractions", "Run phase identification before estimating fractions.")
            return
        try:
            from .fractions import run_fractions
            result = run_fractions(path)
        except Exception as exc:
            self.messagebox.showerror("Phase fractions", str(exc))
            return
        if not result.get("ok"):
            self.messagebox.showerror(
                "Phase fractions", result.get("error") or "No attributed peaks are available.")
            return
        lines = [
            "Semi-quantitative intensity shares were saved to the analysis file.",
            "",
        ]
        for name, summary in result.get("per_phase", {}).items():
            mean = summary.get("mean_fraction", float("nan"))
            lines.append(f"{name}: mean {mean:.1%}")
        lines += [
            "",
            "These values are screening estimates, not refined weight fractions.",
        ]
        self.log(f"Phase fractions saved: {path}")
        self.messagebox.showinfo("Phase fractions", "\n".join(lines))

    def load_identify(self):
        """Render the Step-3a pressure-vs-frame plot from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        # prev-figure-close leak guard
        prev = getattr(self, "_identify_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._identify_fig = None

        for w in self.identify_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.identify_plot_frame,
                text=f"matplotlib unavailable: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .review import identify_tracks
        tr = identify_tracks(path)
        self._identify_tr = tr
        if not tr["ok"]:
            self.ttk.Label(
                self.identify_plot_frame,
                text=tr["error"],
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            if hasattr(self, "_identify_status"):
                self._identify_status.configure(text=tr["error"])
            return

        # Sync the per-frame table + its frame selector range.
        if hasattr(self, "_identify_frame_spin"):
            self._identify_frame_spin.configure(to=max(tr["n_frames"] - 1, 0))
        self._update_identify_table()

        # Parse confidence threshold
        conf_min = 0.5
        try:
            conf_min = float(self._identify_conf_var.get())
            conf_min = max(0.0, min(1.0, conf_min))
        except (ValueError, AttributeError):
            pass

        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._identify_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax_pres = fig.add_subplot(2, 1, 1)
        ax_conf = fig.add_subplot(2, 1, 2)

        # Only plot phases actually seen at least once (max confidence ≥ the bar),
        # so the figure isn't 22 overlapping flat lines + a giant legend. Fall back
        # to the strongest few if nothing clears the bar, so it's never blank.
        def _maxconf(rec):
            c = rec.get("confidence")
            return float(np.nanmax(c)) if c is not None and len(c) else 0.0
        shown = [r for r in tr["phases"] if _maxconf(r) >= conf_min]
        if not shown:
            shown = sorted(tr["phases"], key=_maxconf, reverse=True)[:5]

        for rec in shown:
            name = rec["name"]
            pressure = np.asarray(rec["pressure"], dtype=float)
            conf_arr = (
                np.asarray(rec["confidence"], dtype=float)
                if rec["confidence"] is not None
                else np.zeros(pressure.size, dtype=float)
            )
            x = np.arange(pressure.size)
            mask = conf_arr >= conf_min

            label = name if rec["has_eos"] else f"{name} (no EOS)"

            # Plot pressure where mask is satisfied; capture the line color.
            if mask.any():
                (ln,) = ax_pres.plot(
                    x[mask], pressure[mask],
                    marker=".", markersize=3, linewidth=0.7,
                    label=label,
                )
                color = ln.get_color()
            else:
                # No points meet threshold — still need a color for confidence axis.
                (ln,) = ax_pres.plot([], [], marker=".", markersize=3,
                                     linewidth=0.7, label=label)
                color = ln.get_color()

            # Always show confidence trace in the same color.
            ax_conf.plot(x, conf_arr, linewidth=0.7, color=color, label=label)

        ax_pres.set_ylabel(PRESSURE_LABEL)
        ax_pres.set_title(
            f"Phases seen (confidence ≥ {conf_min:.2f}) — {len(shown)} shown",
            color=theme.C.FG)
        handles, labels = ax_pres.get_legend_handles_labels()
        if handles:
            ax_pres.legend(fontsize=7, framealpha=0.4, ncol=2, loc="upper right")
        self._style_ax(ax_pres)

        ax_conf.axhline(conf_min, color=theme.C.MUTED, linewidth=0.8, linestyle="--")
        ax_conf.set_xlabel(FRAME_LABEL)
        ax_conf.set_ylabel("Confidence")
        ax_conf.set_ylim(0, 1.02)
        self._style_ax(ax_conf)

        self._identify_canvas = self._embed_figure(self.identify_plot_frame, fig)

        if hasattr(self, "_identify_status"):
            self._identify_status.configure(
                text=f"{len(tr['phases'])} phase(s), {tr['n_frames']} frames")
        self._attach_hover(self._identify_canvas, self._identify_status)

    def _step_identify_frame(self, delta: int):
        try:
            cur = int(float(self._identify_frame_var.get()))
        except (ValueError, AttributeError):
            cur = 0
        n = int(getattr(self, "_identify_tr", {}).get("n_frames", 0) or 0)
        cur = max(0, min(cur + delta, max(n - 1, 0)))
        self._identify_frame_var.set(str(cur))
        self._update_identify_table()

    def _update_identify_table(self):
        """Fill the per-frame table: phases ranked by confidence for the selected
        frame, with recall / precision / best-fit pressure. Present phases (≥ the
        Min-confidence bar) are highlighted."""
        import numpy as np
        tbl = getattr(self, "_identify_table", None)
        tr = getattr(self, "_identify_tr", None)
        if tbl is None or not tr or not tr.get("ok"):
            return
        tbl.delete(*tbl.get_children())
        n = int(tr.get("n_frames", 0) or 0)
        try:
            fi = max(0, min(int(float(self._identify_frame_var.get())), max(n - 1, 0)))
        except (ValueError, AttributeError):
            fi = 0
        try:
            conf_min = max(0.0, min(1.0, float(self._identify_conf_var.get())))
        except (ValueError, AttributeError):
            conf_min = 0.5

        def _at(arr, default=np.nan):
            return float(arr[fi]) if arr is not None and fi < len(arr) else default

        rows = []
        for rec in tr["phases"]:
            conf = _at(rec.get("confidence"), 0.0)
            rows.append((conf, rec))
        rows.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1), t[1]["name"].lower()))
        # Short labels for the pressure model the phase was fit under.
        # ("ambient_only" is the pre-rename value in old HDF5 files — read-compat.)
        _MODEL_LABEL = {"eos": "EOS", "axial_eos": "axial", "no_eos": "no-EOS",
                        "ambient_only": "no-EOS"}
        n_present = 0
        for conf, rec in rows:
            recall = _at(rec.get("recall"))
            prec = _at(rec.get("precision"))
            press = _at(rec.get("pressure"))
            penalty = _at(rec.get("prior_penalty"), 1.0)
            nmatch = rec.get("n_matched")
            nm = int(nmatch[fi]) if nmatch is not None and fi < len(nmatch) else 0
            present = conf >= conf_min
            n_present += int(present)
            model = rec.get("pressure_model") or ("eos" if rec.get("has_eos") else "no_eos")
            mlabel = _MODEL_LABEL.get(model, model)
            # Flag exemption from / impact of the pressure prior.
            if rec.get("pressure_assumption") == "ignore_prior":
                mlabel += " (no prior)"
            elif penalty == penalty and penalty < 0.95:
                mlabel += " ↓P"
            name = rec["name"]
            pstr = "—" if (press != press or model in ("no_eos", "ambient_only")) else f"{press:.1f}"
            tbl.insert("", "end", values=(
                name, mlabel, f"{conf:.2f}",
                "—" if recall != recall else f"{recall:.2f}",
                "—" if prec != prec else f"{prec:.2f}",
                pstr, nm),
                tags=("present" if present else "absent",))
        if hasattr(self, "_identify_status"):
            self._identify_status.configure(
                text=f"frame {fi}: {n_present} phase(s) ≥ {conf_min:.2f} "
                     f"of {len(tr['phases'])}")

        self._update_phase_summary()

    def _update_phase_summary(self):
        """Fill the materials-found summary: one row per phase with the number of
        frames it's present in (confidence >= bar AND >=3 matched reflections) and
        its median pressure over those frames. Also stashes per-phase present-frame
        index lists on self._phase_frames for the frames-list browser."""
        import numpy as np
        summary = getattr(self, "_identify_phase_summary", None)
        tr = getattr(self, "_identify_tr", None)
        if summary is None:
            return
        summary.delete(*summary.get_children())
        self._phase_frames = {}
        if not tr or not tr.get("ok"):
            return
        try:
            conf_min = max(0.0, min(1.0, float(self._identify_conf_var.get())))
        except (ValueError, AttributeError):
            conf_min = 0.5

        rows = []
        for rec in tr["phases"]:
            name = rec["name"]
            conf = rec.get("confidence")
            if conf is None:
                continue
            conf = np.asarray(conf, dtype=float)
            present = conf >= conf_min
            nmatch = rec.get("n_matched")
            if nmatch is not None:
                nmatch = np.asarray(nmatch)
                present = present & (nmatch >= 3)
            present_idx = np.nonzero(present)[0]
            self._phase_frames[name] = [int(i) for i in present_idx]
            n_present = int(present_idx.size)
            if n_present:
                pressure = np.asarray(rec.get("pressure"), dtype=float)
                med_p = float(np.nanmedian(pressure[present_idx]))
                med_p_str = "—" if med_p != med_p else f"{med_p:.1f}"
            else:
                med_p_str = "—"
            rows.append((n_present, name, med_p_str))

        rows.sort(key=lambda t: (-t[0], t[1].lower()))
        for n_present, name, med_p_str in rows:
            summary.insert("", "end", iid=name, values=(name, n_present, med_p_str))

    def _on_phase_summary_select(self, event=None):
        import numpy as np
        summary = getattr(self, "_identify_phase_summary", None)
        listbox = getattr(self, "_identify_frames_list", None)
        tr = getattr(self, "_identify_tr", None)
        if summary is None or listbox is None:
            return
        sel = summary.selection()
        listbox.delete(0, "end")
        self._phase_frame_indices = []
        if not sel:
            return
        name = sel[0]
        indices = self._phase_frames.get(name, [])

        conf_arr = None
        pressure_arr = None
        if tr and tr.get("ok"):
            for rec in tr["phases"]:
                if rec["name"] == name:
                    if rec.get("confidence") is not None:
                        conf_arr = np.asarray(rec["confidence"], dtype=float)
                    if rec.get("pressure") is not None:
                        pressure_arr = np.asarray(rec["pressure"], dtype=float)
                    break

        for i in indices:
            conf_txt = ""
            if conf_arr is not None and i < len(conf_arr):
                conf_txt = f"   conf {conf_arr[i]:.2f}"
            p_txt = ""
            if pressure_arr is not None and i < len(pressure_arr):
                p = pressure_arr[i]
                if p == p:
                    p_txt = f"   P {p:.1f}"
            listbox.insert("end", f"frame {i}{conf_txt}{p_txt}")
            self._phase_frame_indices.append(int(i))

    def _on_phase_frame_activate(self, event=None):
        listbox = getattr(self, "_identify_frames_list", None)
        if listbox is None:
            return
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        indices = getattr(self, "_phase_frame_indices", [])
        if idx >= len(indices):
            return
        frame = indices[idx]
        self._identify_frame_var.set(str(frame))
        self._update_identify_table()

    # ------------------------------------------------------------------
    # Helpers shared by Tab 9
    # ------------------------------------------------------------------

    def _enabled_phase_objects(self):
        """Return Phase objects for names in config candidate_phases, resolved from the library."""
        from .phases import load_library
        ws = self._phases_workspace()
        try:
            library = load_library(ws)
        except Exception:
            return []
        names = self.config.get("candidate_phases", [])
        return [library[n] for n in names if n in library]

    # Result caches so toggling a plot option doesn't recompute reflection
    # simulation / ROI integration. Keyed on the file's mtime — a re-run
    # invalidates them automatically.

    @staticmethod
    def _analysis_mtime(path) -> int:
        try:
            return Path(path).stat().st_mtime_ns
        except OSError:
            return 0

    def _cached_tracks(self, path, phase):
        cache = getattr(self, "_tracks_cache", None)
        if cache is None:
            cache = self._tracks_cache = {}
        key = (str(path), self._analysis_mtime(path), phase.name)
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            from .heatmap import reflection_tracks
            cache[key] = reflection_tracks(path, phase)
        return cache[key]

    def _cached_layers(self, path, phases):
        cache = getattr(self, "_layers_cache", None)
        if cache is None:
            cache = self._layers_cache = {}
        key = (str(path), self._analysis_mtime(path),
               tuple(sorted(p.name for p in phases)))
        if key not in cache:
            if len(cache) > 16:
                cache.clear()
            from .heatmap import phase_layers
            cache[key] = phase_layers(path, phases)
        return cache[key]

    # ------------------------------------------------------------------
    # Tab 9 — Pattern map (Hrubiak/XDI-style waterfall + tracks + layers)
    # ------------------------------------------------------------------

    def _tab_patternmap(self, frame):
        tk, ttk = self.tk, self.ttk

        # Controls row 1
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 2))

        ttk.Button(row1, text="Load pattern map",
                   command=self.load_pattern_map).pack(side="left", padx=4)
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_patternmap_fig", None), "Pattern map")
                   ).pack(side="left", padx=4)

        ttk.Label(row1, text="Source:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        from .heatmap import SOURCES as _PM_SOURCES
        self._pm_source = ttk.Combobox(
            row1,
            values=list(_PM_SOURCES),
            state="readonly", width=12,
        )
        self._pm_source.set("clean")
        self._pm_source.pack(side="left", padx=2)
        self._pm_source.bind("<<ComboboxSelected>>",
                             lambda e: self.load_pattern_map())

        ttk.Label(row1, text="X axis:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._pm_xaxis = ttk.Combobox(
            row1,
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11,
        )
        self._pm_xaxis.set("frame")
        self._pm_xaxis.pack(side="left", padx=2)
        self._pm_xaxis.bind("<<ComboboxSelected>>",
                            lambda e: self.load_pattern_map())

        self._pm_tracks = tk.BooleanVar(value=True)
        _trk_cb = ttk.Checkbutton(
            row1, text="Reflection tracks",
            variable=self._pm_tracks, command=self.load_pattern_map,
        )
        _trk_cb.pack(side="left", padx=8)
        _ToolTip(_trk_cb, "Predicted reflection positions of the enabled phases. "
                          "Drawn on the frame axis only.")

        self._pm_layers = tk.BooleanVar(value=False)
        _lay_cb = ttk.Checkbutton(
            row1, text="Phase layers",
            variable=self._pm_layers, command=self.load_pattern_map,
        )
        _lay_cb.pack(side="left", padx=4)
        _ToolTip(_lay_cb, "Second panel: per-phase matched-reflection intensity "
                          "vs the chosen x variable.")

        self._pm_status = ttk.Label(row1, text="", style="Muted.TLabel")
        self._pm_status.pack(side="right", padx=8)

        # Controls row 2
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))

        ttk.Label(row2, text="Export →", style="Muted.TLabel").pack(
            side="left", padx=(4, 6)
        )
        ttk.Button(
            row2,
            text="Refinement bundle…",
            command=self.export_refinement_clicked,
        ).pack(side="left", padx=2)
        ttk.Button(
            row2,
            text="GSAS raw patterns…",
            command=self.export_gsas_raw_clicked,
        ).pack(side="left", padx=2)
        ttk.Button(row2, text="Export ML dataset…",
                   command=self.export_ml_clicked).pack(side="left", padx=2)
        ttk.Button(row2, text="Export simulated set…",
                   command=self.export_sim_clicked).pack(side="left", padx=2)

        self._pm_pymatgen = ttk.Label(row2, text="", style="Muted.TLabel", wraplength=600,
                                      justify="left")
        self._pm_pymatgen.pack(side="left", padx=12)

        # Plot area
        self.patternmap_plot_frame = ttk.Frame(frame)
        self.patternmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.patternmap_plot_frame,
            text="Run the pipeline or Load pattern map to view the pattern waterfall.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

    def load_pattern_map(self):
        """Render the pattern waterfall (and optional tracks/layers) from the analysis HDF5."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return  # silently skip auto-calls

        # prev-figure-close leak guard
        prev = getattr(self, "_patternmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._patternmap_fig = None

        for w in self.patternmap_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.patternmap_plot_frame,
                text=f"matplotlib unavailable: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        from .heatmap import pattern_image
        from .phases import pymatgen_available

        # Update pymatgen hint label
        if hasattr(self, "_pm_pymatgen"):
            if pymatgen_available():
                self._pm_pymatgen.configure(text="", style="Muted.TLabel")
            else:
                self._pm_pymatgen.configure(
                    text=(
                        "pymatgen not installed — reflection tracks, phase layers, and the "
                        "simulated set are disabled (waterfall + ML export of measured data "
                        "still work)"
                    ),
                    style="Warn.TLabel",
                )

        x_axis = getattr(self._pm_xaxis, "get", lambda: "frame")()
        if not x_axis:
            x_axis = "frame"
        img = pattern_image(path, source=self._pm_source.get(), x_axis=x_axis)
        if not img["ok"]:
            self.ttk.Label(
                self.patternmap_plot_frame,
                text=img["error"],
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            if hasattr(self, "_pm_status"):
                self._pm_status.configure(text=img["error"])
            return

        show_layers = bool(self._pm_layers.get()) and pymatgen_available()

        if show_layers:
            fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
            ax = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
        else:
            fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
            ax = fig.add_subplot(1, 1, 1)
            ax2 = None

        fig.patch.set_facecolor(theme.C.BG)
        self._patternmap_fig = fig

        # Waterfall
        Z = img["Z"]                      # (n_bins, n_frames)
        radial = img["radial"]
        n = img["n_frames"]
        x_label = img.get("x_label") or "Frame index"
        xv = (np.asarray(img["x"], dtype=float) if img.get("x") is not None
              else np.arange(n, dtype=float))

        pos = Z[np.isfinite(Z) & (Z > 0)]
        if pos.size:
            vmin = float(np.percentile(pos, 5))
            vmax = float(np.percentile(pos, 99))
        else:
            vmin = None
            vmax = None

        if x_axis == "frame":
            # Uniform grid → imshow; nearest keeps frame columns crisp.
            ax.imshow(
                Z, aspect="auto", origin="lower", cmap="magma",
                interpolation="nearest",
                extent=[-0.5, float(n) - 0.5,
                        float(radial.min()), float(radial.max())],
                vmin=vmin, vmax=vmax,
            )
        else:
            # Physical (possibly non-uniform) coordinates → pcolormesh on the
            # frames sorted by x; imshow would silently pretend the values are
            # evenly spaced.
            fin = np.isfinite(xv)
            if fin.sum() < 2:
                msg = f"Fewer than two frames have a finite {x_axis} value."
                self.ttk.Label(self.patternmap_plot_frame, text=msg,
                               style="Warn.TLabel").pack(anchor="center", expand=True)
                if hasattr(self, "_pm_status"):
                    self._pm_status.configure(text=msg)
                return
            order = np.argsort(xv[fin], kind="stable")
            xs = xv[fin][order]
            Zs = Z[:, fin][:, order]
            ax.pcolormesh(xs, radial, Zs, cmap="magma", shading="nearest",
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(unit_label(img["unit"]))
        ax.set_title(f"Pattern waterfall — {img['source']}", color=theme.C.FG)
        self._style_ax(ax)

        # Reflection-track overlays (frame x-axis only — on a sorted physical
        # axis, frames are reordered and track curves would not align)
        if self._pm_tracks.get() and pymatgen_available() and x_axis == "frame":
            any_phase_plotted = False
            for phase_obj in self._enabled_phase_objects():
                tr = self._cached_tracks(path, phase_obj)
                if not tr["ok"]:
                    continue
                phase_color = None
                first_track = True
                for track in tr["tracks"]:
                    centers = track["centers"]
                    if not np.any(np.isfinite(centers)):
                        continue
                    x_coords = np.arange(n, dtype=float)
                    if first_track:
                        (ln,) = ax.plot(
                            x_coords, centers, lw=0.6, alpha=0.7,
                            label=phase_obj.name,
                        )
                        phase_color = ln.get_color()
                        first_track = False
                        any_phase_plotted = True
                    else:
                        ax.plot(x_coords, centers, lw=0.6, alpha=0.7,
                                color=phase_color, label="_nolegend_")
            if any_phase_plotted:
                ax.legend(fontsize=7, framealpha=0.4)

        # Per-phase intensity on the bottom axis, vs the same x variable.
        if show_layers and ax2 is not None:
            pl = self._cached_layers(path, self._enabled_phase_objects())
            if pl["ok"]:
                for layer in pl["layers"]:
                    y = np.asarray(layer["intensity"], dtype=float)
                    lx = xv[:y.size]
                    m = np.isfinite(lx) & np.isfinite(y)
                    o = np.argsort(lx[m], kind="stable")
                    ax2.plot(lx[m][o], y[m][o], lw=0.8, marker=".",
                             markersize=2, label=layer["name"])
                ax2.set_xlabel(x_label)
                ax2.set_ylabel(INTENSITY_ARB_LABEL)
                handles2, _ = ax2.get_legend_handles_labels()
                if handles2:
                    ax2.legend(fontsize=7, framealpha=0.4)
            else:
                ax2.set_title(pl["error"], color=theme.C.WARN)
            self._style_ax(ax2)

        self._patternmap_canvas = self._embed_figure(self.patternmap_plot_frame, fig)

        if hasattr(self, "_pm_status"):
            self._pm_status.configure(
                text=f"{img['n_frames']} frames × {radial.size} bins")
        self._attach_hover(self._patternmap_canvas, self._pm_status)

    # ------------------------------------------------------------------
    # Tab 11 — Unknowns (Step-3c stacked cluster diagram)
    # ------------------------------------------------------------------

    def _tab_unknowns(self, frame):
        tk, ttk = self.tk, self.ttk

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 4))
        ttk.Button(row1, text="Load unknowns",
                   command=self.load_unknowns).pack(side="left", padx=4)
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_unknowns_fig", None), "Unknowns")
                   ).pack(side="left", padx=4)
        ttk.Label(row1, text="Show:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._unk_show = ttk.Combobox(
            row1, values=["unknown clusters", "spot tracks d(P)"],
            state="readonly", width=16)
        self._unk_show.set("unknown clusters")
        self._unk_show.pack(side="left", padx=2)
        self._unk_show.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())
        _ToolTip(self._unk_show, (
            "unknown clusters — the Step-3c residual-peak co-occurrence "
            "diagram.  spot tracks d(P) — the seriesxrd-spots crystallite "
            "reflections as d-spacing vs pressure curves (one per grain "
            "reflection; RISING curves = d grows under pressure, negative "
            "linear compressibility). Pick an hkl table to label the curves."))
        _hkl_btn = ttk.Button(row1, text="hkl table…",
                              command=self.pick_spot_match_table)
        _hkl_btn.pack(side="left", padx=2)
        _ToolTip(_hkl_btn, (
            "Calculated reflection list (d/I pairs or an 'h k l d … I' table) "
            "used to label the spot-track d(P) curves with hkl assignments. "
            "Remembered in the session config."))
        ttk.Label(row1, text="X axis:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._unk_xaxis = ttk.Combobox(
            row1,
            values=["frame", "pressure", "temperature", "time"],
            state="readonly", width=11,
        )
        self._unk_xaxis.set("frame")
        self._unk_xaxis.pack(side="left", padx=2)
        self._unk_xaxis.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())

        ttk.Label(row1, text="Color by:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self._unk_color = ttk.Combobox(
            row1,
            values=["center", "amplitude", "track", "group"],
            state="readonly", width=10,
        )
        self._unk_color.set("center")
        self._unk_color.pack(side="left", padx=2)
        self._unk_color.bind("<<ComboboxSelected>>", lambda e: self.load_unknowns())

        ttk.Label(row1, text="Min obs/cluster:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self._unk_min_obs = tk.StringVar(value="1")
        _min_entry = ttk.Entry(row1, textvariable=self._unk_min_obs, width=5)
        _min_entry.pack(side="left", padx=2)
        _min_entry.bind("<Return>", lambda e: self.load_unknowns())
        _ToolTip(_min_entry, "Minimum residual-peak observations per unknown cluster.")
        ttk.Label(row1, text="Min frames/cluster:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self._unk_min_frames = tk.StringVar(value="1")
        _min_frames_entry = ttk.Entry(row1, textvariable=self._unk_min_frames, width=5)
        _min_frames_entry.pack(side="left", padx=2)
        _min_frames_entry.bind("<Return>", lambda e: self.load_unknowns())
        _ToolTip(
            _min_frames_entry,
            "Minimum distinct frames supporting the cluster; useful for hiding short bursts.",
        )
        ttk.Button(row1, text="Refresh", command=self.load_unknowns).pack(
            side="left", padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Button(row2, text="Run spot tracking…",
                   command=self.run_spot_tracking_clicked).pack(side="left", padx=2)
        ttk.Label(row2, text="Export →", style="Muted.TLabel").pack(side="left", padx=(4, 6))
        ttk.Button(row2, text="Diagram CSV…",
                   command=self.export_unknown_diagram_clicked).pack(side="left", padx=2)
        ttk.Button(row2, text="Frames with unknowns…",
                   command=self.export_unknown_frames_clicked).pack(side="left", padx=2)
        _spot_btn = ttk.Button(row2, text="Spot tracks…",
                               command=self.export_spot_tracks_clicked)
        _spot_btn.pack(side="left", padx=2)
        _ToolTip(_spot_btn, (
            "Export the /spots single-crystal tracks (seriesxrd-spots) as a "
            "handoff CSV bundle: per-track summary (+ optional hkl matches "
            "against a calculated reflection list), long-format d(P) point "
            "tables, untracked single-band reflections, and a README with "
            "provenance."))
        _mask_btn = ttk.Button(row2, text="Spot masks…",
                               command=self.export_spot_masks_clicked)
        _mask_btn.pack(side="left", padx=2)
        _ToolTip(_mask_btn, (
            "Keep-only detector masks from the kept /spots blobs (everything "
            "else masked, pyFAI convention) + GSAS-ready .xy patterns "
            "re-integrated from the masked raw images. Filters: tracked-only "
            "and/or the hkl match table."))
        self._unknowns_status = ttk.Label(row2, text="", style="Muted.TLabel")
        self._unknowns_status.pack(side="left", padx=12)

        self.unknowns_plot_frame = ttk.Frame(frame)
        self.unknowns_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.unknowns_plot_frame,
            text="Load unknowns after Step 3a residual + Step 3c has run.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

    def _unknown_min_obs_value(self) -> int:
        try:
            return max(1, int(float(self._unk_min_obs.get())))
        except Exception:
            return 1

    def _unknown_min_frames_value(self) -> int:
        try:
            return max(1, int(float(self._unk_min_frames.get())))
        except Exception:
            return 1

    def _unknown_filtered_clusters(self, data=None):
        if data is None:
            data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            return []
        min_obs = self._unknown_min_obs_value()
        min_frames = self._unknown_min_frames_value()
        return [
            c for c in data.get("clusters", [])
            if int(c.get("n_obs", 0)) >= min_obs
            and int(c.get("n_frames_observed", 0)) >= min_frames
        ]

    def _unknown_selected_frames(self, data=None):
        import numpy as np
        if data is None:
            data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            return []
        keep_clusters = {
            int(c["cluster"]) for c in self._unknown_filtered_clusters(data)
        }
        frame = np.asarray(data["frame"], dtype=int)
        cluster = np.asarray(data["cluster"], dtype=int)
        keep = np.array([int(c) in keep_clusters for c in cluster], dtype=bool)
        return sorted(set(int(f) for f in frame[keep].tolist()))

    def load_unknowns(self):
        """Render Step-3c unknown clusters as a stacked phase diagram."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            return

        prev = getattr(self, "_unknowns_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._unknowns_fig = None

        for w in self.unknowns_plot_frame.winfo_children():
            w.destroy()

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            self.ttk.Label(
                self.unknowns_plot_frame,
                text=f"matplotlib unavailable: {e}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            return

        import numpy as np

        show = getattr(self._unk_show, "get", lambda: "unknown clusters")()
        if (show or "").startswith("spot tracks"):
            self._render_spot_tracks(path, Figure, FigureCanvasTkAgg, np)
            return

        from .heatmap import unknown_diagram
        x_axis = getattr(self._unk_xaxis, "get", lambda: "frame")() or "frame"
        data = unknown_diagram(path, x_axis=x_axis)
        self._unknowns_data = data
        if not data.get("ok"):
            err = data.get("error", "unknown error")
            self.ttk.Label(
                self.unknowns_plot_frame,
                text=f"Unknowns: {err}",
                style="Warn.TLabel",
            ).pack(anchor="center", expand=True)
            self._unknowns_status.configure(text=err)
            return

        clusters = self._unknown_filtered_clusters(data)
        cluster_ids = [int(c["cluster"]) for c in clusters]
        row_of = {cid: i for i, cid in enumerate(cluster_ids)}

        frame = np.asarray(data["frame"], dtype=int)
        x = np.asarray(data["x"], dtype=float)
        cluster = np.asarray(data["cluster"], dtype=int)
        center = np.asarray(data["center"], dtype=float)
        amp = np.asarray(data["amplitude"], dtype=float)
        track = np.asarray(data["track"], dtype=float)
        group = np.asarray(data.get("group", np.zeros(frame.size)), dtype=float)
        keep = np.isfinite(x) & np.array([int(c) in row_of for c in cluster], dtype=bool)
        x_plot = x[keep]
        y_plot = np.array([row_of[int(c)] for c in cluster[keep]], dtype=float)
        color_by = getattr(self._unk_color, "get", lambda: "center")() or "center"
        c_plot = {"amplitude": amp, "track": track, "group": group}.get(color_by, center)[keep]

        fig_h = 5.0 if len(clusters) <= 80 else 6.5
        fig = Figure(figsize=(8, fig_h), dpi=100, layout="constrained")
        self._unknowns_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if not clusters or x_plot.size == 0:
            ax.set_title("No unknown clusters to display with current filter", color=theme.C.FG)
            ax.set_xlabel(data.get("x_label") or FRAME_LABEL)
            ax.set_ylabel("Unknown cluster")
        else:
            for c in clusters:
                ci = int(c["cluster"])
                row = row_of[ci]
                x0, x1 = c.get("x_min"), c.get("x_max")
                if x0 == x0 and x1 == x1:
                    ax.hlines(row, float(x0), float(x1), color=theme.C.MUTED,
                              lw=0.8, alpha=0.35)

            sizes = np.full(x_plot.size, 26.0)
            amp_plot = amp[keep]
            finite_amp = amp_plot[np.isfinite(amp_plot) & (amp_plot > 0)]
            if finite_amp.size:
                lo, hi = float(np.percentile(finite_amp, 10)), float(np.percentile(finite_amp, 95))
                if hi > lo:
                    sizes = 18.0 + 34.0 * np.clip((amp_plot - lo) / (hi - lo), 0, 1)
            sc = ax.scatter(
                x_plot, y_plot, c=c_plot, s=sizes,
                cmap="viridis", alpha=0.9, edgecolors=theme.C.FG, linewidths=0.25,
            )
            try:
                cb = fig.colorbar(sc, ax=ax, label=color_by)
                self._style_colorbar(cb)
            except Exception:
                pass
            ax.set_xlabel(data.get("x_label") or FRAME_LABEL)
            ax.set_ylabel("Unknown cluster")
            ax.set_title(
                f"Unknown clusters — {len(clusters)} cluster(s), "
                f"{int(x_plot.size)} observation(s)",
                color=theme.C.FG,
            )
            if len(clusters) <= 35:
                ax.set_yticks(np.arange(len(clusters)))
                labels = []
                for rec in clusters:
                    gl = str(rec.get("group_label", "") or "")
                    labels.append(f"{gl}:{rec['cluster']}" if gl else str(rec["cluster"]))
                ax.set_yticklabels(labels)
            else:
                ticks = np.unique(np.linspace(0, len(clusters) - 1,
                                             min(18, len(clusters))).astype(int))
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(cluster_ids[i]) for i in ticks])
            ax.set_ylim(-0.75, len(clusters) - 0.25)

        canvas = self._embed_figure(self.unknowns_plot_frame, fig)
        n_frames = len(self._unknown_selected_frames(data))
        self._unknowns_status.configure(
            text=(f"{len(clusters)} clusters, {int(x_plot.size)}/{data['n_obs']} obs, "
                  f"{n_frames} frame(s) with unknowns"))
        self._attach_hover(canvas, self._unknowns_status)

    def run_spot_tracking_clicked(self):
        """Detect and link single-crystal spots from the saved 2D cakes."""
        if self._run_proc is not None or getattr(self, "_analysis_tool_busy", False):
            self.messagebox.showinfo("Busy", "Wait for the current analysis tool to finish.")
            return
        self.pull_vars()
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        analysis = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            self.messagebox.showerror(
                "Spot tracking", "Select reduced data with saved 2D cakes first.")
            return
        if not analysis or not Path(analysis).is_file():
            self.messagebox.showerror("Spot tracking", "Run the analysis first.")
            return
        from tkinter import simpledialog
        group_by = simpledialog.askstring(
            "Spot tracking",
            "Independent series grouping: none, scan, or folder",
            initialvalue=str(self.config.get("spot_group_by", "none") or "none"),
            parent=self.root,
        )
        if group_by is None:
            return
        group_by = group_by.strip().lower()
        if group_by not in ("none", "scan", "folder"):
            self.messagebox.showerror(
                "Spot tracking", "Grouping must be none, scan, or folder.")
            return
        self.config["spot_group_by"] = group_by
        self.save_config(silent=True)
        self._analysis_tool_busy = True
        self._unknowns_status.configure(text="Tracking spots …", style="Muted.TLabel")
        box: "Dict[str, Any]" = {}

        def _work():
            try:
                from .spots import run_spot_tracking
                box["result"] = run_spot_tracking(reduced, analysis, group_by=group_by)
            except Exception as exc:
                box["error"] = str(exc)

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

        def _poll():
            if thread.is_alive():
                self.root.after(250, _poll)
                return
            self._analysis_tool_busy = False
            if box.get("error"):
                self._unknowns_status.configure(text=box["error"], style="Warn.TLabel")
                self.log(f"Spot tracking failed: {box['error']}", "ERROR")
                return
            result = box.get("result") or {}
            status = (
                f"{result.get('n_tracks', 0)} tracks from "
                f"{result.get('n_obs', 0)} observations"
            )
            self._unknowns_status.configure(text=status, style="Muted.TLabel")
            self.log(f"Spot tracking complete: {status}")
            self.load_unknowns()

        self.root.after(250, _poll)

    def pick_spot_match_table(self):
        """Choose the calculated-reflection table used to hkl-label spot tracks."""
        path = self.filedialog.askopenfilename(
            title="Calculated reflection table (Cancel = clear)",
            filetypes=[("Reflection tables", "*.txt *.csv *.dat"),
                       ("All files", "*.*")],
        )
        self.config["spot_match_file"] = path or ""
        self.save_config(silent=True)
        if (getattr(self._unk_show, "get", lambda: "")() or "").startswith("spot"):
            self.load_unknowns()

    def _render_spot_tracks(self, path, Figure, FigureCanvasTkAgg, np):
        """Render one d(P) curve per tracked crystallite reflection."""
        from .spots import load_spot_tracks
        match = str(self.config.get("spot_match_file", "") or "").strip() or None
        data = load_spot_tracks(path, min_points=self._unknown_min_frames_value(),
                                match=match)
        if not data.get("ok"):
            err = data.get("error", "unknown error")
            self.ttk.Label(self.unknowns_plot_frame,
                           text=f"Spot tracks: {err}", style="Warn.TLabel"
                           ).pack(anchor="center", expand=True)
            self._unknowns_status.configure(text=err)
            return
        tracks = data["tracks"]

        fig = Figure(figsize=(8, 5.5), dpi=100, layout="constrained")
        self._unknowns_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax = fig.add_subplot(1, 1, 1)
        self._style_ax(ax)

        if not tracks:
            ax.set_title("No spot tracks pass the filter — lower 'Min "
                         "frames/cluster' or run spot tracking", color=theme.C.FG)
        else:
            try:
                from matplotlib import colormaps
                cmap = colormaps["twilight"]         # azimuth is periodic
            except Exception:                        # older matplotlib
                from matplotlib import cm
                cmap = cm.get_cmap("twilight")
            n_pts = int(sum(t["n_points"] for t in tracks))
            n_nlc = 0
            for t in tracks:
                rising = t["dd_dp"] > 5e-4
                n_nlc += int(rising)
                color = cmap(((t["azim"] + 180.0) % 360.0) / 360.0)
                ax.plot(t["pressure"], t["d"], "-", color=color,
                        lw=2.2 if rising else 1.1,
                        alpha=0.95 if rising else 0.75, zorder=3 if rising else 2)
                ax.plot(t["pressure"], t["d"], "^" if rising else "o",
                        color=color, ms=5.5 if rising else 3.5,
                        mec=theme.C.FG, mew=0.3, ls="none", zorder=4)
                label = t["hkl"] or (f"az{t['azim']:+.0f}°" if len(tracks) <= 25 else "")
                if label:
                    ax.annotate(f" {label}", (t["pressure"][-1], t["d"][-1]),
                                color=color, fontsize=7.5, va="center")
            un = data["untracked"]
            if un["pressure"].size:
                ax.plot(un["pressure"], un["d"], "x", color=theme.C.MUTED, ms=4,
                        alpha=0.5, ls="none", zorder=1,
                        label=f"untracked ({un['pressure'].size})")
                ax.legend(loc="best", fontsize=8, framealpha=0.3,
                          labelcolor=theme.C.FG, facecolor=theme.C.BG)
            ax.set_xlabel(PRESSURE_LABEL)
            ax.set_ylabel(D_SPACING_LABEL)
            ax.set_title(
                f"Crystallite spot tracks — {len(tracks)}/{data['n_tracks_total']} "
                f"track(s), {n_pts} points; ▲ rising = d grows with P "
                f"({n_nlc} NLC candidate(s))", color=theme.C.FG)

        canvas = self._embed_figure(self.unknowns_plot_frame, fig)
        self._unknowns_status.configure(
            text=(f"{len(tracks)} spot track(s) shown"
                  + (f", hkl labels from {Path(match).name}" if match
                     else " — pick an hkl table to label curves")))
        self._attach_hover(canvas, self._unknowns_status)

    def export_unknown_diagram_clicked(self):
        """Export observation + cluster summary CSVs for the Unknowns tab."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._unknowns_status.configure(text="No analysis file loaded.")
            return
        default = Path(self.config.get("export_frames_dir", "") or
                       (Path(path).parent / "unknown_diagram"))
        dest = self.filedialog.askdirectory(
            title="Export unknown diagram CSVs",
            initialdir=str(default.parent if default.parent.exists() else Path(path).parent),
        )
        if not dest:
            return
        x_axis = getattr(self._unk_xaxis, "get", lambda: "frame")() or "frame"
        try:
            from .heatmap import write_unknown_diagram_csv
            man = write_unknown_diagram_csv(
                path,
                dest,
                x_axis=x_axis,
                min_obs_per_cluster=self._unknown_min_obs_value(),
                min_frames_per_cluster=self._unknown_min_frames_value(),
            )
        except Exception as e:
            self.log(f"Unknown diagram export failed: {e!r}", "WARN")
            self._unknowns_status.configure(text=f"Export failed: {e}")
            return
        msg = (f"Exported unknown diagram: {man['n_clusters']} clusters, "
               f"{man['n_obs']} obs -> {dest}")
        self._unknowns_status.configure(text=msg)
        self.log(msg)

    def export_spot_tracks_clicked(self):
        """Export /spots single-crystal tracks as the group-handoff CSV bundle."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._unknowns_status.configure(text="No analysis file loaded.")
            return
        try:
            import h5py
            with h5py.File(path, "r") as h:
                has_spots = "spots" in h and "tracks" in h["spots"]
        except Exception:
            has_spots = False
        if not has_spots:
            self._unknowns_status.configure(
                text="No spot tracks are available — run spot tracking first.")
            return
        dest = self.filedialog.askdirectory(
            title="Export spot-track CSV bundle",
            initialdir=str(self.config.get("export_frames_dir", "")
                           or Path(path).parent),
        )
        if not dest:
            return
        match = self.filedialog.askopenfilename(
            title="Calculated reflection list for hkl matching (Cancel = skip)",
            filetypes=[("Reflection tables", "*.txt *.csv *.dat"), ("All files", "*.*")],
        ) or None
        try:
            from .spots import export_spot_tracks
            man = export_spot_tracks(path, dest, match=match,
                                     include_observations=True)
        except Exception as e:
            self.log(f"Spot-track export failed: {e!r}", "WARN")
            self._unknowns_status.configure(text=f"Export failed: {e}")
            return
        self.config["export_frames_dir"] = dest
        self.save_config(silent=True)
        msg = (f"Exported {man['n_tracks']} spot track(s), "
               f"{man['n_track_points']} points "
               f"(+{man['n_untracked_points']} untracked) -> {dest}")
        self._unknowns_status.configure(text=msg)
        self.log(msg)

    def export_spot_masks_clicked(self):
        """Keep-only detector masks + masked re-integration from /spots."""
        self.pull_vars()
        tk, ttk = self.tk, self.ttk
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._unknowns_status.configure(text="No analysis file loaded.")
            return
        best: list = []
        red_attr = ""
        try:
            import h5py
            with h5py.File(path, "r") as h:
                has_obs = "spots" in h and "obs" in h["spots"]
                if has_obs:
                    red_attr = str(h["spots"].attrs.get("source_reduced", ""))
                    tg = h["spots"].get("tracks")
                    if tg is not None and "best_frame" in tg:
                        best = sorted(set(int(v) for v in tg["best_frame"][:]))
        except Exception:
            has_obs = False
        if not has_obs:
            self._unknowns_status.configure(
                text="No spot observations are available — run spot tracking first.")
            return
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            reduced = red_attr
        if not reduced or not Path(reduced).is_file():
            self._unknowns_status.configure(
                text="Reduced HDF5 not found (its PONI supplies the mask "
                     "geometry).")
            return
        raw_default = str(self.config.get("spot_mask_dataset_dir", "") or "")
        if not raw_default:
            man_p = Path(str(Path(reduced).with_suffix("")) + ".manifest.json")
            if man_p.is_file():
                try:
                    import json as _json
                    raw_default = str(_json.loads(
                        man_p.read_text(encoding="utf-8")).get("dataset_dir", ""))
                except Exception:
                    pass
        match_file = str(self.config.get("spot_match_file", "") or "").strip()

        dlg = tk.Toplevel(self.root)
        dlg.title("Spot masks")
        dlg.configure(bg=theme.C.BG)
        dlg.transient(self.root)
        dlg.grab_set()
        content = ttk.Frame(dlg, padding=10)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text=(
            "Per-frame keep-only detector masks from the kept /spots blobs "
            "(.npy + .tif + preview PNG), plus GSAS-ready .xy patterns "
            "re-integrated from the masked raw images."),
            style="Muted.TLabel", wraplength=430, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(content, text="Frames").grid(row=1, column=0, sticky="w",
                                               pady=2)
        v_frames = tk.StringVar(value="best")
        _fr = ttk.Entry(content, textvariable=v_frames, width=36)
        _fr.grid(row=1, column=1, sticky="we")
        _ToolTip(_fr, "'best' = each track's best frame; or a comma-"
                      "separated frame list; blank = every frame with a "
                      "kept spot.")
        ttk.Label(content, text="Raw images root").grid(row=2, column=0,
                                                        sticky="w", pady=2)
        v_raw = tk.StringVar(value=raw_default)
        ttk.Entry(content, textvariable=v_raw, width=36).grid(
            row=2, column=1, sticky="we")

        def _browse_raw():
            d = self.filedialog.askdirectory(title="Raw image root folder")
            if d:
                v_raw.set(d)
        ttk.Button(content, text="Browse", command=_browse_raw).grid(
            row=2, column=2, padx=4)
        v_tracks = tk.BooleanVar(value=True)
        ttk.Checkbutton(content, text="Tracked observations only",
                        variable=v_tracks).grid(row=3, column=0, columnspan=2,
                                                sticky="w", pady=2)
        v_use_match = tk.BooleanVar(value=bool(match_file))
        _mt = ttk.Checkbutton(
            content,
            text=("Match table: " + (Path(match_file).name if match_file
                                     else "none (pick via 'hkl table…')")),
            variable=v_use_match, state="normal" if match_file else "disabled")
        _mt.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(content, text="tol").grid(row=4, column=1, sticky="e")
        v_tol = tk.StringVar(value="0.02")
        ttk.Entry(content, textvariable=v_tol, width=7).grid(row=4, column=2,
                                                             sticky="w")
        ttk.Label(content, text="Destination").grid(row=5, column=0,
                                                    sticky="w", pady=2)
        v_dest = tk.StringVar(value=str(
            Path(str(self.config.get("export_frames_dir", "")
                     or Path(path).parent)) / "spot_masks"))
        ttk.Entry(content, textvariable=v_dest, width=36).grid(
            row=5, column=1, sticky="we")

        def _browse_dest():
            d = self.filedialog.askdirectory(title="Mask export destination")
            if d:
                v_dest.set(d)
        ttk.Button(content, text="Browse", command=_browse_dest).grid(
            row=5, column=2, padx=4)

        def _go():
            spec = v_frames.get().strip().lower()
            if spec == "best":
                frames = best or None
            elif spec:
                try:
                    frames = [int(s) for s in spec.split(",") if s.strip()]
                except ValueError:
                    self.messagebox.showerror(
                        "Spot masks", "Frames must be 'best', blank, or a "
                                      "comma-separated integer list.",
                        parent=dlg)
                    return
            else:
                frames = None
            try:
                tol = float(v_tol.get())
            except ValueError:
                tol = 0.02
            dest = v_dest.get().strip()
            if not dest:
                self.messagebox.showerror("Spot masks",
                                          "Pick a destination folder.",
                                          parent=dlg)
                return
            raw = v_raw.get().strip() or None
            self.config["spot_mask_dataset_dir"] = raw or ""
            self.save_config(silent=True)
            dlg.destroy()
            try:
                from .spots import export_spot_masks
                man = export_spot_masks(
                    reduced, path, dest, frames=frames,
                    match=(match_file if (v_use_match.get() and match_file)
                           else None),
                    match_tol=tol, tracks_only=bool(v_tracks.get()),
                    dataset_dir=raw, integrate=bool(raw))
            except Exception as e:
                self.log(f"Spot-mask export failed: {e!r}", "WARN")
                self._unknowns_status.configure(text=f"Mask export failed: {e}")
                return
            msg = (f"Spot masks: {man['n_frames']} frame(s), "
                   f"{man['n_kept_obs']} kept blob(s) -> {dest}"
                   + ("" if raw else "  (no raw root — masks only, no .xy)"))
            self._unknowns_status.configure(text=msg)
            self.log(msg)

        btns = ttk.Frame(content)
        btns.grid(row=6, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Export", command=_go).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left",
                                                                  padx=4)
        content.columnconfigure(1, weight=1)

    def export_unknown_frames_clicked(self):
        """Export residual patterns for frames carrying unknown observations."""
        data = getattr(self, "_unknowns_data", None)
        if not data or not data.get("ok"):
            self.load_unknowns()
            data = getattr(self, "_unknowns_data", None)
        frames = self._unknown_selected_frames(data)
        if not frames:
            self._unknowns_status.configure(text="No unknown frames to export.")
            return
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        stem = Path(path).stem.replace("_analysis", "")
        default_root = Path(self.config.get("export_frames_dir", "") or "outputs")
        initial = default_root if default_root.exists() else Path(path).parent
        dest = self.filedialog.askdirectory(
            title=f"Export {len(frames)} frame(s) with unknowns",
            initialdir=str(initial),
        )
        if not dest:
            return
        self.config["export_frames_dir"] = dest
        self.save_config(silent=True)
        out_dir = Path(dest) / f"unknown_frames_{stem}"
        self._do_export_frames(
            frames, out_dir, source="residual", peaks=True,
            residual_unknowns=True, status_label=self._unknowns_status,
        )

    # ------------------------------------------------------------------
    # Tab 12 — Grid map (per-frame scalars on the 2D scan grid)
    # ------------------------------------------------------------------

    def _tab_gridmap(self, frame):
        tk, ttk = self.tk, self.ttk

        _gm_intro = ttk.Label(
            frame,
            text=("For mapping runs: frames collected as a raster over the sample "
                  "are refolded onto their 2D scan grid and coloured by a "
                  "per-frame value (total/ROI intensity, contamination, peak "
                  "count, P, T, or one phase's matched intensity)."),
            style="Muted.TLabel", justify="left", wraplength=760,
        )
        _gm_intro.pack(anchor="w", padx=4, pady=(0, 6))
        self.autowrap(_gm_intro)

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 2))
        ttk.Button(row1, text="Load grid map",
                   command=self.load_grid_map).pack(side="left", padx=4)
        ttk.Button(row1, text="Open in window",
                   command=lambda: self._open_plot_window(
                       getattr(self, "_gridmap_fig", None), "Grid map")
                   ).pack(side="left", padx=4)

        ttk.Label(row1, text="Value:", style="Muted.TLabel").pack(side="left", padx=(12, 2))
        self.vars["map_value"] = tk.StringVar(
            value=str(self.config.get("map_value", "total")))
        self._gm_value = ttk.Combobox(
            row1, textvariable=self.vars["map_value"],
            values=["total", "max", "contamination", "n_peaks",
                    "pressure", "temperature"],
            state="readonly", width=16)
        self._gm_value.pack(side="left", padx=2)
        _ToolTip(self._gm_value, HELP["map_value"])

        ttk.Label(row1, text="ROI min/max:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self.vars["map_roi_min"] = tk.StringVar(
            value=str(self.config.get("map_roi_min", "")))
        _roi_lo = ttk.Entry(row1, textvariable=self.vars["map_roi_min"], width=8)
        _roi_lo.pack(side="left", padx=1)
        _ToolTip(_roi_lo, HELP["map_roi_min"])
        self.vars["map_roi_max"] = tk.StringVar(
            value=str(self.config.get("map_roi_max", "")))
        _roi_hi = ttk.Entry(row1, textvariable=self.vars["map_roi_max"], width=8)
        _roi_hi.pack(side="left", padx=1)
        _ToolTip(_roi_hi, HELP["map_roi_max"])

        self._gm_status = ttk.Label(row1, text="", style="Muted.TLabel")
        self._gm_status.pack(side="right", padx=8)

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="Layout:", style="Muted.TLabel").pack(side="left", padx=(4, 2))
        self.vars["map_layout"] = tk.StringVar(
            value=str(self.config.get("map_layout", "scan lines")))
        _lay_c = ttk.Combobox(row2, textvariable=self.vars["map_layout"],
                              values=["scan lines", "coordinates"],
                              state="readonly", width=11)
        _lay_c.pack(side="left", padx=2)
        _ToolTip(_lay_c, HELP["map_layout"])

        ttk.Label(row2, text="Frames per line:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self.vars["map_line_len"] = tk.StringVar(
            value=str(self.config.get("map_line_len", "")))
        _len_e = ttk.Entry(row2, textvariable=self.vars["map_line_len"], width=8)
        _len_e.pack(side="left", padx=2)
        _ToolTip(_len_e, HELP["map_line_len"])

        ttk.Label(row2, text="Scan lines:", style="Muted.TLabel").pack(
            side="left", padx=(12, 2))
        self.vars["map_order"] = tk.StringVar(
            value=str(self.config.get("map_order", "horizontal")))
        _ord_c = ttk.Combobox(row2, textvariable=self.vars["map_order"],
                              values=["horizontal", "vertical"],
                              state="readonly", width=11)
        _ord_c.pack(side="left", padx=2)
        _ToolTip(_ord_c, HELP["map_order"])

        self.vars["map_serpentine"] = tk.BooleanVar(
            value=bool(self.config.get("map_serpentine", True)))
        _serp = ttk.Checkbutton(row2, text="Boustrophedon (serpentine)",
                                variable=self.vars["map_serpentine"])
        _serp.pack(side="left", padx=12)
        _ToolTip(_serp, HELP["map_serpentine"])

        self.gridmap_plot_frame = ttk.Frame(frame)
        self.gridmap_plot_frame.pack(fill="both", expand=True)
        ttk.Label(
            self.gridmap_plot_frame,
            text="Set the frames-per-line to your scan width and Load grid map.",
            style="Muted.TLabel",
        ).pack(anchor="center", expand=True)

    def load_grid_map(self):
        """Render the scan-grid map of the selected per-frame value."""
        self.pull_vars()
        self.save_config(silent=True)
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self._gm_status.configure(text="No analysis HDF5 — run Step 1 first.")
            return

        prev = getattr(self, "_gridmap_fig", None)
        if prev is not None:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(prev)
            except Exception:
                pass
            self._gridmap_fig = None
        for w in self.gridmap_plot_frame.winfo_children():
            w.destroy()

        def _fail(msg):
            self.ttk.Label(self.gridmap_plot_frame, text=msg, wraplength=520,
                           justify="left", style="Warn.TLabel").pack(
                anchor="center", expand=True)
            self._gm_status.configure(text=msg)

        layout = str(self.config.get("map_layout", "scan lines") or "scan lines")
        line_len = 0
        if not layout.startswith("coord"):
            try:
                line_len = int(str(self.config.get("map_line_len", "")).strip())
                if line_len <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                _fail("Enter the frames-per-line of your scan (a positive "
                      "integer), or switch Layout to 'coordinates' if your "
                      "frames carry stage positions.")
                return

        def _opt(key):
            raw = str(self.config.get(key, "")).strip()
            return float(raw) if raw else None
        try:
            roi_lo, roi_hi = _opt("map_roi_min"), _opt("map_roi_max")
        except ValueError:
            _fail("ROI min/max must be numbers (or blank).")
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg", force=False)
            from matplotlib.figure import Figure
        except Exception as e:
            _fail(f"matplotlib unavailable: {e}")
            return
        import numpy as np
        from .heatmap import frame_values, frame_grid, grid_map

        kind = str(self.config.get("map_value", "total") or "total")
        if kind.startswith("phase:"):
            name = kind.split(":", 1)[1].strip()
            match = [p for p in self._enabled_phase_objects() if p.name == name]
            if not match:
                _fail(f"Phase {name!r} is not enabled on the Phases tab.")
                return
            pl = self._cached_layers(path, match)
            if not pl["ok"] or not pl["layers"]:
                _fail(pl.get("error") or f"No layer for {name!r} — run Step 3a.")
                return
            values = np.asarray(pl["layers"][0]["intensity_raw"], dtype=float)
            label = f"phase intensity — {name}"
        else:
            fv = frame_values(path, kind, radial_min=roi_lo, radial_max=roi_hi)
            if not fv["ok"]:
                _fail(fv["error"])
                return
            values = fv["values"]
            label = fv["label"]

        if layout.startswith("coord"):
            # Automatic placement from per-frame stage coordinates.
            from .frame_metadata import read_frame_metadata
            from .heatmap import coordinate_grid
            meta = read_frame_metadata(path)
            if not meta.get("ok"):
                _fail(meta.get("error") or "Could not read frame metadata.")
                return
            cg = coordinate_grid(meta["pos_x"], meta["pos_y"])
            if not cg["ok"]:
                _fail(cg["error"] + " (Frame meta tab: Import CSV with "
                      "pos_x/pos_y columns, or Read X/Y from headers.)")
                return
            gidx = cg["grid"]
            grid = np.full(gidx.shape, np.nan)
            m = gidx >= 0
            grid[m] = values[gidx[m]]
            xc = np.asarray(cg["x_centers"], dtype=float)
            yc = np.asarray(cg["y_centers"], dtype=float)

            def _half(c):
                return float(np.median(np.diff(c))) / 2.0 if c.size > 1 else 0.5
            hx, hy = _half(xc), _half(yc)
            extent = [xc[0] - hx, xc[-1] + hx, yc[0] - hy, yc[-1] + hy]
            origin = "lower"
            xlab, ylab = "Stage x (mm)", "Stage y (mm)"
            title = f"{label} — from frame coordinates"
            base_txt = (f"{cg['n_placed']} frames on a "
                        f"{grid.shape[0]}×{grid.shape[1]} coordinate grid")
            if cg["n_collisions"]:
                base_txt += f" ({cg['n_collisions']} collision(s))"

            def _cell(event):
                c = int(np.argmin(np.abs(xc - event.xdata)))
                r = int(np.argmin(np.abs(yc - event.ydata)))
                return r, c
        else:
            order = str(self.config.get("map_order", "horizontal") or "horizontal")
            serp = bool(self.config.get("map_serpentine", True))
            kwargs = ({"n_cols": line_len} if order == "horizontal"
                      else {"n_rows": line_len})
            try:
                grid = grid_map(values, order=order, serpentine=serp, **kwargs)
                gidx = frame_grid(values.size, order=order, serpentine=serp,
                                  **kwargs)
            except ValueError as e:
                _fail(str(e))
                return
            extent = None
            origin = "upper"
            xlab, ylab = "Scan column", "Scan row"
            path_txt = "boustrophedon" if serp else "unidirectional"
            title = f"{label} — {order} lines, {path_txt}"
            n_pad = int(np.sum(gidx < 0))
            base_txt = (f"{values.size} frames on a {grid.shape[0]}×"
                        f"{grid.shape[1]} grid"
                        + (f" ({n_pad} empty cells)" if n_pad else ""))

            def _cell(event):
                return int(round(event.ydata)), int(round(event.xdata))

        fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
        self._gridmap_fig = fig
        fig.patch.set_facecolor(theme.C.BG)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(grid, origin=origin, cmap="viridis",
                       interpolation="nearest", aspect="equal", extent=extent)
        try:
            cb = fig.colorbar(im, ax=ax, label=label)
            self._style_colorbar(cb)
        except Exception:
            pass
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=theme.C.FG)
        self._style_ax(ax)

        canvas = self._embed_figure(self.gridmap_plot_frame, fig)
        self._gridmap_canvas = canvas
        self._gm_status.configure(text=base_txt)

        # Hover: resolve the cursor's grid cell back to the frame index.
        def _move(event):
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            r, c = _cell(event)
            if 0 <= r < gidx.shape[0] and 0 <= c < gidx.shape[1]:
                fi = int(gidx[r, c])
                if fi >= 0:
                    v = grid[r, c]
                    self._gm_status.configure(
                        text=f"{base_txt}   |   frame {fi}, value {v:.5g}")
        def _leave(event):
            self._gm_status.configure(text=base_txt)
        try:
            canvas.mpl_connect("motion_notify_event", _move)
            canvas.mpl_connect("axes_leave_event", _leave)
        except Exception:
            pass

    def _refresh_gridmap_values(self):
        """Extend the Value combo with per-phase entries for enabled phases."""
        base = ["total", "max", "contamination", "n_peaks",
                "pressure", "temperature"]
        names = [f"phase: {p.name}" for p in self._enabled_phase_objects()]
        try:
            self._gm_value.configure(values=base + names)
        except Exception:
            pass

    def export_refinement_clicked(self):
        """Export patterns, phase files, and instrument metadata for refinement."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror(
                "Refinement export",
                "No analysis results are available. Run the analysis first.",
            )
            return
        initial = str(Path(path).parent)
        destination = self.filedialog.askdirectory(
            title="Choose refinement export folder", initialdir=initial)
        if not destination:
            return
        stem = Path(path).stem.removesuffix("_analysis")
        out_dir = Path(destination) / f"{stem}_refinement"
        try:
            from .refine_export import export_refinement_bundle
            result = export_refinement_bundle(
                path,
                out_dir,
                workspace=self.config.get("workspace_root") or None,
            )
        except Exception as exc:
            self.messagebox.showerror("Refinement export", str(exc))
            return
        written = len(result.get("files_written", []))
        skipped = result.get("phases_skipped", [])
        message = (
            f"Exported {result.get('n_frames', 0)} patterns and {written} files to:\n"
            f"{out_dir}"
        )
        if skipped:
            message += f"\n\n{len(skipped)} phase file(s) could not be resolved."
        self.log(message.replace("\n", " "))
        self.notify(f"Refinement bundle: {result.get('n_frames', 0)} patterns, "
                    f"{written} files → {out_dir}"
                    + (f" ({len(skipped)} phase file(s) unresolved — see log)"
                       if skipped else ""))

    def export_gsas_raw_clicked(self):
        """Re-integrate raw frames with uncertainties for GSAS import."""
        if self._run_proc is not None or getattr(self, "_analysis_tool_busy", False):
            self.messagebox.showinfo("Busy", "Wait for the current analysis tool to finish.")
            return
        self.pull_vars()
        reduced = str(self.config.get("reduced_h5_file", "") or "").strip()
        analysis = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            self.messagebox.showerror(
                "GSAS export", "Select reduced data before exporting raw patterns.")
            return

        group_by_pressure = self.messagebox.askyesnocancel(
            "GSAS export",
            "Group frames recorded at the same pressure?\n\n"
            "Yes: one summed pattern per pressure\n"
            "No: one pattern per frame",
        )
        if group_by_pressure is None:
            return
        if group_by_pressure and (not analysis or not Path(analysis).is_file()):
            self.messagebox.showerror(
                "GSAS export",
                "Pressure grouping requires analysis results with frame metadata.",
            )
            return
        destination = self.filedialog.askdirectory(
            title="Choose GSAS export folder", initialdir=str(Path(reduced).parent))
        if not destination:
            return

        out_dir = Path(destination) / f"{Path(reduced).stem}_gsas"
        self._analysis_tool_busy = True
        self._pm_status.configure(text="Exporting GSAS patterns …", style="Muted.TLabel")
        box: "Dict[str, Any]" = {}

        def _work():
            try:
                from .refine_export import export_gsas_raw
                box["result"] = export_gsas_raw(
                    reduced,
                    out_dir,
                    analysis_h5=analysis if analysis else None,
                    group_by_pressure=bool(group_by_pressure),
                )
            except Exception as exc:
                box["error"] = str(exc)

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

        def _poll():
            if thread.is_alive():
                self.root.after(250, _poll)
                return
            self._analysis_tool_busy = False
            if box.get("error"):
                self._pm_status.configure(text="GSAS export failed", style="Warn.TLabel")
                self.log(f"GSAS export failed: {box['error']}", "ERROR")
                self.messagebox.showerror("GSAS export", box["error"])
                return
            result = box.get("result") or {}
            count = int(result.get("n_groups", 0))
            self._pm_status.configure(text=f"Exported {count} GSAS pattern(s)", style="Muted.TLabel")
            self.log(f"GSAS raw patterns exported: {out_dir}")
            self.notify(f"GSAS export: {count} pattern(s) → {out_dir}")

        self.root.after(250, _poll)

    def import_gsas_results_clicked(self):
        """Import sequential GSAS-II results without assuming a DAC protocol."""
        self.pull_vars()
        analysis = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not analysis or not Path(analysis).is_file():
            self.messagebox.showerror(
                "Import GSAS-II results",
                "Run Step 1 or select an existing analysis HDF5 first.",
            )
            return
        results = self.filedialog.askopenfilename(
            title="Choose GSAS-II sequential results",
            initialdir=str(Path(analysis).parent),
            filetypes=[
                ("GSAS-II / SeriesXRD results", "*.gpx *.json"),
                ("GSAS-II project", "*.gpx"),
                ("SeriesXRD refinement JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not results:
            return
        try:
            from .refine_import import import_gsasii_results
            result = import_gsasii_results(analysis, results)
        except Exception as exc:
            self.log(f"GSAS-II result import failed: {exc!r}", "ERROR")
            self.messagebox.showerror("Import GSAS-II results", str(exc))
            return

        message = (
            f"Imported {result['mapped_histograms']} sequential histogram(s) "
            f"into {result['n_frames_mapped']} frame(s) for "
            f"{len(result['phases'])} phase(s).\n\n"
            "Refined weight fractions, uncertainties, cells, and fit quality "
            "are now under /refinement. Existing /fractions screening "
            "estimates were preserved."
        )
        if result["unmapped_histograms"]:
            message += (f"\n\nUnmapped histograms: "
                        f"{len(result['unmapped_histograms'])}")
        if result["warnings"]:
            message += "\n\nWarnings:\n" + "\n".join(result["warnings"][:8])
        self.log(message.replace("\n", " "))
        self.notify(
            f"GSAS-II results: {result['n_frames_mapped']} frame(s), "
            f"{len(result['phases'])} phase(s) → /refinement",
        )
        self.messagebox.showinfo("Import GSAS-II results", message)

    def export_ml_clicked(self):
        """Export the analysis frames as an ML-ready .npz dataset."""
        self.pull_vars()
        path = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not path or not Path(path).is_file():
            self.messagebox.showerror(
                "Export ML dataset",
                "No analysis HDF5 found. Run the pipeline or set the path on the Input tab.")
            return

        out = self.filedialog.asksaveasfilename(
            title="Export ML dataset",
            defaultextension=".npz",
            filetypes=[("NumPy npz", "*.npz")],
        )
        if not out:
            return

        from . import mldata
        try:
            man = mldata.export_ml_dataset(
                path, out,
                channels=("clean", "spot_residual"),
                normalize=True,
            )
        except Exception as e:
            self.messagebox.showerror("Export ML dataset failed", repr(e))
            return

        self.log(
            f"ML dataset exported: {man['n_frames']} frames × {man['n_channels']} channels "
            f"→ {out}  labels: {'yes' if man['has_labels'] else 'no'}"
        )
        self.notify(f"ML export: {man['n_frames']} frames × "
                    f"{man['n_channels']} channels → {out} "
                    f"(labels: {'yes' if man['has_labels'] else 'no'})")

    def export_sim_clicked(self):
        """Export a pressure-augmented simulated training set as .npz."""
        from .phases import pymatgen_available
        if not pymatgen_available():
            self.messagebox.showinfo(
                "Export simulated set",
                "pymatgen is required to simulate XRD patterns.\n"
                "Install it with:  pip install pymatgen")
            return

        phases = self._enabled_phase_objects()
        if not phases:
            self.messagebox.showinfo(
                "Export simulated set",
                "Enable candidate phases on the Phases tab first.")
            return

        out = self.filedialog.asksaveasfilename(
            title="Export simulated training set",
            defaultextension=".npz",
            filetypes=[("NumPy npz", "*.npz")],
        )
        if not out:
            return

        import numpy as np
        try:
            pmin = float(self.config.get("p_min", 0) or 0)
        except (ValueError, TypeError):
            pmin = 0.0
        try:
            pmax = float(self.config.get("p_max", 100) or 100)
        except (ValueError, TypeError):
            pmax = 100.0
        pressures = np.linspace(pmin, pmax, 21)

        from . import mldata
        try:
            man = mldata.export_simulated_dataset(out, phases, pressures=pressures)
        except Exception as e:
            self.messagebox.showerror("Export simulated set failed", repr(e))
            return

        self.log(
            f"Simulated dataset exported: {man['n_samples']} patterns "
            f"({len(man['phases'])} phases) → {out}"
        )
        self.notify(f"Simulated set: {man['n_samples']} patterns "
                    f"({len(man['phases'])} phases) → {out}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def confirm_shutdown(self) -> bool:
        """Return whether the pane may close without changing pane state."""
        if getattr(self, "_analysis_tool_busy", False):
            self.messagebox.showinfo(
                "Processing in progress",
                "An analysis tool is still running. Wait for it to finish before closing.",
            )
            return False
        if self._run_proc is not None and self._run_proc.poll() is None:
            return self.messagebox.askyesno(
                "Analysis running", "Analysis is still running. Stop it and close?",
            )
        return True

    def shutdown(self, confirm: bool = True) -> bool:
        """Save and tear down. Returns False if the user cancelled."""
        if confirm and not self.confirm_shutdown():
            return False
        if self._run_proc is not None and self._run_proc.poll() is None:
            terminate_process_tree(self._run_proc)
        self._closing = True  # stop the log-drain poller from rescheduling
        self.save_config(silent=True)
        return True

    def on_close(self):
        if not self.shutdown(confirm=True):
            return
        if self._owns_root:
            self.root.destroy()


# ---------------------------------------------------------------------------
# Public factory / entry-point functions
# ---------------------------------------------------------------------------

def make_analysis_pane(
    parent_frame, config_path: "str | Path"
) -> "AnalysisApp":
    """Construct AnalysisApp embedded in a parent frame (for the unified app)."""
    return AnalysisApp(config_path, parent=parent_frame)


def run_app(config_path: "str | Path") -> int:
    """Standalone entry point."""
    from ..guikit.dpi import enable_hi_dpi
    enable_hi_dpi()
    app = AnalysisApp(config_path)
    assert app._owns_root, "run_app is the standalone entry point and must own the root"
    app.root.mainloop()
    return 0
