"""Native Tk interface for the SeriesXRD correlation stage.

The pane is embeddable in the unified application and can also own a
standalone root. Scientific work remains in ``seriesxrd.correlations.batch``;
this module supervises that subprocess and lazily previews its PNG outputs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
from typing import Any, Dict, Mapping

from ..core.config import TOOL_NAME, ensure_dir, now_iso, read_json, write_json
from ..core.processes import terminate_process_tree, worker_popen
from ..guikit import theme
from ..guikit.tkstyle import apply_theme
from ..guikit.tooltip import ToolTip as _ToolTip


SAMPLE_TYPES = ("powder", "single_crystal")
SOURCES = (
    "fit", "auto", "clean", "hybrid", "mean", "sigmaclip", "spots", "residual",
)
PAGE_KEYS = ("input", "settings", "run", "results")

RESULT_CATEGORY_LABELS = {
    "roi_area": "ROI area",
    "location": "Peak location",
    "waterfall": "Waterfall",
    "window_across": "Window across frames",
    "window_within": "Window within frame",
}
RESULT_CATEGORY_ORDER = {
    key: index for index, key in enumerate(RESULT_CATEGORY_LABELS)
}
RESULT_FILTER_ALL = "All diagrams"


def _pressure_label(folder: str) -> "tuple[str, float]":
    """Decode ``pressure_3p5_GPa`` into a display label and sort value."""

    text = str(folder or "")
    if text == "pressure_unknown":
        return "Pressure unavailable", float("inf")
    match = re.fullmatch(r"pressure_(.+)_GPa", text)
    if match is None:
        return "Pressure unavailable", float("inf")
    token = match.group(1).replace("m", "-").replace("p", ".")
    try:
        value = float(token)
    except ValueError:
        return "Pressure unavailable", float("inf")
    return f"{value:g} GPa", value


def _load_result_pressures(result_root: Path) -> "dict[tuple[str, int], float]":
    """Read original-frame pressure labels from sample-specific artifacts."""

    pressure_by_frame: "dict[tuple[str, int], float]" = {}
    try:
        import h5py  # type: ignore
    except Exception:
        return pressure_by_frame
    for sample_type in SAMPLE_TYPES:
        artifact = result_root / f"correlations_{sample_type}.h5"
        if not artifact.is_file():
            continue
        try:
            with h5py.File(str(artifact), "r") as h5:
                frame = h5.get("frames")
                if frame is None or "index" not in frame or "pressure" not in frame:
                    continue
                indices = frame["index"][:]
                pressures = frame["pressure"][:]
                if len(indices) != len(pressures):
                    continue
                for index, pressure in zip(indices, pressures):
                    try:
                        value = float(pressure)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        pressure_by_frame[(sample_type, int(index))] = value
        except (OSError, ValueError):
            continue
    return pressure_by_frame


def _classify_result_path(
    path: Path,
    result_root: Path,
    pressure_by_frame: "Mapping[tuple[str, int], float] | None" = None,
) -> Dict[str, Any]:
    """Return searchable hierarchy metadata for one generated PNG."""

    relative = path.relative_to(result_root)
    parts = relative.parts
    pressure_by_frame = pressure_by_frame or {}
    try:
        heatmap_index = parts.index("heatmaps")
        sample_type = parts[heatmap_index + 1]
        category = parts[heatmap_index + 2]
    except (ValueError, IndexError):
        sample_type = "other"
        category = "other"

    sample_label = {
        "powder": "Powder",
        "single_crystal": "Single crystal",
    }.get(sample_type, sample_type.replace("_", " ").title() or "Other")
    category_label = RESULT_CATEGORY_LABELS.get(
        category, category.replace("_", " ").title() or "Other diagrams",
    )
    pressure_label = "Pressure unavailable"
    pressure_value = float("inf")
    method_label = ""

    if category in ("roi_area", "location", "waterfall"):
        try:
            pressure_label, pressure_value = _pressure_label(parts[heatmap_index + 3])
        except IndexError:
            pass
    elif category == "window_across":
        pressure_label = "All pressures"
        pressure_value = float("-inf")
        try:
            method_label = str(parts[heatmap_index + 3]).upper()
        except IndexError:
            pass
    elif category == "window_within":
        try:
            method_label = str(parts[heatmap_index + 3]).upper()
        except IndexError:
            pass
        match = re.fullmatch(r"frame_(\d+)", path.stem)
        if match is not None:
            frame_index = int(match.group(1))
            pressure = pressure_by_frame.get((sample_type, frame_index))
            if pressure is not None:
                pressure_value = float(pressure)
                pressure_label = f"{pressure_value:g} GPa"

    if path.stem.startswith("anchor_"):
        leaf_label = path.stem.replace("anchor_", "Anchor ", 1)
    elif path.stem.startswith("frame_"):
        leaf_label = path.stem.replace("frame_", "Frame ", 1)
    elif path.stem.startswith("window_"):
        fields = path.stem.split("_")
        leaf_label = path.stem.replace("_", " ").title()
        if len(fields) >= 4:
            leaf_label = f"Window {fields[1]} — {fields[2]}–{fields[3]}"
    else:
        leaf_label = path.stem.replace("_", " ").title()

    searchable = " ".join(
        (
            sample_label,
            sample_type,
            category_label,
            category,
            pressure_label,
            method_label,
            leaf_label,
            str(relative),
        )
    ).casefold()
    return {
        "path": path,
        "relative": relative,
        "sample_type": sample_type,
        "sample_label": sample_label,
        "category": category,
        "category_label": category_label,
        "pressure_label": pressure_label,
        "pressure_value": pressure_value,
        "method_label": method_label,
        "leaf_label": leaf_label,
        "searchable": searchable,
        "sort_key": (
            0 if sample_type == "powder" else 1 if sample_type == "single_crystal" else 2,
            RESULT_CATEGORY_ORDER.get(category, len(RESULT_CATEGORY_ORDER)),
            pressure_value,
            method_label,
            leaf_label.casefold(),
            str(relative).casefold(),
        ),
    }


def _result_matches(entry: Mapping[str, Any], query: str, category_label: str) -> bool:
    """Return whether one result leaf passes the Results-page filters."""

    if category_label and category_label != RESULT_FILTER_ALL:
        if str(entry.get("category_label", "")) != category_label:
            return False
    words = [
        word
        for word in re.sub(r"[-_]+", " ", str(query or "").casefold()).split()
        if word
    ]
    searchable = re.sub(
        r"[-_]+", " ", str(entry.get("searchable", "")).casefold()
    )
    return all(word in searchable for word in words)


def _find_result_paths(result_root: Path) -> "list[Path]":
    """Index only complete, sample-specific heatmap trees.

    Atomic rendering may leave hidden ``.powder.tmp-*`` or ``.powder.old-*``
    siblings after a hard process termination. Those trees and unrelated PNGs
    are not published results and must never appear in the reviewer.
    """

    paths = []
    for sample_type in SAMPLE_TYPES:
        sample_root = result_root / "heatmaps" / sample_type
        if sample_root.is_dir():
            paths.extend(path for path in sample_root.rglob("*.png") if path.is_file())
    return sorted(
        paths,
        key=lambda path: str(path.relative_to(result_root)).casefold(),
    )


def _tk_imports():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    return tk, ttk, filedialog, messagebox


class CorrelationApp:
    """Configure, run, and review the Log²-only correlation workflow."""

    def __init__(self, config_path: "str | Path", parent=None):
        tk, ttk, filedialog, messagebox = _tk_imports()
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: Dict[str, Any] = read_json(self.config_path)
        self.config.setdefault("session_config_path", str(self.config_path))
        self.config.setdefault("analysis_h5_file", "")
        self.config.setdefault("result_root", "")
        self.config.setdefault("sample_type", "powder")
        self.config.setdefault(
            "source",
            "spots" if self.config.get("sample_type") == "single_crystal" else "fit",
        )
        self.config.setdefault("radial_min", "")
        self.config.setdefault("radial_max", "")
        self.config.setdefault("window_width", "5.0")
        self.config.setdefault("window_step", "1.0")
        self.config.setdefault("location_tolerance", "0.02")
        self.config["transform"] = "log_squared"

        if parent is None:
            self._owns_root = True
            self.root = tk.Tk()
            self.root.title(f"{TOOL_NAME} Correlations")
            self.root.geometry("1180x780")
            self.root.minsize(960, 640)
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        else:
            self._owns_root = False
            self.root = parent.winfo_toplevel()
        self._embed_parent = parent

        self.vars: Dict[str, Any] = {}
        self._run_proc: "subprocess.Popen | None" = None
        self._active_result_root: "Path | None" = None
        self._review_result_root: "Path | None" = None
        self._cancel_requested = False
        self._closing = False
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._event_queue: "queue.Queue[tuple]" = queue.Queue()
        self._log_history: "list[str]" = []
        self._log_window = None
        self._preview_after_id = None
        self._result_filter_after_id = None
        self._poll_after_id = None
        self._result_paths: "dict[str, Path]" = {}
        self._result_entries: "list[Dict[str, Any]]" = []
        self._preview_photo = None

        self._build_gui()
        theme.register_widget_tree(self._embed_parent or self.root)
        theme.register_restyle(self._restyle_theme)
        self._schedule_queue_poll()
        self.log("Correlation GUI initialized")
        self.save_config(silent=True)
        self._update_input_status()

    # ------------------------------------------------------------------
    # Build and theme
    # ------------------------------------------------------------------

    def _restyle_theme(self):
        apply_theme(self.root, self.ttk)
        theme.register_widget_tree(self._embed_parent or self.root)
        if self._log_window is not None:
            try:
                theme.register_widget_tree(self._log_window)
            except Exception:
                pass
        theme.restyle_widgets()

    def _build_gui(self):
        ttk = self.ttk
        if self._owns_root:
            apply_theme(self.root, ttk)
        container = self._embed_parent if self._embed_parent is not None else self.root
        outer = ttk.Frame(container, padding=6)
        outer.pack(fill="both", expand=True)

        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 6))
        ttk.Label(
            topbar, text="Correlations", font=("TkDefaultFont", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            topbar, text="Log² only", style="Accent.TLabel",
        ).pack(side="left", padx=12)
        ttk.Button(
            topbar, text="View log", command=self.open_console_logs,
        ).pack(side="right", padx=4)

        self._status_bar = ttk.Label(
            outer, text="idle", style="Muted.TLabel", anchor="w",
            relief="sunken",
        )
        self._status_bar.pack(side="bottom", fill="x", pady=(2, 0), padx=1)
        self._build_navigation(outer)

    def _build_navigation(self, outer):
        ttk = self.ttk
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
                ("input", "Input", self._page_input),
                ("settings", "Settings", self._page_settings),
            ]),
            ("Run", [
                ("run", "Run correlations", self._page_run),
            ]),
            ("Review", [
                ("results", "Results", self._page_results),
            ]),
        ]

        self.pages: Dict[str, Any] = {}
        self._nav_items: "dict[str, str]" = {}
        self._nav_first_child: "dict[str, str]" = {}
        for section_label, page_specs in sections:
            section_id = rail.insert("", "end", text=section_label, open=True)
            for key, label, builder in page_specs:
                frame = ttk.Frame(content, padding=10)
                builder(frame)
                frame.grid(row=0, column=0, sticky="nsew")
                self.pages[key] = frame
                item_id = rail.insert(
                    section_id, "end", text=label, values=(key,), tags=(key,),
                )
                self._nav_items[key] = item_id
                self._nav_first_child.setdefault(section_id, key)

        def _on_select(_event=None):
            selected = rail.selection()
            if not selected:
                return
            item_id = selected[0]
            values = rail.item(item_id, "values")
            if values:
                key = str(values[0])
                self.pages[key].tkraise()
                if key == "results":
                    self._schedule_preview()
            else:
                first = self._nav_first_child.get(item_id)
                if first:
                    rail.selection_set(self._nav_items[first])

        rail.bind("<<TreeviewSelect>>", _on_select)
        self.select_page("input")

    def select_page(self, key: str) -> None:
        item = self._nav_items.get(key)
        if item is None:
            return
        self._nav_rail.selection_set(item)
        self._nav_rail.see(item)
        self.pages[key].tkraise()

    def _field(self, parent, key: str, label: str, *, row: int, browse=None):
        ttk, tk = self.ttk, self.tk
        variable = tk.StringVar(value=str(self.config.get(key, "") or ""))
        self.vars[key] = variable
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=4, pady=4,
        )
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        if browse:
            ttk.Button(
                parent, text="Browse…",
                command=lambda: self._browse_into(key, browse),
            ).grid(row=row, column=2, sticky="w", padx=4, pady=4)
        parent.columnconfigure(1, weight=1)
        return entry

    def _page_input(self, frame):
        ttk = self.ttk
        analysis_entry = self._field(
            frame, "analysis_h5_file", "Analysis HDF5", row=0, browse="file",
        )
        _ToolTip(
            analysis_entry,
            "The Analysis-stage HDF5 containing radial profiles, frame metadata, and peaks.",
        )
        result_entry = self._field(
            frame, "result_root", "Result folder", row=1, browse="dir",
        )
        _ToolTip(
            result_entry,
            "Generated matrices and heatmaps stay in the workspace. Powder "
            "and single crystal use separate sample-specific HDF5 and manifest "
            "files, so one shared result folder is safe.",
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=3, sticky="w", padx=2, pady=8)
        ttk.Button(
            buttons, text="Check input", command=self._update_input_status,
        ).pack(side="left", padx=2)
        self._input_status = ttk.Label(
            frame, text="", style="Muted.TLabel", justify="left", wraplength=760,
        )
        self._input_status.grid(
            row=3, column=0, columnspan=3, sticky="nw", padx=4, pady=6,
        )

    def _page_settings(self, frame):
        ttk, tk = self.ttk, self.tk
        ttk.Label(frame, text="Sample type").grid(
            row=0, column=0, sticky="w", padx=4, pady=4,
        )
        sample_var = tk.StringVar(value=str(self.config.get("sample_type", "powder")))
        self.vars["sample_type"] = sample_var
        sample_box = ttk.Combobox(
            frame, textvariable=sample_var, values=SAMPLE_TYPES,
            state="readonly", width=22,
        )
        sample_box.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        sample_box.bind("<<ComboboxSelected>>", self._sample_type_changed)

        ttk.Label(frame, text="Profile source").grid(
            row=1, column=0, sticky="w", padx=4, pady=4,
        )
        source_var = tk.StringVar(value=str(self.config.get("source", "fit")))
        self.vars["source"] = source_var
        ttk.Combobox(
            frame, textvariable=source_var, values=SOURCES,
            state="readonly", width=22,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Intensity transform").grid(
            row=2, column=0, sticky="w", padx=4, pady=4,
        )
        transform = ttk.Label(
            frame, text="Log² (fixed)", style="Accent.TLabel",
        )
        transform.grid(row=2, column=1, sticky="w", padx=4, pady=4)
        _ToolTip(
            transform,
            "The correlation pipeline uses fixed bounded Log² preprocessing.",
        )

        self._field(frame, "radial_min", "Radial minimum (optional)", row=3)
        self._field(frame, "radial_max", "Radial maximum (optional)", row=4)
        self._field(
            frame, "window_width", "Window width (native radial unit)", row=5,
        )
        self._field(
            frame, "window_step", "Window step (native radial unit)", row=6,
        )
        self._field(
            frame, "location_tolerance", "Peak-location tolerance", row=7,
        )
        note = ttk.Label(
            frame,
            text=(
                "Both sample types consume the Analysis HDF5. Powder ROI maps "
                "require /peaks; single-crystal ROI maps require /spots/obs. "
                "Single crystal defaults to the spots profile source and uses "
                "a 1D radial ROI approximation, not raw detector-pixel ROIs. "
                "Create /spots/obs with seriesxrd-spots REDUCED.h5 --analysis "
                "ANALYSIS.h5. The batch preflight reports missing prerequisites."
            ),
            style="Muted.TLabel", justify="left", wraplength=760,
        )
        note.grid(row=8, column=0, columnspan=3, sticky="w", padx=4, pady=(12, 4))
        frame.columnconfigure(1, weight=1)

    def _page_run(self, frame):
        ttk = self.ttk
        ttk.Label(
            frame,
            text=(
                "Run ROI-area, peak-location, and window correlations in an "
                "isolated worker process. Anchor-frame peak cells are left "
                "blank. The user interface remains responsive."
            ),
            style="Muted.TLabel", justify="left", wraplength=760,
        ).pack(anchor="w", pady=(0, 10))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="w")
        self.run_button = ttk.Button(
            buttons, text="Run correlations", style="Accent.TButton",
            command=self.run_clicked,
        )
        self.run_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(
            buttons, text="Cancel", command=self._cancel_run, state="disabled",
        )
        self.cancel_button.pack(side="left", padx=4)
        self.run_status = ttk.Label(
            frame, text="Ready.", style="Muted.TLabel", justify="left",
            wraplength=760,
        )
        self.run_status.pack(anchor="w", fill="x", padx=4, pady=14)

    def _page_results(self, frame):
        ttk, tk = self.ttk, self.tk
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            toolbar, text="Refresh heatmaps", command=self.review_results,
        ).pack(side="left", padx=2)
        self.results_status = ttk.Label(
            toolbar, text="No results loaded.", style="Muted.TLabel",
        )
        self.results_status.pack(side="left", padx=10)

        filters = ttk.Frame(frame)
        filters.pack(fill="x", pady=(0, 8))
        ttk.Label(filters, text="Search").pack(side="left", padx=(2, 4))
        self._result_search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(
            filters, textvariable=self._result_search_var, width=30,
        )
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        _ToolTip(
            search_entry,
            "Search by diagram type, pressure, frame, anchor, window, or file path.",
        )
        ttk.Button(
            filters, text="Clear", width=7,
            command=lambda: self._result_search_var.set(""),
        ).pack(side="left", padx=(0, 10))
        ttk.Label(filters, text="Diagram").pack(side="left", padx=(0, 4))
        self._result_category_var = tk.StringVar(value=RESULT_FILTER_ALL)
        category_box = ttk.Combobox(
            filters,
            textvariable=self._result_category_var,
            values=(RESULT_FILTER_ALL, *RESULT_CATEGORY_LABELS.values()),
            state="readonly",
            width=23,
        )
        category_box.pack(side="left", padx=(0, 2))
        category_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._schedule_result_filter(delay_ms=0),
        )

        pane = ttk.Panedwindow(frame, orient="horizontal")
        pane.pack(fill="both", expand=True)
        list_frame = ttk.Frame(pane)
        preview_frame = ttk.Frame(pane)
        pane.add(list_frame, weight=1)
        pane.add(preview_frame, weight=4)

        tree = ttk.Treeview(list_frame, show="tree", selectmode="browse")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.column("#0", width=310, stretch=True)
        tree.heading("#0", text="Correlation diagrams")
        tree.bind("<<TreeviewSelect>>", lambda _event: self._schedule_preview())
        self.results_tree = tree
        self._result_search_var.trace_add(
            "write", lambda *_args: self._schedule_result_filter(),
        )

        self.preview_label = ttk.Label(
            preview_frame, text="Select a heatmap to preview it.",
            style="Muted.TLabel", anchor="center", justify="center",
        )
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)
        self.preview_path_label = ttk.Label(
            preview_frame, text="", style="Muted.TLabel", anchor="w",
        )
        self.preview_path_label.pack(fill="x", padx=8, pady=(0, 6))
        preview_frame.bind("<Configure>", lambda _event: self._schedule_preview())
        self._preview_frame = preview_frame

    # ------------------------------------------------------------------
    # Config and handoff
    # ------------------------------------------------------------------

    def _browse_into(self, key: str, mode: str):
        if mode == "dir":
            value = self.filedialog.askdirectory(title=f"Select {key}")
        else:
            value = self.filedialog.askopenfilename(
                title=f"Select {key}",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            )
        if value:
            self.vars[key].set(value)
            self.save_config(silent=True)
            if key == "analysis_h5_file":
                self._update_input_status()

    def _sample_type_changed(self, _event=None):
        """Apply the scientifically safe source default for the chosen sample."""
        sample_type = str(self.vars["sample_type"].get() or "powder")
        source = "spots" if sample_type == "single_crystal" else "fit"
        self.vars["source"].set(source)
        self.save_config(silent=True)
        self._update_input_status()

    def pull_vars(self):
        for key, variable in self.vars.items():
            self.config[key] = variable.get()
        # Never accept a stale or hand-edited alternative transform.
        self.config["transform"] = "log_squared"

    def save_config(self, silent: bool = False):
        self.pull_vars()
        self.config["updated_at"] = now_iso()
        write_json(self.config_path, self.config)
        if not silent:
            self.log(f"Config saved: {self.config_path}")

    def set_analysis(self, path: "str | Path") -> None:
        """Receive a completed Analysis HDF5 from the host application."""
        value = str(path or "").strip()
        if not value:
            return
        self.config["analysis_h5_file"] = value
        variable = self.vars.get("analysis_h5_file")
        if variable is not None:
            variable.set(value)
        self.save_config(silent=True)
        self.log(f"Analysis HDF5 received: {value}")
        self._update_input_status()
        self.select_page("input")

    def _update_input_status(self):
        self.pull_vars()
        raw = str(self.config.get("analysis_h5_file", "") or "").strip()
        if not raw:
            text = "No Analysis HDF5 selected. Complete Analysis or browse to an existing file."
            style = "Warn.TLabel"
        else:
            path = Path(raw).expanduser()
            if path.is_file():
                try:
                    size_mib = path.stat().st_size / (1024 * 1024)
                    unit = "radial unit unknown"
                    try:
                        import h5py  # type: ignore

                        with h5py.File(str(path), "r") as h5:
                            raw_unit = h5.attrs.get("unit", "")
                        if isinstance(raw_unit, bytes):
                            raw_unit = raw_unit.decode("utf-8", "replace")
                        if str(raw_unit).strip():
                            unit = f"native radial unit: {raw_unit}"
                    except Exception:
                        pass
                    text = f"Ready: {path.name} ({size_mib:.1f} MiB; {unit})"
                    style = "Ok.TLabel"
                except OSError:
                    text = f"Ready: {path.name}"
                    style = "Ok.TLabel"
            else:
                text = f"Analysis HDF5 not found: {path}"
                style = "Warn.TLabel"
        if hasattr(self, "_input_status"):
            self._input_status.configure(text=text, style=style)
        self._status_bar.configure(text=text)

    # ------------------------------------------------------------------
    # Run supervision
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_float(value: Any, label: str) -> "float | None":
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number or blank.") from exc

    @staticmethod
    def _positive_float(value: Any, label: str) -> float:
        try:
            number = float(str(value or "").strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a positive number.") from exc
        if not number > 0:
            raise ValueError(f"{label} must be greater than zero.")
        return number

    def _build_batch_command(self) -> "tuple[list[str], Path]":
        self.pull_vars()
        analysis = Path(
            str(self.config.get("analysis_h5_file", "") or "").strip()
        ).expanduser()
        if not analysis.is_file():
            raise ValueError("Select an existing Analysis HDF5 before running.")
        result_text = str(self.config.get("result_root", "") or "").strip()
        if not result_text:
            raise ValueError("Select a result folder before running.")
        result_root = ensure_dir(result_text)

        sample_type = str(self.config.get("sample_type", "powder") or "powder")
        source = str(self.config.get("source", "fit") or "fit")
        if sample_type not in SAMPLE_TYPES:
            raise ValueError(f"Unknown sample type: {sample_type}")
        if source not in SOURCES:
            raise ValueError(f"Unknown profile source: {source}")
        radial_min = self._optional_float(self.config.get("radial_min"), "Radial minimum")
        radial_max = self._optional_float(self.config.get("radial_max"), "Radial maximum")
        if radial_min is not None and radial_max is not None and radial_min >= radial_max:
            raise ValueError("Radial minimum must be less than radial maximum.")
        window_width = self._positive_float(self.config.get("window_width"), "Window width")
        window_step = self._positive_float(self.config.get("window_step"), "Window step")
        location_tolerance = self._positive_float(
            self.config.get("location_tolerance"), "Peak-location tolerance",
        )

        command = [
            sys.executable,
            "-m",
            "seriesxrd.correlations.batch",
            str(analysis.resolve()),
            "--out",
            str(result_root),
            "--sample-type",
            sample_type,
            "--source",
            source,
            "--window-width",
            str(window_width),
            "--window-step",
            str(window_step),
            "--location-tolerance",
            str(location_tolerance),
        ]
        if radial_min is not None:
            command.extend(("--radial-min", str(radial_min)))
        if radial_max is not None:
            command.extend(("--radial-max", str(radial_max)))
        return command, result_root

    def run_clicked(self):
        if self._run_proc is not None and self._run_proc.poll() is None:
            self.messagebox.showinfo(
                "Correlations running", "A correlation run is already in progress.",
            )
            return
        try:
            command, result_root = self._build_batch_command()
            self.save_config(silent=True)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Cannot run correlations", str(exc))
            return

        environment = dict(os.environ)
        environment.setdefault("MPLBACKEND", "Agg")
        try:
            process = worker_popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
        except OSError as exc:
            self.messagebox.showerror("Worker launch failed", str(exc))
            return

        self._run_proc = process
        self._cancel_requested = False
        self._active_result_root = result_root
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.run_status.configure(
            text="Running Log² correlations…", style="Accent.TLabel",
        )
        self._status_bar.configure(text="correlation worker: running")
        self.log("$ " + " ".join(command))
        threading.Thread(
            target=self._read_worker, args=(process,), daemon=True,
        ).start()

    def _read_worker(self, process: subprocess.Popen):
        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    self._log_queue.put(raw_line.rstrip())
            returncode = int(process.wait())
            self._event_queue.put(("done", returncode))
        except Exception as exc:
            self._event_queue.put(("error", repr(exc)))

    def _schedule_queue_poll(self):
        if self._closing:
            return
        self._poll_after_id = self.root.after(100, self._drain_queues)

    def _drain_queues(self):
        self._poll_after_id = None
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._insert_log_line(line)
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_worker_event(event)
        self._schedule_queue_poll()

    def _handle_worker_event(self, event: tuple):
        kind = event[0]
        active_result_root = self._active_result_root
        self._active_result_root = None
        self._run_proc = None
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if kind == "error":
            self.run_status.configure(text="Worker error.", style="Warn.TLabel")
            self._status_bar.configure(text="correlation worker: failed")
            self.messagebox.showerror("Correlation worker error", str(event[1]))
            return

        returncode = int(event[1])
        if self._cancel_requested:
            self._cancel_requested = False
            self.run_status.configure(text="Run cancelled.", style="Muted.TLabel")
            self._status_bar.configure(text="correlation worker: cancelled")
            self.log("Correlation run cancelled by user")
        elif returncode != 0:
            self.run_status.configure(
                text=f"Run failed (return code {returncode}). See the log.",
                style="Warn.TLabel",
            )
            self._status_bar.configure(text="correlation worker: failed")
            self.messagebox.showerror(
                "Correlation run failed",
                f"Worker return code {returncode}. See View log for details.",
            )
        else:
            self.run_status.configure(
                text="Correlation run complete.", style="Ok.TLabel",
            )
            self._status_bar.configure(text="correlation worker: done")
            self.log("Correlation run complete")
            self.review_results(
                show_errors=False,
                result_root=active_result_root,
            )

    def _cancel_run(self):
        process = self._run_proc
        if process is None or process.poll() is not None:
            return
        self._cancel_requested = True
        self.cancel_button.configure(state="disabled")
        self.run_status.configure(text="Cancelling…", style="Muted.TLabel")
        terminate_process_tree(process)

    # ------------------------------------------------------------------
    # Lazy result review
    # ------------------------------------------------------------------

    def _schedule_result_filter(self, delay_ms: int = 200) -> None:
        """Debounce text filtering so large result trees remain responsive."""

        if self._closing or not hasattr(self, "results_tree"):
            return
        after_id = self._result_filter_after_id
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._result_filter_after_id = None
        if delay_ms <= 0:
            self._apply_result_filters()
        else:
            self._result_filter_after_id = self.root.after(
                delay_ms, self._apply_result_filters
            )

    def _clear_result_browser(self, status: str, preview: str) -> None:
        """Clear stale results when the selected result folder cannot be read."""

        self._result_entries = []
        self._review_result_root = None
        self._apply_result_filters()
        self.results_status.configure(text=status)
        self._preview_photo = None
        self.preview_label.configure(
            image="", text=preview, style="Muted.TLabel",
        )
        self.preview_path_label.configure(text="")

    def _apply_result_filters(self) -> None:
        """Rebuild the classified result tree from the cached PNG index."""

        if not hasattr(self, "results_tree"):
            return
        after_id = self._result_filter_after_id
        self._result_filter_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        tree = self.results_tree
        selected_path = None
        selected = tree.selection()
        if selected:
            selected_path = self._result_paths.get(selected[0])
        for item in tree.get_children(""):
            tree.delete(item)
        self._result_paths.clear()

        query_var = getattr(self, "_result_search_var", None)
        category_var = getattr(self, "_result_category_var", None)
        query = str(query_var.get() if query_var is not None else "")
        category_filter = str(
            category_var.get() if category_var is not None else RESULT_FILTER_ALL
        )
        visible = [
            entry for entry in self._result_entries
            if _result_matches(entry, query, category_filter)
        ]

        groups: "dict[tuple[str, str], str]" = {}

        def _group(parent: str, key: str, label: str, *, opened: bool) -> str:
            cache_key = (parent, key)
            item = groups.get(cache_key)
            if item is None:
                item = tree.insert(parent, "end", text=label, open=opened)
                groups[cache_key] = item
            return item

        first_leaf = None
        selected_leaf = None
        expand_matches = bool(query.strip())
        for entry in visible:
            sample = _group(
                "",
                f"sample:{entry['sample_type']}",
                str(entry["sample_label"]),
                opened=True,
            )
            category = _group(
                sample,
                f"category:{entry['category']}",
                str(entry["category_label"]),
                opened=True,
            )
            pressure = _group(
                category,
                f"pressure:{entry['pressure_label']}",
                str(entry["pressure_label"]),
                opened=expand_matches,
            )
            parent = pressure
            method_label = str(entry.get("method_label", "") or "")
            if method_label:
                parent = _group(
                    pressure,
                    f"method:{method_label}",
                    method_label,
                    opened=expand_matches,
                )
            leaf = tree.insert(parent, "end", text=str(entry["leaf_label"]))
            self._result_paths[leaf] = Path(entry["path"])
            if first_leaf is None:
                first_leaf = leaf
            if selected_path == Path(entry["path"]):
                selected_leaf = leaf

        total = len(self._result_entries)
        shown = len(visible)
        status = (
            f"{total} correlation diagram PNG file(s)"
            if shown == total and not query.strip() and category_filter == RESULT_FILTER_ALL
            else f"{shown} of {total} correlation diagram PNG file(s)"
        )
        self.results_status.configure(text=status)
        self._preview_photo = None
        self.preview_path_label.configure(text="")
        chosen_leaf = selected_leaf or first_leaf
        if chosen_leaf is not None:
            tree.selection_set(chosen_leaf)
            tree.see(chosen_leaf)
            self._schedule_preview()
        else:
            self.preview_label.configure(
                image="",
                text=(
                    "No diagrams match the current search and diagram filter."
                    if total else "No correlation diagram PNGs were found in this result folder."
                ),
                style="Muted.TLabel",
            )

    def review_results(
        self,
        show_errors: bool = True,
        *,
        result_root: "str | Path | None" = None,
    ):
        if result_root is None:
            self.pull_vars()
            raw_root = str(self.config.get("result_root", "") or "").strip()
            if not raw_root:
                self._clear_result_browser(
                    "No result folder selected.",
                    "Select a result folder before reviewing diagrams.",
                )
                if show_errors:
                    self.messagebox.showinfo(
                        "Correlation results", "Select a result folder first."
                    )
                return
            result_root = Path(raw_root)
        result_root = Path(result_root).expanduser()
        if not result_root.is_dir():
            self._clear_result_browser(
                "Result folder not found.",
                "The selected result folder is unavailable.",
            )
            if show_errors:
                self.messagebox.showinfo(
                    "Correlation results", f"Result folder not found:\n{result_root}",
                )
            return

        try:
            paths = _find_result_paths(result_root)
        except OSError as exc:
            self._clear_result_browser(
                "Could not read result folder.",
                "The selected result folder could not be read.",
            )
            if show_errors:
                self.messagebox.showerror("Could not read results", str(exc))
            return

        self._review_result_root = result_root.resolve()
        pressure_by_frame = _load_result_pressures(result_root)
        self._result_entries = sorted(
            (
                _classify_result_path(path, result_root, pressure_by_frame)
                for path in paths
            ),
            key=lambda entry: entry["sort_key"],
        )
        self.preview_label.configure(
            image="", text="Select a heatmap to preview it.",
            style="Muted.TLabel",
        )
        self.preview_path_label.configure(text="")
        self._preview_photo = None
        self._apply_result_filters()
        self.select_page("results")

    def _schedule_preview(self):
        if self._closing or not hasattr(self, "results_tree"):
            return
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.root.after(100, self._preview_selected)

    def _preview_selected(self):
        self._preview_after_id = None
        selected = self.results_tree.selection()
        if not selected:
            return
        path = self._result_paths.get(selected[0])
        if path is None:
            self._preview_photo = None
            self.preview_label.configure(
                image="", text="Expand the group and select a correlation plot.",
                style="Muted.TLabel",
            )
            self.preview_path_label.configure(text="")
            return
        if not path.is_file():
            self._preview_photo = None
            self.preview_label.configure(
                image="",
                text="This heatmap no longer exists. Refresh the results.",
                style="Warn.TLabel",
            )
            self.preview_path_label.configure(text=str(path))
            return
        try:
            from PIL import Image, ImageTk

            width = max(320, int(self._preview_frame.winfo_width()) - 24)
            height = max(240, int(self._preview_frame.winfo_height()) - 56)
            with Image.open(path) as source:
                image = source.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((width, height), resampling)
            photo = ImageTk.PhotoImage(image, master=self.root)
        except Exception as exc:
            self.preview_label.configure(
                image="", text=f"Could not preview image:\n{exc}",
                style="Warn.TLabel",
            )
            self.preview_path_label.configure(text=str(path))
            self._preview_photo = None
            return
        self._preview_photo = photo
        self.preview_label.configure(image=photo, text="")
        try:
            relative = path.relative_to(self._review_result_root or path.parent)
        except (OSError, ValueError):
            relative = path
        self.preview_path_label.configure(text=str(relative))

    # ------------------------------------------------------------------
    # Logging and lifecycle
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO"):
        line = f"[{now_iso()}] [{level}] {message}"
        print(line, flush=True)
        if threading.current_thread() is threading.main_thread():
            self._insert_log_line(line)
        else:
            self._log_queue.put(line)

    def _insert_log_line(self, line: str):
        self._log_history.append(str(line))
        if len(self._log_history) > 5000:
            self._log_history = self._log_history[-5000:]
        text_widget = getattr(self, "log_text", None)
        if text_widget is not None:
            try:
                text_widget.configure(state="normal")
                text_widget.insert("end", str(line) + "\n")
                text_widget.see("end")
                text_widget.configure(state="disabled")
            except Exception:
                pass

    def open_console_logs(self):
        tk, ttk = self.tk, self.ttk
        if self._log_window is not None:
            try:
                if self._log_window.winfo_exists():
                    self._log_window.deiconify()
                    self._log_window.lift()
                    return
            except Exception:
                pass
        window = tk.Toplevel(self.root)
        window.title("Correlation log")
        window.geometry("860x520")
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        text_widget = tk.Text(
            window, bg=theme.C.BG2, fg=theme.C.FG,
            insertbackground=theme.C.FG, relief="flat", state="normal",
            wrap="none", font=("TkFixedFont", 9),
        )
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text_widget.pack(side="left", fill="both", expand=True)
        if self._log_history:
            text_widget.insert("end", "\n".join(self._log_history) + "\n")
            text_widget.see("end")
        text_widget.configure(state="disabled")
        self._log_window = window
        self.log_text = text_widget
        theme.register_widget_tree(window)

    def confirm_shutdown(self) -> bool:
        process = self._run_proc
        if process is not None and process.poll() is None:
            return self.messagebox.askyesno(
                "Correlations running",
                "A correlation run is still active. Stop it and close?",
            )
        return True

    def shutdown(self, confirm: bool = True) -> bool:
        if confirm and not self.confirm_shutdown():
            return False
        process = self._run_proc
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        self._closing = True
        for after_id in (
            self._poll_after_id,
            self._preview_after_id,
            self._result_filter_after_id,
        ):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        self._poll_after_id = None
        self._preview_after_id = None
        self._result_filter_after_id = None
        self.save_config(silent=True)
        return True

    def on_close(self):
        if not self.shutdown(confirm=True):
            return
        if self._owns_root:
            self.root.destroy()


def make_correlation_pane(
    parent_frame, config_path: "str | Path"
) -> CorrelationApp:
    """Construct an embedded correlation pane for the unified application."""
    return CorrelationApp(config_path, parent=parent_frame)


def run_app(config_path: "str | Path") -> int:
    """Run the correlation pane as a standalone Tk application."""
    from ..guikit.dpi import enable_hi_dpi

    enable_hi_dpi()
    app = CorrelationApp(config_path)
    assert app._owns_root, "run_app is the standalone entry point"
    app.root.mainloop()
    return 0
