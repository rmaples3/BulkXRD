"""Unified SeriesXRD desktop application.

The application presents calibration, reduction, analysis, and correlations
as one guided workflow. Tk imports remain deferred so the package can be
imported and tested on headless systems.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from .core.config import (
    TOOL_NAME, VERSION, ensure_dir, write_json,
    SessionConfig, config_path_for_notebook,
)
from .reduce.session import seed_reduction_config
from .analysis.session import seed_analysis_config
from .correlations.session import seed_correlation_config
from .guikit import theme

DEFAULT_WORKSPACE = Path.home() / "seriesxrd_workspace"
STAGE_TABS = (
    "1 Calibration",
    "2 Reduction",
    "3 Analysis",
    "4 Correlations",
)
REPO_URL = "https://github.com/RushMaples/SeriesXRD"


def workspace_launch_args(workspace: "str | Path", executable: str | None = None) -> list[str]:
    """Return the platform-neutral command used to open another workspace."""
    path = Path(workspace).expanduser().resolve()
    return [executable or sys.executable, "-m", "seriesxrd.app", "--workspace", str(path)]


def _workspace_display_name(path: "str | Path") -> str:
    """Return a concise workspace label without exposing its parent path."""
    workspace = Path(path)
    return workspace.name or workspace.anchor or str(workspace)


def _seed_calibration_config(workspace: Path) -> Path:
    """Ensure a calibration config exists in the workspace; return its path."""
    cfg_path = config_path_for_notebook(workspace)
    if not cfg_path.exists():
        cfg = SessionConfig.default(workspace).to_dict()
        cfg["workspace_root"] = str(workspace)
        cfg["session_config_path"] = str(cfg_path)
        write_json(cfg_path, cfg)
    return cfg_path


class SeriesXRDApp:
    """The unified host window."""

    def __init__(self, workspace: "str | Path"):
        import tkinter as tk
        from tkinter import ttk
        from .guikit.tkstyle import apply_theme
        from .calib.gui import make_calib_pane
        from .reduce.gui import make_reduce_pane
        from .analysis.gui import make_analysis_pane
        from .correlations.gui import make_correlation_pane

        self.tk = tk
        self.workspace = ensure_dir(workspace)
        calib_cfg = _seed_calibration_config(self.workspace)
        reduce_cfg = seed_reduction_config(self.workspace)
        analysis_cfg = seed_analysis_config(self.workspace)
        correlation_cfg = seed_correlation_config(self.workspace)

        self.root = tk.Tk()
        self.root.title(f"{TOOL_NAME} — {self.workspace.name}")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1280, sw - 80)}x{min(900, sh - 120)}")
        self.root.minsize(min(1000, sw - 80), min(700, sh - 120))
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        apply_theme(self.root, ttk)
        self.root.configure(bg=theme.C.BG)

        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(
            header, text=TOOL_NAME, font=("TkDefaultFont", 16, "bold"),
        ).pack(side="left")
        # Show only the workspace name. The full path remains available from
        # the tooltip and by clicking the label, without exposing it in images.
        from .guikit.tooltip import ToolTip
        self._ws_label = ttk.Label(
            header, text=f"  {_workspace_display_name(self.workspace)}",
            style="Muted.TLabel", cursor="hand2",
        )
        self._ws_label.pack(side="left")
        ToolTip(self._ws_label,
                f"Workspace: {self.workspace}\nClick to copy the full path.")
        self._ws_label.bind("<Button-1>", lambda _e: self._copy_workspace_path())
        ttk.Label(
            header,
            text="1  Calibrate   →   2  Reduce   →   3  Analyze   →   4  Correlate",
            style="Muted.TLabel",
        ).pack(side="right")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)

        calib_tab = ttk.Frame(self.nb)
        reduce_tab = ttk.Frame(self.nb)
        analysis_tab = ttk.Frame(self.nb)
        correlation_tab = ttk.Frame(self.nb)
        self.nb.add(calib_tab, text=STAGE_TABS[0])
        self.nb.add(reduce_tab, text=STAGE_TABS[1])
        self.nb.add(analysis_tab, text=STAGE_TABS[2])
        self.nb.add(correlation_tab, text=STAGE_TABS[3])

        self.calib_pane = make_calib_pane(calib_tab, calib_cfg)
        self.reduce_pane = make_reduce_pane(reduce_tab, reduce_cfg)
        self.calib_pane.add_accept_listener(self._on_calibration_accepted)

        self.analysis_pane = make_analysis_pane(analysis_tab, analysis_cfg)
        self.reduce_pane.add_reduced_listener(self._on_reduction_ready)

        self.correlation_pane = make_correlation_pane(
            correlation_tab, correlation_cfg)
        self.analysis_pane.add_analysis_listener(self._on_analysis_ready)

        self._build_menubar()
        theme.register_widget_tree(self.root)
        theme.register_restyle(self._restyle_theme)

    # ------------------------------------------------------------------

    def _build_menubar(self):
        tk = self.tk
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open workspace…", command=self._open_workspace)
        file_menu.add_command(label="Save all settings", accelerator="Ctrl+S", command=self._save_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_quit)
        self.root.bind("<Control-s>", lambda e: self._save_all())

        go_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Go", menu=go_menu)
        for index, label in enumerate(STAGE_TABS):
            go_menu.add_command(
                label=label, accelerator=f"Ctrl+{index + 1}",
                command=lambda i=index: self._select_stage(i),
            )
            self.root.bind(
                f"<Control-Key-{index + 1}>",
                lambda _event, i=index: self._select_stage(i),
            )

        calib_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Calibration", menu=calib_menu)
        calib_menu.add_command(label="Environment settings…", command=self.calib_pane.open_env_settings)
        calib_menu.add_command(label="Launch Dioptas", command=self.calib_pane.open_dioptas)
        calib_menu.add_command(label="View log", command=self.calib_pane.open_console_logs)

        reduction_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reduction", menu=reduction_menu)
        reduction_menu.add_command(
            label="Use latest calibration", command=self.reduce_pane._use_latest_calibration)
        reduction_menu.add_command(label="Scan data", command=self.reduce_pane.scan_dataset_clicked)
        reduction_menu.add_command(
            label="Review reduced data", command=self.reduce_pane.inspect_h5_clicked)
        reduction_menu.add_command(label="View log", command=self.reduce_pane.open_console_logs)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Use latest reduced data",
                                  command=self._handoff_reduced_to_analysis)
        analysis_menu.add_command(label="Inspect input", command=self.analysis_pane.inspect_input_clicked)
        analysis_menu.add_command(
            label="GSAS-II refinement round trip…",
            command=lambda: self.analysis_pane.select_page("refinement"))
        analysis_menu.add_command(
            label="Export refinement bundle…",
            command=self.analysis_pane.export_refinement_clicked)
        analysis_menu.add_command(
            label="Export GSAS-ready raw patterns…",
            command=self.analysis_pane.export_gsas_raw_clicked)
        analysis_menu.add_command(
            label="Import GSAS-II sequential results…",
            command=self.analysis_pane.import_gsas_results_clicked)
        analysis_menu.add_command(label="View log", command=self.analysis_pane.open_console_logs)

        correlation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Correlations", menu=correlation_menu)
        correlation_menu.add_command(
            label="Use latest analysis",
            command=self._handoff_analysis_to_correlations,
        )
        correlation_menu.add_command(
            label="Run correlations", command=self.correlation_pane.run_clicked)
        correlation_menu.add_command(
            label="Review results", command=self.correlation_pane.review_results)
        correlation_menu.add_command(
            label="View log", command=self.correlation_pane.open_console_logs)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Check environment", command=self._check_environment)
        tools_menu.add_command(label="Inspect detector image…", command=self._inspect_image)
        tools_menu.add_separator()
        tools_menu.add_command(label="Model development…",
                               command=self._model_development)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        self._theme_var = tk.StringVar(value=theme.active_theme())
        view_menu.add_radiobutton(
            label="Mocha (dark)", value="mocha", variable=self._theme_var,
            command=self._change_theme,
        )
        view_menu.add_radiobutton(
            label="Latte (light)", value="latte", variable=self._theme_var,
            command=self._change_theme,
        )
        # System theme is intentionally deferred until cross-platform detection
        # can be reliable without adding a GUI dependency.

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="User guide (workflow)",
            command=lambda: self._open_doc("docs/workflow.md"))
        help_menu.add_command(
            label="Run the demonstration (Ti-6Al-4V)",
            command=lambda: self._open_doc("examples/ti64_demo/README.md"))
        help_menu.add_command(
            label="Validation and limitations",
            command=lambda: self._open_doc("docs/validation.md"))
        help_menu.add_separator()
        help_menu.add_command(label="Cite SeriesXRD…", command=self._citation)
        help_menu.add_command(
            label="Report a problem…",
            command=lambda: self._open_url(f"{REPO_URL}/issues/new"))
        help_menu.add_separator()
        help_menu.add_command(label="Copy diagnostics", command=self._copy_diagnostics)
        help_menu.add_command(label="About", command=self._about)

    def _change_theme(self):
        """Apply and persist a UI-only theme without disturbing workers."""
        name = self._theme_var.get()
        theme.set_theme(name)
        try:
            from .core.uiprefs import save_prefs
            save_prefs({"theme": name})
        except OSError as exc:
            self.calib_pane.log(f"Could not save UI theme preference: {exc}",
                                "WARN")

    def _restyle_theme(self):
        """Repaint host-owned widgets; stage panes repaint via callbacks."""
        from tkinter import ttk
        from .guikit.tkstyle import apply_theme
        apply_theme(self.root, ttk)
        theme.register_widget_tree(self.root)
        theme.restyle_widgets()
        var = getattr(self, "_theme_var", None)
        if var is not None and var.get() != theme.active_theme():
            var.set(theme.active_theme())

    # ------------------------------------------------------------------

    def _select_stage(self, index: int):
        self.nb.select(max(0, min(int(index), len(STAGE_TABS) - 1)))

    def _on_calibration_accepted(self, handoff_path) -> None:
        """Pass an accepted calibration forward and reveal the next stage."""
        self.reduce_pane.set_handoff(handoff_path)
        self._select_stage(1)

    def _on_reduction_ready(self, reduced_path) -> None:
        """Pass reduced data forward and reveal the analysis stage."""
        self.analysis_pane.set_reduced(reduced_path)
        self._select_stage(2)

    def _on_analysis_ready(self, analysis_path) -> None:
        """Pass analyzed data forward and reveal the correlations stage."""
        self.correlation_pane.set_analysis(analysis_path)
        self._select_stage(3)

    def _save_all(self):
        try:
            self.calib_pane.save_config(silent=True)
            self.reduce_pane.save_config(silent=True)
            self.analysis_pane.save_config(silent=True)
            self.correlation_pane.save_config(silent=True)
            self.calib_pane.log("Saved all configs")
        except Exception as e:
            self.calib_pane.log(f"Save all failed: {e!r}", "WARN")

    def _handoff_reduced_to_analysis(self):
        """Push the reduction stage's most recent reduced .h5 into the analysis pane."""
        from tkinter import messagebox
        self.reduce_pane.pull_vars()
        reduced = str(self.reduce_pane.config.get("reduced_h5_file", "") or "").strip()
        if not reduced or not Path(reduced).is_file():
            messagebox.showinfo(
                "No reduced output",
                "No reduced .h5 is available yet. Run a reduction (or pick one on the "
                "Reduction → Review tab) first.")
            return
        self._on_reduction_ready(reduced)

    def _handoff_analysis_to_correlations(self):
        """Push the latest Analysis HDF5 into the correlations pane."""
        from tkinter import messagebox
        self.analysis_pane.pull_vars()
        analysis = str(
            self.analysis_pane.config.get("analysis_h5_file", "") or ""
        ).strip()
        if not analysis or not Path(analysis).is_file():
            messagebox.showinfo(
                "No analysis output",
                "No Analysis HDF5 is available yet. Run Analysis first, or "
                "choose an existing Analysis HDF5 on the Correlations input page.",
            )
            return
        self._on_analysis_ready(analysis)

    def _open_workspace(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askdirectory(title="Open workspace folder")
        if not path:
            return
        selected = Path(path).expanduser().resolve()
        if selected == self.workspace:
            messagebox.showinfo("Workspace", "That workspace is already open.")
            return
        if not self._confirm_shutdown_panes():
            return
        try:
            subprocess.Popen(workspace_launch_args(selected))
        except OSError as exc:
            messagebox.showerror(
                "Could not open workspace",
                f"SeriesXRD could not start a new window.\n\n{exc}",
            )
            return
        self._shutdown_panes(confirm=False)
        self.root.destroy()

    def _check_environment(self):
        from tkinter import messagebox
        from .core.env import check_dependencies
        dep = check_dependencies(sys.executable)
        lines = [f"Python checked: {getattr(dep, 'checked_in', '') or sys.executable}", ""]
        lines += [f"  {'OK ' if ok else 'MISSING'}  {pkg}" for pkg, ok in dep.required.items()]
        if dep.missing_required:
            lines += ["", "Missing: " + ", ".join(dep.missing_required)]
        else:
            lines += ["", "All required packages present."]
        messagebox.showinfo("Environment check", "\n".join(lines))

    def _inspect_image(self):
        import tkinter as tk
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Inspect detector image")
        if not path:
            return
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "seriesxrd.core.inspect", path],
                capture_output=True, text=True, timeout=60)
            out = (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:
            out = f"Inspection failed: {e!r}"
        win = tk.Toplevel(self.root)
        win.title(f"Inspect: {Path(path).name}")
        win.geometry("760x520")
        txt = tk.Text(win, bg=theme.C.BG2, fg=theme.C.FG,
                      insertbackground=theme.C.FG, relief="flat",
                      font=("TkFixedFont", 9), wrap="none")
        txt.insert("end", out)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        theme.register_widget_tree(win)

    def _open_url(self, url: str):
        try:
            webbrowser.open(url)
        except Exception:
            self._copy_to_clipboard(url, note="Could not open a browser — the "
                                               "URL is on the clipboard.")

    def _open_doc(self, repo_relative: str):
        """Open a documentation page — the local file when running from a
        source checkout, else the repository copy on GitHub."""
        local = Path(__file__).resolve().parents[1] / repo_relative
        if local.is_file():
            self._open_url(local.as_uri())
        else:
            self._open_url(f"{REPO_URL}/blob/main/{repo_relative}")

    def _copy_to_clipboard(self, text: str, note: str = "Copied to clipboard."):
        from tkinter import messagebox
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
        except Exception as exc:
            messagebox.showerror("Clipboard", f"Could not copy: {exc}")
            return
        # Non-modal confirmation: flash the workspace label instead of a popup.
        try:
            prev = self._ws_label.cget("text")
            self._ws_label.configure(text=f"  {note}")
            self._ws_label.after(2000, lambda: self._ws_label.configure(text=prev))
        except Exception:
            pass

    def _copy_workspace_path(self):
        self._copy_to_clipboard(str(self.workspace), note="Workspace path copied.")

    def _copy_diagnostics(self):
        """Copy a support-ready diagnostics block: versions, dependencies,
        platform, workspace, and the analysis file's recorded provenance."""
        from .core.provenance import provenance_report
        analysis_h5 = ""
        try:
            self.analysis_pane.pull_vars()
            analysis_h5 = str(
                self.analysis_pane.config.get("analysis_h5_file", "")
                or self.analysis_pane.config.get("reduced_h5_file", "") or "")
        except Exception:
            pass
        report = provenance_report(analysis_h5 if analysis_h5
                                   and Path(analysis_h5).is_file() else None)
        report = f"{report}\n\nWorkspace: {self.workspace}"
        self._copy_to_clipboard(report, note="Diagnostics copied.")

    def _citation(self):
        from tkinter import messagebox
        citation = (
            f"Maples, R. SeriesXRD (version {VERSION}) [Computer software]. "
            f"{REPO_URL}\n\n"
            "Machine-readable metadata: CITATION.cff in the repository "
            "(GitHub's 'Cite this repository' button uses it). A DOI is "
            "minted per release via Zenodo once the repository is public.")
        if messagebox.askyesno(
                f"Cite {TOOL_NAME}",
                f"{citation}\n\nCopy this citation to the clipboard?"):
            self._copy_to_clipboard(citation, note="Citation copied.")

    def _model_development(self):
        """Tools → Model development: GUI access to the training-side CLIs
        (corpus building, benchmarking, learned-scorer training) as command
        builders that stream their output — the workflow tabs stay focused on
        measurement analysis."""
        import tkinter as tk
        from tkinter import ttk, filedialog
        win = tk.Toplevel(self.root)
        win.title("Model development")
        win.geometry("860x640")
        win.configure(bg=theme.C.BG)

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        out_text = tk.Text(win, height=14, bg=theme.C.BG2, fg=theme.C.FG,
                           insertbackground=theme.C.FG,
                           relief="flat", state="disabled",
                           font=("TkFixedFont", 9))
        out_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        theme.register_widget_tree(win)
        state = {"proc": None}

        def _append(line: str):
            out_text.configure(state="normal")
            out_text.insert("end", line)
            out_text.see("end")
            out_text.configure(state="disabled")

        def _run(cmd: "list[str]"):
            if state["proc"] is not None:
                _append("[busy] a tool is already running — wait for it to "
                        "finish\n")
                return
            _append("\n$ " + " ".join(cmd) + "\n")
            import threading
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        bufsize=1)
            except Exception as exc:
                _append(f"[failed to start] {exc!r}\n")
                return
            state["proc"] = proc

            def _pump():
                assert proc.stdout is not None
                for line in proc.stdout:
                    win.after(0, _append, line)
                rc = proc.wait()
                def _done():
                    state["proc"] = None
                    _append(f"[exit {rc}]\n")
                win.after(0, _done)
            threading.Thread(target=_pump, daemon=True).start()

        def _row(parent, label, var, browse=None, width=48):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=18, anchor="w").pack(side="left")
            ttk.Entry(row, textvariable=var, width=width).pack(
                side="left", fill="x", expand=True)
            if browse:
                def _pick():
                    if browse == "dir":
                        p = filedialog.askdirectory(parent=win)
                    elif browse == "save":
                        p = filedialog.asksaveasfilename(parent=win)
                    else:
                        p = filedialog.askopenfilename(parent=win)
                    if p:
                        var.set(p)
                ttk.Button(row, text="…", width=3, command=_pick).pack(
                    side="left", padx=2)
            return row

        ws = str(self.workspace)

        # --- Corpus tab -------------------------------------------------
        pg = ttk.Frame(nb, padding=10)
        nb.add(pg, text="CIF corpus")
        ttk.Label(pg, style="Muted.TLabel", wraplength=760, justify="left", text=(
            "Screen a directory of CIFs (parse/dedupe/size-screen) into a "
            "training-only corpus manifest for seriesxrd-ml-train --cif-dir. "
            "Fetching from COD needs network access.")).pack(anchor="w")
        c_dir = tk.StringVar()
        _row(pg, "CIF directory", c_dir, browse="dir")
        ttk.Label(pg, style="Muted.TLabel", text=(
            "Writes corpus_manifest.json into the CIF directory.")).pack(
            anchor="w")
        ttk.Button(pg, text="Screen corpus", command=lambda: _run(
            [sys.executable, "-m", "seriesxrd.analysis.corpus", "screen",
             c_dir.get()],
        )).pack(anchor="w", pady=6)

        # --- Benchmark tab ----------------------------------------------
        pg = ttk.Frame(nb, padding=10)
        nb.add(pg, text="Benchmark")
        ttk.Label(pg, style="Muted.TLabel", wraplength=760, justify="left", text=(
            "Score a scorer against labelled XY patterns (RRUFF/opXRD-style) "
            "through the real Step-1/2 preprocessing — the gate a trained "
            "scorer must pass against the cosine baseline. See "
            "docs/ml-training.md.")).pack(anchor="w")
        b_labels = tk.StringVar(); b_scorer = tk.StringVar()
        _row(pg, "Labels file/dir", b_labels, browse="open")
        _row(pg, "Scorer (optional)", b_scorer, browse="open")
        ttk.Button(pg, text="Run benchmark", command=lambda: _run(
            [sys.executable, "-m", "seriesxrd.analysis.benchmark",
             "--labels", b_labels.get(), "--workspace", ws]
            + (["--ml-scorer", f"torch:{b_scorer.get()}"]
               if b_scorer.get() else []),
        )).pack(anchor="w", pady=6)

        # --- Training tab -----------------------------------------------
        pg = ttk.Frame(nb, padding=10)
        nb.add(pg, text="Train scorer")
        ttk.Label(pg, style="Muted.TLabel", wraplength=760, justify="left", text=(
            "Train the Step-3b learned pair scorer (requires the [ml] extra / "
            "PyTorch; heavy — a cluster or GPU workstation is the usual "
            "venue). The deterministic cosine ranker remains the default "
            "until a trained model beats it on the benchmark.")).pack(
            anchor="w")
        t_out = tk.StringVar(value=str(Path(ws) / "scorer.pt"))
        t_cif = tk.StringVar()
        _row(pg, "Model output", t_out, browse="save")
        _row(pg, "CIF corpus (optional)", t_cif, browse="dir")
        ttk.Button(pg, text="Train", command=lambda: _run(
            [sys.executable, "-m", "seriesxrd.analysis.ml_train",
             "--workspace", ws, "--out", t_out.get()]
            + (["--cif-dir", t_cif.get()] if t_cif.get() else []),
        )).pack(anchor="w", pady=6)

        def _on_close():
            proc = state.get("proc")
            if proc is not None and proc.poll() is None:
                from tkinter import messagebox
                if not messagebox.askyesno(
                        "Tool running",
                        "A model-development tool is still running. "
                        "Terminate it and close?", parent=win):
                    return
                try:
                    proc.terminate()
                except Exception:
                    pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _about(self):
        from tkinter import messagebox
        try:
            from .calib.processing import runtime_versions
            v = runtime_versions()
            ver = "\n".join(f"  {k}: {val}" for k, val in v.items())
        except Exception:
            ver = f"  seriesxrd: {VERSION}"
        messagebox.showinfo(
            f"About {TOOL_NAME}",
            f"{TOOL_NAME} {VERSION} — MIT license\n{REPO_URL}\n\n"
            f"Workspace:\n  {self.workspace}\n\nVersions:\n{ver}")

    # ------------------------------------------------------------------

    def _stage_panes(self):
        return (
            self.calib_pane,
            self.reduce_pane,
            self.analysis_pane,
            self.correlation_pane,
        )

    def _confirm_shutdown_panes(self) -> bool:
        """Ask every stage before mutating any stage's lifecycle state."""
        for pane in self._stage_panes():
            confirm = getattr(pane, "confirm_shutdown", None)
            if callable(confirm) and not confirm():
                return False
        return True

    def _shutdown_panes(self, confirm: bool = True) -> bool:
        if confirm and not self._confirm_shutdown_panes():
            return False
        for pane in self._stage_panes():
            if not pane.shutdown(confirm=False):
                return False
        return True

    def _on_quit(self):
        if self._shutdown_panes(confirm=True):
            self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="seriesxrd", description="SeriesXRD desktop application")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE),
                        help=f"Workspace folder for configs and outputs (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--theme", choices=("mocha", "latte"), default=None,
                        help="UI theme override (default: saved preference).")
    args = parser.parse_args(argv)

    from .guikit.dpi import enable_hi_dpi
    from .core.uiprefs import load_prefs
    theme.set_theme(args.theme or load_prefs().get("theme", "mocha"))
    enable_hi_dpi()
    app = SeriesXRDApp(workspace=args.workspace)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
