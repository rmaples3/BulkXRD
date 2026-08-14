"""Stacked-panel pattern figure (one pressure per panel) from an analysis HDF5.

The journal-standard way to show a compression series: touching vertical
panels, one pattern per pressure, shared 2theta/q axis, a sequential color
ramp encoding pressure. ``seriesxrd-stack`` / the Frame-meta export dialog's
"stacked figure" checkbox produce it directly from any analysis channel.

Design notes (mirrors the interactive session that prototyped it):

* **Channel** — any :func:`refine_export._pattern_source` channel
  (``fit``/``clean``/``mean``/``hybrid``/``sigmaclip``/``spots``/``auto``/
  ``robust``/``residual``). ``spots`` is the coarse-grain/single-crystal
  sample channel.
* **One panel per pressure** — with ``frames=None`` the frames are grouped
  by ``/frames/pressure`` and the best exposure of each group is chosen:
  highest zinger-proof signal-to-noise (median-filtered max over MAD), with
  saturated frames vetoed (``saturation_cutoff``; a detector-count-cutoff
  overflow blooms across a wide 2theta band and ruins the panel). An
  explicit ``frames=[...]`` keeps exactly those frames (no veto), still
  ordered by pressure.
* **Normalization** — each panel is scaled by its median-filtered maximum
  (a single-bin zinger cannot set the scale), floored at 10x the panel's
  MAD noise so a signal-free frame renders flat instead of amplified noise.
* **exclude_d** — optional list of d-spacings (Angstrom) whose windows
  (fractional half-width ``exclude_width``) are zeroed before plotting:
  the 1D analog of ``seriesxrd-spots --exclude-d`` for known contaminant
  lines (gasket W, diamond). Positions are used as given — scale them to
  pressure yourself if it matters at your compression.

Pure numpy + h5py; matplotlib is imported lazily inside :func:`stack_figure`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["mad_noise", "main", "robust_amp", "stack_figure"]


def _robust_amp(y: np.ndarray) -> float:
    """Max of a 5-bin median-filtered copy — a 1-2 bin zinger can't win."""
    from scipy.signal import medfilt
    return float(np.nanmax(medfilt(np.nan_to_num(np.asarray(y, float)), 5)))


def _mad_noise(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return 0.0
    return float(np.nanmedian(np.abs(y - np.nanmedian(y))) * 1.4826)


# Public names: the correlations waterfall shares this per-frame
# normalization so a zinger cannot flatten a trace there either.
robust_amp = _robust_amp
mad_noise = _mad_noise


# Figure-output presets for GUI/CLI exports: where the figure will live
# decides its resolution and format. "publication" prefers a vector format so
# journals can rescale without rasterization artifacts.
FIGURE_PRESETS = {
    "screen":       {"dpi": 110, "format": "png",
                     "palette": None, "background": None},
    "presentation": {"dpi": 200, "format": "png",
                     "palette": None, "background": None},
    "publication":  {"dpi": 600, "format": "pdf",
                     "palette": "latte", "background": "#ffffff"},
}

FIGURE_FORMATS = ("png", "svg", "pdf")


def stack_figure(
    analysis_h5: "str | Path",
    out_png: "str | Path",
    *,
    source: str = "fit",
    frames: "Optional[Sequence[int]]" = None,
    exclude_d: "Optional[Sequence[float]]" = None,
    exclude_width: float = 0.028,
    saturation_cutoff: "Optional[float]" = 1.0e6,
    x_min: "Optional[float]" = None,
    x_max: "Optional[float]" = None,
    title: "Optional[str]" = None,
    style: str = "panels",
    dpi: int = 300,
    palette: "str | None" = None,
    background: "str | None" = None,
) -> Dict[str, Any]:
    """Write a stacked PNG of ``source`` patterns vs pressure.

    ``style="panels"`` (default) draws touching bordered subplots, one per
    frame — the journal compression-series layout. ``style="waterfall"``
    overlays all traces on ONE axes with a constant vertical offset (labels
    at the right end of each trace) — better for eyeballing peak drift
    across many frames because nothing interrupts the vertical alignment.

    Returns a manifest ``{out_png, n_panels, frames, labels, source, axis,
    style}``. Raises ``ValueError`` when no frame qualifies (all
    excluded/saturated) or the style is unknown.
    """
    import h5py  # type: ignore

    from .refine_export import _pattern_source, _to_two_theta_deg

    src = Path(analysis_h5).expanduser()
    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        wl_raw = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        wavelength = wl_raw if wl_raw > 0 else None
        data, resolved = _pattern_source(h5, source)
        n_total = int(data.shape[0])
        radial = (np.asarray(h5["radial"][:], dtype=float) if "radial" in h5
                  else np.arange(data.shape[1], dtype=float))
        fr = h5.get("frames")
        names: List[str] = [""] * n_total
        if fr is not None and "filename" in fr:
            names = [(s.decode("utf-8", "replace")
                      if isinstance(s, (bytes, bytearray)) else str(s))
                     for s in fr["filename"][:]]
        pressure = (np.asarray(fr["pressure"][:], dtype=float)
                    if (fr is not None and "pressure" in fr
                        and fr["pressure"].shape[0] == n_total)
                    else np.full(n_total, np.nan))
        excluded = (np.asarray(fr["excluded"][:], dtype=bool)
                    if (fr is not None and "excluded" in fr
                        and fr["excluded"].shape[0] == n_total)
                    else np.zeros(n_total, dtype=bool))

    # ---- axis: 2theta when derivable, else the native radial axis ----
    tth = _to_two_theta_deg(radial, unit, wavelength)
    if tth is not None:
        x, x_label, axis = tth, (r"2$\theta$ (degree)"
                                 + (f",  $\\lambda$ = {wavelength:g} $\\mathrm{{\\AA}}$"
                                    if wavelength else "")), "2th_deg"
    else:
        x, x_label, axis = radial, unit or "radial", unit or "radial"

    # ---- optional contaminant-window zeroing (1D exclude-d) ----
    excluded_windows: List[List[float]] = []
    if exclude_d:
        from .refine_export import _zero_d_windows
        data, excluded_windows = _zero_d_windows(
            data, radial, unit, wavelength, exclude_d, exclude_width)

    # ---- panel selection ----
    if frames is not None:
        chosen = [int(i) for i in frames if 0 <= int(i) < n_total]
    else:
        groups: "Dict[Any, tuple]" = {}
        singles: List[tuple] = []
        for i in range(n_total):
            if excluded[i]:
                continue
            y = data[i]
            if (saturation_cutoff is not None
                    and float(np.nanmax(np.nan_to_num(y))) > saturation_cutoff):
                continue
            snr = _robust_amp(y) / max(_mad_noise(y), 1e-9)
            if np.isfinite(pressure[i]):
                key = round(float(pressure[i]), 3)
                if key not in groups or snr > groups[key][0]:
                    groups[key] = (snr, i)
            else:
                singles.append((snr, i))
        chosen = [i for _, i in groups.values()] + [i for _, i in singles]
    if not chosen:
        raise ValueError("No frames to plot (all excluded or saturated).")
    order = sorted(
        chosen,
        key=lambda i: (not np.isfinite(pressure[i]),
                       float(pressure[i]) if np.isfinite(pressure[i]) else 0.0,
                       i))

    labels = []
    for i in order:
        if np.isfinite(pressure[i]):
            labels.append(f"{pressure[i]:g} GPa")
        else:
            labels.append(Path(names[i]).stem or f"frame {i}")

    # ---- render ----
    from ..guikit.theme import matplotlib_palette, style_figure

    colors = matplotlib_palette(palette)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    n = len(order)
    cmap = colormaps["Blues"]
    p_known = [float(pressure[i]) for i in order if np.isfinite(pressure[i])]
    p_lo = min(p_known) if p_known else 0.0
    p_span = (max(p_known) - p_lo) if len(p_known) > 1 else 1.0

    def _color(i):
        if np.isfinite(pressure[i]):
            t = (float(pressure[i]) - p_lo) / max(p_span, 1e-9)
            return cmap(0.35 + 0.6 * t)
        return "#888888"

    def _norm(i):
        y = np.nan_to_num(np.asarray(data[i], float))
        return np.clip(y / max(_robust_amp(y), 10.0 * _mad_noise(y), 1e-9),
                       -0.06, 1.05)

    lo_x = x_min if x_min is not None else float(np.nanmin(x))
    hi_x = x_max if x_max is not None else float(np.nanmax(x))
    fig_title = title if title is not None else f"{src.stem} — {resolved}"

    if style == "waterfall":
        fig, ax = plt.subplots(figsize=(9.0, 0.62 * n + 2.0))
        off = 0.0
        for i, lbl in zip(order, labels):
            ax.plot(x, _norm(i) + off, lw=0.8, color=_color(i))
            ax.annotate(lbl, (hi_x, off + 0.08), fontsize=8.5,
                        color=colors["FG"], ha="right", fontweight="bold")
            off += 1.1
        ax.set_xlim(lo_x, hi_x)
        ax.set_yticks([])
        ax.set_ylabel("normalized intensity (offset per frame)")
        ax.set_xlabel(x_label)
        ax.set_title(fig_title, fontsize=11, pad=8)
        ax.grid(axis="x", color=colors["BORDER"], lw=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
    elif style == "panels":
        fig, axes = plt.subplots(n, 1, figsize=(7.2, 0.78 * n + 1.2),
                                 sharex=True, gridspec_kw={"hspace": 0},
                                 squeeze=False)
        axes = axes[:, 0]
        for ax, i, lbl in zip(axes[::-1], order, labels):
            ax.plot(x, _norm(i), lw=0.8, color=_color(i))
            ax.set_ylim(-0.12, 1.28)
            ax.set_yticks([])
            ax.annotate(lbl, (0.985, 0.72), xycoords="axes fraction",
                        ha="right", fontsize=9.5, fontweight="bold",
                        color=colors["FG"])
            for s in ("top", "bottom"):
                ax.spines[s].set_linewidth(0.6)
                ax.spines[s].set_color(colors["BORDER"])
        for ax, s in ((axes[0], "top"), (axes[-1], "bottom")):
            ax.spines[s].set_linewidth(0.8)
            ax.spines[s].set_color(colors["FG"])
        axes[-1].set_xlim(lo_x, hi_x)
        axes[-1].set_xlabel(x_label)
        fig.text(0.045, 0.5, "Intensity (a.u.)", va="center",
                 rotation="vertical", fontsize=11)
        axes[0].set_title(fig_title, fontsize=11, pad=8)
        fig.tight_layout(rect=(0.05, 0, 1, 1))
    else:
        raise ValueError(f"Unknown style {style!r} (panels|waterfall).")
    style_figure(fig, palette=palette, background=background)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    man = {"out_png": str(out), "n_panels": n, "frames": [int(i) for i in order],
           "labels": labels, "source": resolved, "axis": axis, "style": style,
           "excluded_windows": excluded_windows,
           "palette": palette or "active", "background": background}
    print(f"[STACK] {n} panel(s) ({resolved}, {axis}, {style}) -> {out}",
          flush=True)
    return man


def main(argv: "list[str] | None" = None) -> int:
    """CLI: ``seriesxrd-stack analysis.h5 out.png [options]``."""
    import argparse
    p = argparse.ArgumentParser(
        prog="seriesxrd-stack",
        description="Stacked-panel pattern figure (one pressure per panel) "
                    "from an analysis HDF5.")
    p.add_argument("analysis", help="Path to an *_analysis.h5.")
    p.add_argument("out_png", help="Output PNG path.")
    p.add_argument("--source", default="fit",
                   choices=["fit", "clean", "mean", "hybrid", "sigmaclip",
                            "spots", "auto", "robust", "residual"],
                   help="Pattern channel (default fit; spots = the "
                        "coarse-grain sample channel).")
    p.add_argument("--frames", default="",
                   help="Comma-separated frame indices (default: best "
                        "non-saturated exposure per pressure).")
    p.add_argument("--exclude-d", default="",
                   help="Comma-separated d-spacings (A) to zero out "
                        "(gasket/diamond lines).")
    p.add_argument("--exclude-width", type=float, default=0.028,
                   help="Fractional half-width of each excluded window "
                        "(default 0.028).")
    p.add_argument("--saturation-cutoff", type=float, default=1.0e6,
                   help="Veto auto-selected frames whose channel exceeds "
                        "this (default 1e6 counts); <=0 disables.")
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--style", default="panels",
                   choices=["panels", "waterfall"],
                   help="panels = touching subplots (journal layout); "
                        "waterfall = offset traces on one axes.")
    p.add_argument("--palette", choices=["mocha", "latte"], default=None,
                   help="Figure palette (default: active UI theme / mocha in CLI).")
    p.add_argument("--background", default=None,
                   help="Optional Matplotlib background color override.")
    args = p.parse_args(argv)
    frames = ([int(s) for s in args.frames.split(",") if s.strip()]
              if args.frames.strip() else None)
    exd = ([float(s) for s in args.exclude_d.split(",") if s.strip()]
           if args.exclude_d.strip() else None)
    cutoff = args.saturation_cutoff if args.saturation_cutoff > 0 else None
    try:
        stack_figure(args.analysis, args.out_png, source=args.source,
                     frames=frames, exclude_d=exd,
                     exclude_width=args.exclude_width,
                     saturation_cutoff=cutoff, x_min=args.x_min,
                     x_max=args.x_max, title=args.title, style=args.style,
                     palette=args.palette, background=args.background)
    except (OSError, ValueError, KeyError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
