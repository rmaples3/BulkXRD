"""Command-line entry point for headless correlation generation."""
from __future__ import annotations

import argparse


def _run(args: argparse.Namespace) -> int:
    from .processing import run_correlations

    try:
        manifest = run_correlations(
            args.analysis,
            args.out,
            sample_type=args.sample_type,
            source=args.source,
            radial_min=args.radial_min,
            radial_max=args.radial_max,
            window_width=args.window_width,
            window_step=args.window_step,
            location_tolerance=args.location_tol,
            scale_quantile=args.scale_quantile,
            make_plots=not args.no_plots,
            max_anchor_plots=args.max_anchor_plots,
            order_by=args.order_by,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] correlation generation failed: {exc}", flush=True)
        return 1
    print(
        "[CORRELATIONS] "
        f"{manifest['sample_type']}: {manifest['n_frames']} frame(s), "
        f"{manifest['n_peaks']} all-peak anchor(s), "
        f"{manifest['n_windows']} window(s) -> {manifest['correlations_h5']}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seriesxrd-correlate",
        description=(
            "Generate fixed Log-squared ROI, geometric location, waterfall, "
            "and within/across-frame window maps from a SeriesXRD Analysis HDF5."
        ),
    )
    parser.add_argument("analysis", help="Path to a SeriesXRD *_analysis.h5 file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--sample-type",
        required=True,
        choices=("powder", "single_crystal"),
        help="Peak source: /peaks or all observations in /spots/obs.",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=(
            "fit",
            "auto",
            "clean",
            "hybrid",
            "mean",
            "sigmaclip",
            "spots",
            "residual",
        ),
        help=(
            "Analysis pattern channel (default: fit for powder, spots for "
            "single crystal)."
        ),
    )
    parser.add_argument(
        "--radial-min", type=float, default=None,
        help="Optional lower bound in the Analysis HDF5 native radial unit.",
    )
    parser.add_argument(
        "--radial-max", type=float, default=None,
        help="Optional upper bound in the Analysis HDF5 native radial unit.",
    )
    parser.add_argument(
        "--window-width", type=float, default=5.0,
        help=(
            "Window width in the native radial unit; it cannot exceed the "
            "selected span (default: 5)."
        ),
    )
    parser.add_argument(
        "--window-step", type=float, default=1.0,
        help="Window step in the native radial unit (default: 1).",
    )
    parser.add_argument(
        "--location-tolerance",
        "--location-tol",
        dest="location_tol",
        type=float,
        default=0.02,
        help="Location-similarity cutoff in the native radial unit.",
    )
    parser.add_argument(
        "--scale-quantile",
        type=float,
        default=0.995,
        help="One pooled positive-intensity scale shared by all frames.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the sample-specific HDF5 and manifest only.",
    )
    parser.add_argument(
        "--order-by",
        default="frame",
        choices=("frame", "pressure", "temperature", "time"),
        help=(
            "Order retained frames by this /frames metadata axis before "
            "correlating (default: the Analysis file order)."
        ),
    )
    parser.add_argument(
        "--max-anchor-plots",
        type=int,
        default=None,
        help=(
            "Render per-anchor PNGs for at most this many valid anchors "
            "(first N in id order); every matrix stays complete in the HDF5. "
            "Default: no cap."
        ),
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    from ..core.config import make_stdio_robust

    make_stdio_robust()
    return _run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
