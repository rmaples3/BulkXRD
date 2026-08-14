"""Standalone launcher for the native SeriesXRD correlation GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import print_status
from .session import CONFIG_FILENAME


def _auto_find_config() -> "Path | None":
    candidate = Path.cwd() / CONFIG_FILENAME
    return candidate.resolve() if candidate.is_file() else None


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the SeriesXRD Correlations GUI",
    )
    parser.add_argument(
        "--config", default="",
        help=f"Path to {CONFIG_FILENAME} (auto-found in the current folder if omitted)",
    )
    parser.add_argument(
        "--theme", choices=("mocha", "latte"), default=None,
        help="UI theme override (default: saved preference).",
    )
    args = parser.parse_args(argv)

    from ..core.uiprefs import load_prefs
    from ..guikit import theme

    theme.set_theme(args.theme or load_prefs().get("theme", "mocha"))
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Session config not found: {config_path}")
    else:
        config_path = _auto_find_config()
        if config_path is None:
            print(
                f"[ERROR] Could not find {CONFIG_FILENAME} in {Path.cwd()}.",
                flush=True,
            )
            print(
                "Run SeriesXRD once to seed a workspace, or pass --config <path>.",
                flush=True,
            )
            return 1

    from .gui import run_app

    print_status(f"Correlation GUI started with config {config_path}")
    return int(run_app(config_path))


if __name__ == "__main__":
    raise SystemExit(main())
