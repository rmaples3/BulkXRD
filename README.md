# SeriesXRD

[![PyPI](https://img.shields.io/pypi/v/seriesxrd.svg)](https://pypi.org/project/seriesxrd/)
[![Python versions](https://img.shields.io/pypi/pyversions/seriesxrd.svg)](https://pypi.org/project/seriesxrd/)
[![CI](https://github.com/RushMaples/SeriesXRD/actions/workflows/ci.yml/badge.svg)](https://github.com/RushMaples/SeriesXRD/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21479736.svg)](https://doi.org/10.5281/zenodo.21479736)
[![License: MIT](https://img.shields.io/pypi/l/seriesxrd.svg)](LICENSE)

GUI-driven workflow for diffraction series: detector calibration review,
dataset reduction, pattern analysis, and correlation mapping. Facility-neutral
by design — it works the same way for a synchrotron beamline or a lab
(in-house) diffractometer, any calibrant, any detector pyFAI supports, and any
beamline-specific frame naming or metadata convention (see "Site adoption" in
[`docs/roadmap.md`](docs/roadmap.md) for exactly what a new site needs to
supply). A single unified desktop application (`seriesxrd`) hosts all pipeline
stages in one window. Heavy pyFAI work runs in separate `worker.py`
subprocesses, so a crash in pyFAI or matplotlib is contained in the worker
process rather than in the GUI.

![SeriesXRD Analysis pattern review showing an integrated diffraction pattern, a two-dimensional cake, and the frame-series contamination trend](docs/images/seriesxrd-analysis-pattern-review.png)

*Pattern review in the unified SeriesXRD application, using the included
Ti-6Al-4V demonstration workflow.*

## Pipeline

The workflow is one subpackage per stage, communicating only through artifacts
on disk plus a shared workspace folder:

1. **`seriesxrd.calib`** — calibration review: standard image →
   accepted `.poni` + mask + QA record, ending in a
   `calibration_handoff.json` (internal artifact passed automatically to
   the Reduction tab — not a user-facing step).
2. **`seriesxrd.reduce`** — dataset reduction: apply the accepted
   geometry/mask to a sample dataset, parallel batch azimuthal integration →
   1D patterns (mean, azimuthal-quantile-band "robust", and optional
   sigma-clipped trimmed mean) and optional 2D cakes in one HDF5 file + JSON
   manifest. Frame sources include plain images and HDF5/NeXus stack
   containers (Eiger-style master files), with per-frame metadata (timestamp,
   stage position, temperature) harvested automatically. `seriesxrd-watch`
   adds a live mode that reduces and periodically re-analyzes a dataset
   folder while frames are still being collected.
3. **`seriesxrd.analysis`** — pattern analysis: SNIP background + diamond-spot
   separation (Step 1), pseudo-Voigt peak fitting (Step 2), pressure-aware
   EOS phase identification + residual removal (Step 3a), the ML
   candidate-ranking seam (Step 3b: deterministic cosine ranker by default,
   optional learned scorer — see `docs/ml-training.md`), and unknown-phase
   clustering of the leftover residual (Step 3c). Semi-quantitative phase
   fractions, azimuthal texture metrics, and a Rietveld hand-off export round
   out the tooling.
4. **`seriesxrd.correlations`** — Log²-only correlation mapping from an
   Analysis HDF5: all-peak ROI-area and peak-location maps, original-positive
   waterfall traces shaded by the positive-Log² ROI result, and
   window-to-window maps from signed-residual Log². Both transforms share one
   pooled scale and epsilon. Powder observations come from `/peaks`;
   single-crystal observations come from `/spots/obs` and use a documented 1D
   radial ROI approximation. Numerical results are written to
   `correlations_powder.h5` or `correlations_single_crystal.h5`, with matching
   `manifest_powder.json` or `manifest_single_crystal.json` files and review
   PNGs kept in the workspace. The Results browser is searchable and groups
   images by sample type, diagram type, and pressure. Across-frame window maps
   appear under **All pressures**; within-frame maps use that frame's pressure.
   Square window-map PNGs show only the strict lower triangle and hide the
   known-one diagonal, while the HDF5 keeps each complete numeric matrix.

The calib→reduce handoff JSON is an internal artifact written to the workspace
and automatically loaded by the Reduction tab — users do not need to manage it
manually.

## Repository layout

```
├── seriesxrd/             The installable package
│   ├── core/            Shared by all stages (stdlib/numpy only)
│   │   ├── config.py        SessionConfig, JSON/hash/file helpers
│   │   ├── env.py           dependency / conda environment checks
│   │   ├── naming.py        output folder/file naming conventions
│   │   ├── io.py            detector image readers (fabio/tifffile/PIL) and
│   │   │                    HDF5/NeXus frame-stack ingestion (Eiger master files)
│   │   ├── masks.py         automatic + polygon detector masks
│   │   ├── handoff.py       the calib→reduce handoff contract (load/validate)
│   │   └── inspect.py       detector-image diagnostic CLI (seriesxrd-inspect)
│   ├── guikit/          Shared GUI/plot theming
│   │   ├── theme.py         dark Catppuccin palette (Tk + matplotlib)
│   │   ├── tkstyle.py       shared ttk style (apply_dark_theme)
│   │   └── dpi.py           HiDPI / Windows DPI-awareness helpers
│   ├── calib/           Calibration review stage
│   │   ├── processing.py    pyFAI integration + QA figure generation
│   │   ├── worker.py        crash-isolated worker subprocess
│   │   ├── gui.py           tabbed Tkinter GUI (embeddable pane)
│   │   ├── dioptas.py       optional Dioptas hand-off
│   │   └── run_gui.py       CLI entry point (seriesxrd-calib-gui)
│   ├── reduce/          Batch reduction stage
│   │   ├── processing.py    batch azimuthal integration logic
│   │   ├── worker.py        crash-isolated worker subprocess
│   │   ├── session.py       workspace config seeding (seed_reduction_config)
│   │   ├── review.py        read-only HDF5 checkpoint review
│   │   ├── straighten.py    cake-waviness diagnosis + straightened-1D rescue channel
│   │   ├── texture.py       azimuthal texture metrics per saved cake (seriesxrd-texture)
│   │   ├── watch.py         live (during-beamtime) reduction + rolling analysis (seriesxrd-watch)
│   │   ├── gui.py           tabbed Tkinter GUI (embeddable pane)
│   │   └── run_gui.py       CLI entry point (seriesxrd-reduce-gui)
│   ├── app.py           unified application (seriesxrd entry point)
│   ├── analysis/        analysis stage (background, peaks, identification,
│   │                    maps, ML ranking/training, and exports)
│   └── correlations/    Log² correlations, heatmaps, CLI, and native GUI pane
├── tests/               automated pytest suite
├── examples/            calibration_session_config.example.json (schema reference),
│                        fetch_benchmark_example.sh (downloads a real-data
│                        seriesxrd-benchmark example set — see docs/ml-training.md)
├── environment.yml      conda environment (recommended install route)
└── pyproject.toml       package metadata + pip dependencies
```

Stage convention: pure logic modules + a crash-isolated worker/batch entry
point + an embeddable `gui.py` pane. Logic stays importable and headless so
stages can also run as batch jobs without any GUI.

## Installation

Recommended for a source checkout (pyFAI installs most reliably from conda-forge):

```bash
conda env create -f environment.yml
conda activate seriesxrd
```

Install SeriesXRD from PyPI:

```bash
python -m pip install seriesxrd
python -m pip install "seriesxrd[io,stacks,phases]"  # optional readers and crystallography
```

For development from a source checkout:

```bash
python -m pip install -e ".[dev]"
```

`tkinter` must be available in your Python (it ships with python.org and
conda-forge Python; some Linux distros need `python3-tk`).

## Usage

### Unified application (primary)

```bash
seriesxrd --workspace <dir>
# Windows/macOS GUI entry point (no console window):
seriesxrd-gui --workspace <dir>
# or without installing:
python -m seriesxrd.app --workspace <dir>
```

Opens one window with **1 Calibration**, **2 Reduction**, **3 Analysis**, and
**4 Correlations** tabs. Accepting a calibration hands its PONI + mask to the
Reduction tab automatically; a finished reduction hands its output HDF5 to
the Analysis tab, and a finished analysis hands its Analysis HDF5 to
Correlations. The workspace folder holds the stage configs and all outputs.
On first launch the configs are auto-created with sensible defaults.

The GUI embeds all four stage panes in one process; scientific runs execute in
subprocesses, so a worker crash is isolated from the host window and the other
stages.

### End-to-end demonstration data

The [Ti-6Al-4V demo](examples/ti64_demo/README.md) downloads a compact,
checksum-pinned set of real synchrotron detector frames and guides it through
the unified GUI. The third-party CBF data remain in a gitignored local
workspace and are not bundled with SeriesXRD.

### Per-stage standalone GUIs

Each stage also has a standalone entry point for advanced use:

```bash
seriesxrd-calib-gui    --config <path/to/calibration_session_config.json>
seriesxrd-reduce-gui   --config <path/to/reduction_session_config.json>
seriesxrd-analysis-gui --config <path/to/analysis_session_config.json>   # optional; auto-found if omitted
seriesxrd-correlations-gui --config <path/to/correlation_session_config.json>
```

### Detector-image diagnostic

```bash
seriesxrd-inspect <image_file>
# or:
python -m seriesxrd.core.inspect <image_file>
```

### Headless analysis, correlations, and ML training

```bash
seriesxrd-analyze reduced.h5 --phases Au,Re          # Steps 1-3a, no GUI
seriesxrd-analyze reduced.h5 --ml-rank               # candidate-free: rank whole library
seriesxrd-correlate analysis.h5 --out results/correlations --sample-type powder
seriesxrd-correlate analysis.h5 --out results/correlations --sample-type single_crystal --source spots
seriesxrd-ml-train --workspace <dir> --out scorer.pt # train the learned scorer
```

The GUI and CLI select/recommend `spots` for single-crystal runs. Powder and
single-crystal numerical outputs can safely share one result directory;
neither sample type overwrites the other. Before a single-crystal correlation
run, add `/spots/obs` with
`seriesxrd-spots REDUCED.h5 --analysis ANALYSIS.h5`.

Training the Step-3b learned scorer (data collection, environment setup,
corpus building, validation gates, deployment) is documented in
[`docs/ml-training.md`](docs/ml-training.md).

### All console scripts

| Command | Purpose |
|---|---|
| `seriesxrd` | Unified GUI: Calibration + Reduction + Analysis + Correlations in one window. |
| `seriesxrd-gui` | Unified GUI without a console window on supported desktop platforms. |
| `seriesxrd-calib-gui` | Calibration stage standalone GUI. |
| `seriesxrd-reduce-gui` | Reduction stage standalone GUI (includes live watch-mode controls). |
| `seriesxrd-analysis-gui` | Analysis stage standalone GUI. |
| `seriesxrd-correlations-gui` | Correlations stage standalone GUI. |
| `seriesxrd-analyze` | Headless analysis CLI (Steps 1-3, ML ranking, exports). |
| `seriesxrd-correlate` | Headless fixed-Log² correlation mapping from an Analysis HDF5. |
| `seriesxrd-watch` | Live reduction + rolling analysis while frames are still being collected. |
| `seriesxrd-ml-train` | Train the Step-3b learned candidate scorer. |
| `seriesxrd-benchmark` | Score a scorer against labelled XY patterns (RRUFF/opXRD-style known-truth harness). |
| `seriesxrd-corpus` | Fetch/screen a training-only CIF corpus for `seriesxrd-ml-train --cif-dir`. |
| `seriesxrd-texture` | Azimuthal texture metrics (`/texture`) from a cakes-enabled reduction. |
| `seriesxrd-export-refinement` | Rietveld hand-off bundle (`.xy` patterns + phase CIFs + GSAS-II instprm) from an analysis HDF5. |
| `seriesxrd-export-gsas` | GSAS-ready raw patterns grouped by pressure. |
| `seriesxrd-import-gsas` | Map GSAS-II sequential weight fractions, uncertainties, cells, and fit quality back to SeriesXRD frames under `/refinement`. |
| `seriesxrd-stack` | Publication-style stacked or waterfall pattern plots. |
| `seriesxrd-inspect` | Detector-image diagnostic: true format from magic bytes, per-reader interpretation, intensity statistics, and a verdict. |

See [`docs/workflow.md`](docs/workflow.md) for how each of these fits into
the end-to-end pipeline and [`docs/ml-training.md`](docs/ml-training.md) for
the ML-specific ones.

## Tests

Install the development dependencies and run the test suite with pytest:

```bash
python -m pip install -e .[dev]
python -m pytest
```

## Documentation

- [`docs/workflow.md`](docs/workflow.md) — end-to-end analysis workflow.
- [`docs/architecture.md`](docs/architecture.md) — how the stages fit
  together and the design decisions behind the pipeline.
- [`docs/file-format.md`](docs/file-format.md) — the HDF5 layouts and JSON
  manifests each stage reads and writes.
- [`docs/validation.md`](docs/validation.md) — what is validated against
  what, expected tolerances, and the limits of each output.
- [`docs/phase-sources.md`](docs/phase-sources.md) — bibliography and
  provenance for every value in the bundled phase library.
- [`docs/ml-training.md`](docs/ml-training.md) — training, validating, and
  deploying the Step-3b learned scorer (cluster-agnostic — works on any
  cluster or workstation).
- [`docs/roadmap.md`](docs/roadmap.md) — implemented vs. planned features,
  and what a new facility needs to provide to adopt SeriesXRD.
- [`docs/test-data.md`](docs/test-data.md) — open datasets you can download to
  exercise each stage (calibration frames, measured patterns, CIFs, simulated
  patterns) and which command each one feeds.
- [`examples/ti64_demo/README.md`](examples/ti64_demo/README.md) — prepared,
  end-to-end Ti-6Al-4V calibration and exposure-series demonstration.
- [`docs/releasing.md`](docs/releasing.md) — build, TestPyPI, and release
  verification checklist.

## Citation

If SeriesXRD contributes to published research, please cite the exact version
used:

> Maples, R. (2026). *SeriesXRD* (Version 0.3.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.21511143

Machine-readable citation metadata is provided in
[`CITATION.cff`](CITATION.cff). GitHub can use this file to generate a
formatted software citation. The DOI badge at the top of this page links to
the permanent record for all SeriesXRD versions.

Project roles and acknowledgments are listed in [`CREDITS.md`](CREDITS.md).

Feature status and planned work are maintained in
[`docs/roadmap.md`](docs/roadmap.md).

## Contributing and support

See [`CONTRIBUTING.md`](CONTRIBUTING.md) to set up a development environment
and submit changes, and [`GOVERNANCE.md`](GOVERNANCE.md) for how decisions
and releases are made. Report defects through the GitHub issue tracker;
report security concerns according to [`SECURITY.md`](SECURITY.md).
Community participation is governed by
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
