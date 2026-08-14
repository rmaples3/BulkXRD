# Architecture

How SeriesXRD is put together, and the design decisions that shaped it.
For the HDF5 layouts each stage reads and writes, see
[`docs/file-format.md`](file-format.md); for stage-by-stage usage, see
[`docs/workflow.md`](workflow.md).

## Stage pattern

The pipeline is one subpackage per stage — `calib`, `reduce`, `analysis`,
`correlations` —
communicating only through artifacts on disk plus a shared workspace folder.
Every stage follows the same internal convention:

| Module | Role |
|---|---|
| `processing.py` | Pure logic. No GUI, no subprocess — importable and headless, so the stage can run as a batch job. |
| `worker.py` or `batch.py` | Crash-isolated subprocess entry point. Heavy pyFAI/matplotlib/numerical work runs here, so a native-code crash is contained rather than taking down the GUI. |
| `gui.py` | Embeddable Tkinter pane. With `parent=None` it owns its root window; otherwise it embeds into the unified application. |
| `run_gui.py` | Standalone CLI entry point for the stage GUI. |
| `session.py` | Workspace config seeding and handoff reading. |

The unified application (`seriesxrd.app`) embeds all four stage panes in one
window. Stage handoffs are automatic: an accepted calibration writes
`calibration_handoff.json` (PONI + mask + QA record) that the Reduction tab
loads; a finished reduction hands its output HDF5 to Analysis; a finished
analysis hands its Analysis HDF5 to Correlations. These handoffs pass paths to
artifacts on disk, not in-memory scientific arrays.

Shared code lives in `core/` (config, environment checks, naming, detector
I/O and HDF5/NeXus stack ingestion, masks, handoff contract, provenance) and
`guikit/` (theme, ttk styling, tooltips, DPI awareness, embedded matplotlib
figures) — `core/` depends only on the stdlib and numpy.

Analysis frame drivers use `analysis.parallel.process_map_or_serial`: they
attempt ordered process-pool execution first, but automatically repeat the
same payload sequence serially when the operating system cannot create the
pool's semaphore resources. Results are collected before they are handed to
the caller, so fallback cannot mix partial parallel and serial outputs. A
genuine calculation exception is raised again by the serial run.

### UI theming

`guikit/theme.py` is the single source of semantic UI colors. Its mutable
palette object changes in place, then weakly registered pane callbacks repaint
ttk styles, persistent raw-Tk widgets, and any live embedded Matplotlib
figures. A theme change is therefore a presentation-only operation: panes,
scientific state, and crash-isolated workers remain alive. The selected theme
is stored in the platform's per-user configuration directory by
`core/uiprefs.py`, never in a workspace or scientific provenance record.

Interactive figures follow the UI palette. Publication exports deliberately
select the light palette and a white background, so their appearance is
stable regardless of the screen theme. Plot data colors are preserved during
restyling; only surfaces, structural text, ticks, spines, legends, and grids
change.

## Analysis pipeline

```
Step 1  background.py   diamond-spot removal + SNIP baseline
Step 2  peaks.py        pseudo-Voigt peak/profile fitting
Step 3a identify.py     pressure-aware EOS phase matching
        residual.py     evidence-gated subtraction + residual re-fit
Step 3b ml_rank.py      ML proposes, physics verifies (candidate ranking)
Step 3c unknowns.py     residual peaks → coherent tracks → unknown clusters
        + heatmap.py maps, fractions.py, microstructure.py, spots.py,
          refine_export.py exports, refine_import.py refinement round trip
```

**Step 1 — background separation.** The reduce stage writes both an azimuthal
mean and a spot-suppressed ("robust") channel per frame. Diamond
single-crystal spots average into the mean but not into the robust channel,
so `spot_residual = mean − robust` isolates them. A SNIP baseline (with
log-log-sqrt dynamic-range compression) estimated on the robust channel gives
`clean = robust − baseline`, and the integrated positive spot residual per
frame is the contamination score that flags diamond-dominated frames. Step 1
also diagnoses *where the Bragg signal lives*: on a coarse-grained /
near-single-crystal sample the azimuthal median rejects the sample itself,
and the recorded diagnosis lets Step 2's automatic source selection fall back
to the mean channel on data-driven grounds.

**Step 2 — peak fitting.** Pseudo-Voigt profiles (`A·(η·L + (1−η)·G)`),
detected with a MAD noise floor and SNR threshold, fitted jointly when
windows overlap, with rejection flags rather than silent drops. The fit
*source* is selectable — the conservative spot-suppressed `clean`, the
`mean`, a `hybrid` (clean + winsorized spot residual), or the reduce-side
sigma-clipped trimmed mean — because the azimuthal median that suppresses
diamond spots also drops real peaks on spotty or textured rings. Good peak
centers from frame *k* seed detection in frame *k+1*, so a reflection keeps
its identity as the lattice compresses.

**Step 3a — deterministic identification.** For each candidate phase, the
Birch–Murnaghan EOS (or a signed axial expansivity for axes that expand
under pressure) predicts d-spacings as a function of pressure; matching is
one-to-one, weighted by each observed peak's fitted center uncertainty, and
softly intensity-aware. The per-frame pressure metadata (`/frames/pressure`)
confines each phase's fit to that frame's pressure ± a window, which stops a
wrong phase sliding along pressure until a few lines coincide. Confidence is
conservative: F1(recall, precision) × an evidence gate × a Gaussian
pressure-prior penalty. Phases that clear the evidence gate are subtracted
(pseudo-Voigt reconstruction) and the residual is re-fitted with the full
Step-2 pipeline to surface weaker features.

**Step 3b — ML proposes, physics verifies.** A scorer ranks the whole phase
library against each frame's measured pattern (residual by default),
simulating every candidate at that frame's pressure with the same anisotropic
d-spacing model as Step 3a. The union of per-frame top-K candidates feeds
`run_identification` as its candidate set, so the deterministic matcher only
*verifies* a shortlist. The similarity function sits behind a scorer seam:
the default is a pure-numpy cosine scorer; a trained TorchScript pair scorer
can be swapped in (`seriesxrd[ml]`), but it must beat the cosine baseline on
the known-truth benchmark harness before being trusted
(see [`docs/ml-training.md`](ml-training.md)).

**Step 3c — unknowns.** Peaks that no known phase explains are linked into
gap-tolerant tracks across the series (they drift coherently if they belong
to a real phase), and tracks that appear/disappear/drift together are
clustered by co-occurrence into candidate unknown phases, each with a
d-spacing fingerprint and candidate transition frames.

## Correlation pipeline

`seriesxrd.correlations` is a separate fourth stage that consumes the public
Analysis-HDF5 contract. It does not import or execute external,
experiment-specific research scripts. The formal stage is headless at its core
and is exposed through both `seriesxrd-correlate` and an embeddable Tk pane.

The numerical sequence is deliberately fixed:

```text
Analysis background-corrected source
        -> one pooled scale and noise-derived epsilon
        -> positive clipping + bounded Log² for ROI-area maps
        -> signed residual clipping to [-1, 1] + bounded Log² for windows
        -> all-observation ROI-area maps (same-frame cells omitted)
        -> independent geometric location maps
        -> original-positive waterfall + Log²-derived shading
        -> direct and standardized positive-lag FFT-ACF window correlations
        -> ROI-gated peak tracks (analysis.unknowns.link_tracks + the mutual
           ROI similarity as its evidence gate) + exploratory transition
           screening
```

`correlations/tracks.py` supplies the linking/screening layer,
`correlations/export.py` the CSV exports, and `correlations/checkpoint.py`
the stop flag and staging-tree recovery; all are headless. Figure building
is separate from saving: `plots.build_figure` returns a canvas-less
`Figure` that the exporter saves through Agg and the GUI embeds through
TkAgg, so one code path serves files and the interactive canvas, and the
module stays importable where there is no display. The stage never
grows a second track linker — the one in `analysis/unknowns.py` gained an
optional similarity gate instead, inert for Step 3c.

Powder anchors are the retained `/peaks` rows. Single-crystal anchors are the
`/spots/obs` rows. Track identifiers may be recorded but never collapse the
all-peak/all-observation calculation. Single-crystal mode selects `spots` by
default in the GUI and CLI; its ROI feature is a one-dimensional radial
approximation, not the prototype's raw-pixel ROI. Location similarity is not
Log²-derived: it depends only on peak centers and a native-axis tolerance.
Peak cells belonging to the anchor frame are structural `NaN` values and are
blank in review plots.

FFT autocorrelation fingerprints are standardized and contain positive lags
only; lag zero is excluded. Window width and step use the HDF5 native radial
unit, and width cannot exceed the selected radial span. The supported MVP does
not migrate the prototype's `shift_tolerant_secondary` diagnostic or its
same-scan aggregation: the public Analysis HDF5 has no stable `scan_id` on
which to base that aggregation.

`correlations_powder.h5` and `correlations_single_crystal.h5` are the numeric
sources of truth. Their manifests are `manifest_powder.json` and
`manifest_single_crystal.json`; both pairs can coexist in one result folder.
PNG heatmaps remain replaceable review artifacts under
`heatmaps/<sample_type>`. HDF5 and manifest writers use temporary files
followed by `os.replace`, matching the pipeline's atomic-output policy.
The GUI indexes those PNGs into a searchable sample-type → diagram-type →
pressure hierarchy. Across-frame window maps are categorized as **All
pressures**, while within-frame maps inherit their frame pressure. Window-map
PNGs mask the diagonal and one mirrored half of each symmetric matrix; this
review mask never changes the complete square arrays stored in HDF5.

### Simulation physics conventions (Step 3b)

- **Peak widths are constant in q, not d.** Instrument resolution is
  approximately constant in q, so per-peak `Δd = d²·Δq/2π`; the resolution
  curve `FWHM_q²(q)` is fitted from the Step-2 peaks when available.
- **One pressure per simulated mixture.** All phases of a training mixture
  share a single pressure, as in a real DAC frame — independent per-phase
  pressures would teach a scorer an unphysical manifold.
- **EOS validity ceilings.** A stability-limited library entry carries
  `p_max`; identification caps its pressure search there and every
  simulator clamps to it, so a phase is never fit or trained beyond its
  transition.

## Key design decisions

- **Fit in q, not 2θ.** Peak width is roughly constant in q, so window
  sizing is uniform across the pattern, and q needs no wavelength for
  d-conversion. 2θ remains selectable; downstream handles both.
- **Robust integration is a quantile band, not a pure median.** The
  spot-suppressed channel is the mean of a narrow azimuthal quantile band
  around the median. A pure median of integer photon counts is quantized,
  which renders low-count patterns as staircases; the band mean is
  continuous-valued with the same spot-rejection behavior.
- **The robust channel is the baseline reference, not the forced fit
  source.** Suppressing diamond spots must not silently eat real sample
  peaks — hence Step 2's selectable source and Step 1's signal diagnosis.
- **Conservative SNIP window.** An over-aggressive background window erodes
  real broad peaks — true information loss. The baseline is stored, so the
  original data is always recoverable.
- **Atomic HDF5 writes.** Every writer produces a `.tmp` file and
  `os.replace`s it into place, so a crash cannot leave a partially-written
  archival file. One deliberate exception: the live watch-mode file trades
  this for append speed, because the archival file comes from a normal full
  reduction after the run ends.
- **scipy least-squares, no GPU/JAX dependency.** A vectorized numpy model
  handles thousands of frames in seconds; variable per-frame peak counts fit
  poorly into fixed-shape batching, and the interface stays clean enough to
  add an accelerated backend later if it ever becomes the bottleneck.
- **Calibration handoff is an artifact, not shared state.** The accepted
  PONI (with geometry overrides applied) travels through
  `calibration_handoff.json`; stages never reach into each other's memory.
- **Provenance in every artifact.** Each HDF5 file and JSON manifest records
  the SeriesXRD version, schema version, creation time, effective
  configuration, dependency versions, and input-file fingerprints
  (`core/provenance.py`, `/provenance` group).
