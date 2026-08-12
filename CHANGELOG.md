# Changelog

All notable changes to SeriesXRD will be documented here. The project follows
semantic versioning once a stable public API is declared.

## [Unreleased]

### Added

- **Correlations, a fourth pipeline stage** (`seriesxrd.correlations`, merged
  from the Log²-only MVP and hardened here): reads the Analysis HDF5 and
  writes a sample-specific `correlations_<sample>.h5` + manifest with a fixed
  bounded Log² transform (one pooled scale and noise-derived epsilon),
  directional ROI-area and geometric location anchor maps, correlation-shaded
  waterfalls, and direct/FFT-ACF window correlations across and within
  frames. Exposed as the unified app's fourth tab, `seriesxrd-correlate`, and
  `seriesxrd-correlations-gui`; analysis hands a completed HDF5 off
  automatically.
- **ROI-gated peak tracks + exploratory transition screening**: the
  correlation stage now links its all-peak table across the series by reusing
  the one Step-3c linker — `analysis.unknowns.link_tracks` gained an optional
  evidence gate (`similarity` callable + `min_similarity`, inert by default
  and bit-identical for Step 3c) fed with the mutual ROI similarity
  `sqrt(S(A→B)·S(B→A))`. Artifacts grow `/tracks` (observations, summaries,
  similarity-scored edges, per-interval births/deaths/window-drop screening
  with the decision rule recorded), a per-group track-overview PNG, and CLI
  knobs (`--no-tracks`, `--track-min-similarity`, `--track-min-frames`,
  `--track-link-tol-fwhm`, `--track-max-gap`, `--track-group-by`). Flagged
  intervals are labeled exploratory candidates, never confirmed transitions.
- **Frame ordering by metadata** (`--order-by frame|pressure|temperature|time`):
  retained frames are stably sorted through the new public
  `analysis.series` helpers (promoted from unknowns; Step-2 seed propagation
  now imports them there too), so waterfall rows, window frame axes, and peak
  tracks all follow the physical series; `/frames/order_value` and
  `order_by`/`order_label` attrs record it.
- **CSV export** (`--export-csv`, `--export-matrices`, and an Export CSV…
  button): per-anchor top-N ROI matches, long-format across/within window
  tables, track summaries/observations, transition intervals, and on-demand
  full K×K matrix dumps, written through `core.io.write_table_csv` and listed
  in the manifest.
- **Anchor validity** (`/peaks/valid` + manifest counts): anchors whose ROI
  support crosses the radial boundary or contains a masked bin are flagged,
  skipped in per-anchor plots, and reported instead of silently producing
  blank maps.
- **Progress protocol + GUI progress bar**: the stage streams
  `[CORRELATIONS] <done> <total>` during scoring and rendering; the GUI shows
  a live bar, reads the run manifest on success (recording
  `correlations_h5` into the session config), and summarizes frames/anchors/
  windows/tracks/PNGs/elapsed. `--max-anchor-plots` caps per-anchor PNGs.

### Fixed

- **Streaming progress for Steps 1/2/3a restored**: the shared
  `process_map_or_serial` helper materialized `list(executor.map(...))`, so
  the `[ANALYSIS]`/`[PEAKS]`/`[IDENTIFY]` progress lines all fired after the
  work finished and the GUI bar jumped 0→100. It now yields lazily in payload
  order, and its serial fallback catches only pool-construction/submission
  `OSError` — worker errors propagate immediately instead of being mislabeled
  "process pool unavailable" and recomputed serially before failing.
- **One zero-signal convention for ROI scores**: two all-zero single-crystal
  ROIs scored a perfect 1.0 (a masked/empty region read as maximal
  similarity) and `integrated_iou` returned NaN for a finite zero
  denominator; both now score 0, with NaN reserved for structural
  invalidity.
- **Masked bins are structural in the ROI path**: a non-finite native bin
  inside an anchor support (or the support overlap, for the target) voids the
  comparison instead of being bridged by interpolation, matching the window
  rule the module already documented.
- **Correlations GUI hardening**: the queue poll loop survives a failing
  handler (previously one `TclError` killed it silently and the GUI never
  learned the run finished); the worker launch honors the configured
  `python_exe`/`backend_dir` with `cwd` at the checkout root so it works from
  an uninstalled source tree; settings are locked during a run; failed runs
  persist a log tail under `logs_root`; the Results browser skips redundant
  tree rebuilds, caps rendered leaves, caches the decoded preview image, and
  gains Open-folder; session seeding adds the `backend_dir`/`logs_root` keys
  every other stage seeds.
- **Waterfall figures bounded and zinger-proof**: per-anchor waterfall height
  is capped at 18 in (a 200-frame series previously rendered 15 000-px-tall
  PNGs) and traces share `analysis.stackplot`'s robust normalization instead
  of a 99th-percentile scale a single hot bin could flatten.
- Unique staging-file names for the correlation HDF5 so concurrent
  same-sample runs cannot clobber each other's temp file.

### Changed

- **The K×K ROI matrix is vectorized**: one closed-form pass per anchor over
  all targets replaces K² Python kernel calls (measured 1.6 s where the
  scalar loop extrapolates to ~12 min at K=900), pinned cell-for-cell against
  the unchanged scalar reference. Pairwise Pearson uses one Gram matrix, the
  window ACFs one batched FFT, and window resampling one weight vector per
  window — each with a loop-reference equality test.

## [0.4.0] - 2026-07-29

### Fixed

- The peak-fit quality gate no longer rejects a peak for being well measured.
  `_fit_group` weights every point by one scalar sigma — the MAD noise floor of
  the background — so a reduced chi-square measures the misfit in units of
  background noise, and for a profile mismatch that is a fixed *fraction* of
  the peak it grows as (height/noise)². `max_chi2` on its own was therefore an
  SNR gate. On a 1288-frame diamond-anvil series the log-log slope of
  chi-square against SNR is 2.12 (r = 0.87) above SNR 200, rejection runs 8% /
  61% / 99.9% / 100% across the SNR 100-300 / 300-1000 / 1000-3000 / 3000+
  bands, and the strongest reflection of the frame was rejected in 1055 of 1288
  frames while fitting to about 1% of its own height. Raising `max_chi2` does
  not help: the median chi-square of those rejected peaks is 1015.

  Fit quality is now judged on two tests, **per peak**, and a peak must fail
  both to be flagged `FLAG_BAD_CHI2`. They model two error regimes: a weak peak
  is limited by random noise, so its residual matters in units of the noise
  floor (`max_chi2` on the new `/peaks/chi2_local`); a bright peak exposes
  systematic profile mismatch, so what matters is the residual as a fraction of
  its own height (the new `max_rel_misfit`, default 0.05, on
  `/peaks/rel_misfit`). Both are measured over that peak's own span, using the
  full joint model so a neighbour's intensity is accounted for rather than
  counted as misfit.

  Group adequacy and individual reliability are reported as the separate claims
  they are. `/peaks/chi2` still carries the whole joint fit's reduced
  chi-square, but it no longer decides anything: one verdict shared across a
  group is only ever more lenient, because the tallest peak sets the
  denominator. On that series 121 peaks were excused by a bright neighbour while
  being bad on their own, and none were condemned by one.

  The span a peak is judged over is fixed at ±2 FWHM
  (`peaks.QUALITY_WINDOW_FWHM`) rather than following `window_factor`, so a
  calibrated threshold keeps its meaning when the fit window is tuned.

  The threshold is calibrated rather than derived — it depends on sampling,
  overlap and background treatment. On 1402 peaks from 144 real frames, a fit
  with no hard failure misfits by 1.6–2.0% (median) at every SNR band above 30,
  nearly flat in brightness, and the rejection curve knees near 5%. Measured
  effect, with detection unchanged: good peaks 66.3% → 97.9%, and the strongest
  reflection of the frame survives in 97.9% of frames instead of 16.0%.

  Each peak is scored only on the points it owns — inside its span and where its
  own profile is the group's tallest — so a badly modelled component cannot leak
  into the peak beside it.

  Both measures are dimensionless and were verified invariant to detector gain
  (×1000) and to a flat pedestal exactly, and to within 1% under a doubling of
  bins per FWHM, so neither has to be re-derived per detector. The threshold is
  not universal: on opXRD, 2188 peaks from labelled patterns taken on many
  different instruments, the same measure is larger and falls with brightness
  instead of staying flat (median 8.1% below SNR 10, 0.9% at SNR 1000–3000),
  because laboratory profiles are broader and more overlapped than a synchrotron
  DAC ring. The two-clause structure protects those weak peaks — one with a 10%
  relative misfit is still governed by chi-square — and on that corpus the
  threshold barely matters (rejection moves 9.7% → 5.9% across 2–10%).
  `docs/workflow.md` says what to re-calibrate, when, and what a strongly
  asymmetric instrument profile does to it.

  Hard failures — no convergence, width or centre pinned at a bound — are
  unchanged and still reject regardless of residual size.

  **This changes results.** Steps 3a, 3c, the spot tracker and the benchmark
  harness all consume unflagged peaks, so an existing analysis re-run on this
  version will attribute more reflections, subtract more of them, and — because
  `spots.detect_spots` excludes blobs near attributed powder peaks — may return
  a different spot list. Set `max_rel_misfit=0` to restore the old behaviour
  exactly.

### Added

- `max_rel_misfit` on `fit_pattern`, `fit_dataset`, `run_peak_fitting` and
  `run_residual`, as `--max-rel-misfit` on `seriesxrd-analyze`, as a field in
  the Analysis GUI and the worker config, and recorded in the `/peaks` and
  `/residual` attributes.
- `/peaks/chi2_local` and `/peaks/rel_misfit` — the two per-peak measures the
  quality decision is made on, stored so a consumer can apply its own standard.
- `peaks.quality_tier()` and `TIER_REJECT` / `TIER_POSITION` /
  `TIER_QUANTITATIVE`. "Good peak" is not one claim: a reflection whose centre
  is solid can still be modelled too poorly for its area or width to mean
  anything, and `flag` alone cannot say so — a weak peak passes the
  noise-limited test while its rms residual is tens of percent of its height.
  Hard failures (no convergence, width or centre at a bound) are rejections at
  every tier regardless of residual size. Nothing in the pipeline calls this
  yet: `identify`, `fractions` and `microstructure` all still take `flag == 0`.
  Wiring the tiers in changes what those report, and loosening what
  identification accepts needs a false-attribution measurement on labelled data,
  so it is deliberately not riding along with the gate change.

### Changed

- `seriesxrd-spots` records **which** frames a run excluded, as
  `/spots/excluded_frames`, not just how many. Only the count was stored, and
  nothing else in the file said what had been dropped, so a spot list could not
  be regenerated from its own provenance.
- `run_identification` simulates each candidate phase's reflection list in
  parallel (`num_workers`) instead of serially in the parent process, and
  reports how long it took. On an open-set run over a large library this was
  the dominant cost before any frame was scored.
- `benchmark.run_benchmark` takes `num_workers` (`--workers` on
  `seriesxrd-benchmark`), defaulting to one less than the core count, so the
  Step-3a verification pass is no longer single-threaded.

## [0.3.1] - 2026-07-28

### Fixed

- CIFs whose site occupancies sum above 1 — the norm for natural-sample and
  mineral structures, where a site is shared between species — no longer make a
  phase silently unusable. pymatgen's strict default returns *no structure* for
  these, so the phase was accepted into the library at import time and then
  skipped at rank and identify time with only a log line. `structure_from_cif`
  now retries with a relaxed `occupancy_tolerance`
  (`phases.CIF_OCCUPANCY_TOLERANCE`, default 5.0), warns when it does so
  because rescaling occupancies perturbs calculated relative intensities, and
  records the rescale in the imported phase's notes. Peak positions, which is
  what identification decides on, are unaffected. Measured on 243 mineral CIFs
  from COD: 216 parsed before, 232 now.
- A CIF carrying no atomic coordinates at all (`_atom_site_fract_*` set to `?`,
  common for database entries that refined only a unit cell) now raises a
  message saying exactly that, instead of pymatgen's generic "Invalid CIF file
  with no structures!". Other parse failures report the underlying error rather
  than swallowing it.

### Added

- `phases.structure_from_cif()` — the CIF-to-Structure entry point used by
  `parse_cif` and `structure_from_phase`, with an `occupancy_tolerance`
  parameter. Pass `1.0` to restore strict pre-0.3.1 parsing.

## [0.3.0] - 2026-07-23

### Added

- A GSAS-II sequential-refinement round trip: exports now include explicit
  frame/group manifests and a standalone GPX-to-JSON helper, while
  `seriesxrd-import-gsas` (also available in the Analysis GUI) atomically
  imports refined weight fractions with uncertainties, unit cells, and fit
  quality under `/refinement` without replacing the earlier `/fractions`
  screening estimates.

### Changed

- The Analysis GUI now presents export, external refinement, and result import
  together on a dedicated **Refinement → GSAS-II round trip** page.
- Live Mocha/Latte theme switching from the unified application's View menu,
  with a per-user preference and `--theme` overrides for standalone stage
  launchers.

### Changed

- Shared ttk, raw-Tk, and embedded-Matplotlib styling now follows one mutable
  semantic palette without restarting GUI panes or their worker processes.
- Publication figure exports use a predictable light palette on a white
  background independently of the active UI theme.

## [0.2.0] - 2026-07-21

### Changed

- Adopted SeriesXRD as the project, Python package, application, and
  command-line tool name before the first public release.
- Clarified the calibration → reduction → analysis workflow and GUI labels.
- Analysis manifests now record the real SeriesXRD version
  (`seriesxrd_version`) separately from the file-layout `schema_version`;
  the analysis HDF5 carries root version attrs and a `/provenance` group
  (effective configuration, dependency versions, platform, input-file
  fingerprints), and each appending step records itself under
  `/provenance/steps/<step>`.
- Dependency declarations are truthful and tested: the environment check
  covers scipy and h5py as core requirements, `pyproject.toml` declares
  minimum versions validated by a lowest-supported-dependencies CI job,
  and `environment.yml` lists scipy/h5py explicitly.
- The Analysis stage navigates through a hierarchical left rail
  (Configure / Run / Review / Export) instead of a single row of 12 tabs.
- Plot axes use standard scientific notation — q (Å⁻¹), 2θ (°),
  Azimuth (°), Intensity (counts) — everywhere a person reads them;
  internal unit codes are unchanged.
- Successful saves and exports notify through the status bar instead of
  modal dialogs; errors remain modal.
- HDF5 inspection shows a human-readable summary by default, with the raw
  tree and full attributes behind an "Advanced details" toggle.
- Corrected phase-library source attributions (Pt and Si author lists,
  Re citation) and pinned the marker EOS parameters to named literature
  scales: Au to Anderson et al. 1989 as recommended (167 GPa / 5.5), Pt
  re-cited to the Fei et al. 2007 Vinet scale (273 GPa / 5.20), Re to
  Anzellini et al. 2014's Vinet fit unrounded (352.6 GPa / 4.56) — see
  `docs/phase-sources.md`.

### Added

- Unified desktop navigation and automatic stage handoffs.
- GUI access to texture, spot tracking, phase-fraction, microstructure, and
  refinement export tools.
- Continuous integration, distribution checks, citation metadata, and
  community contribution guidance.
- Tag-triggered release pipeline: build once, install-test wheel and sdist,
  TestPyPI, manual approval, PyPI Trusted Publishing with attestations, and
  an automatic GitHub release (`.github/workflows/release.yml`;
  `docs/releasing.md` documents the one-time setup).
- CI matrix: Python 3.10–3.14 on Ubuntu plus the newest Python on Windows
  and macOS, a dependency-floor job, wheel/sdist install smoke tests, a
  headless GUI startup test under xvfb, and a weekly ML-extras run.
- Documentation set for publication: `docs/architecture.md`,
  `docs/file-format.md`, `docs/validation.md` (validation and limitations),
  and `docs/phase-sources.md` (DOI-verified bibliography for every bundled
  phase-library value).
- Run page preflight (input, frame count, steps, output, warnings) and a
  completion summary with "Review results" / "Open output folder" actions.
- Help menu: user guide, demonstration, validation/limitations, citation,
  Report a problem, and Copy diagnostics (a support-ready provenance
  report); About states license and repository.
- Tools → Model development dialog: GUI access to corpus screening,
  benchmarking, and learned-scorer training with live output.
- Figure-export presets (screen / presentation / publication with
  PNG/SVG/PDF) and an `export_provenance.txt` sidecar on frame exports.
- Keyboard-accessible tooltips (focus shows, Escape dismisses) and an
  ellipsized workspace path in the header (full path in tooltip,
  click to copy).
- Governance/maintainer policy (`GOVERNANCE.md`), pull-request template,
  and the NSF REU funding acknowledgment (Award No. 2547979) in
  `CREDITS.md`.
- Six test files that pytest previously never collected (background,
  peaks, fit-source, analysis-review, spots, smoke) now run in CI.
