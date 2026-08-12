# File formats

The HDF5 layouts SeriesXRD writes and reads. Two files carry the pipeline:
the **reduced** file (output of the Reduction stage) and the **analysis**
file (created by analysis Step 1; later steps append groups to it). All
writers are atomic (`.tmp` + `os.replace`) except the live watch-mode file,
which trades atomicity for append speed and is superseded by a normal full
reduction afterwards.

Axis convention: the radial axis is q (Å⁻¹) by default; 2θ (°) is
selectable at reduce time and every consumer handles both (`unit` attr).

## Reduced HDF5 (`reduce/processing.py`)

```
/  attrs: schema_version, seriesxrd_version, created_at, unit, poni_text,
          poni_sha256, mask_sha256, npt_1d, npt_1d_mode, npt_1d_suggested, ...
/patterns/intensity            (N_frames, N_bins)  azimuthal MEAN
/patterns/intensity_robust     (N_frames, N_bins)  spot-suppressed: mean of a
                               narrow azimuthal quantile band around the median
/patterns/intensity_sigmaclip  (N_frames, N_bins)  optional sigma-clipped
                               trimmed mean (keeps textured-ring peaks the
                               median drops while rejecting diamond spots)
/patterns/intensity_straightened         (N_frames, N_bins)  optional; cake-
                               de-waved azimuthal mean (reduce/straighten.py)
/patterns/intensity_straightened_robust  (N_frames, N_bins)  optional; cake-
                               de-waved spot-suppressed median (NaN for frames
                               without a saved cake)
/patterns/radial               (N_bins,)  q or 2θ axis
/cakes/intensity               (N_cakes, N_radial, N_azimuthal)  optional
/cakes/radial, /cakes/azimuthal, /cakes/frame_index
/frames/filename, ok, seconds, excluded, frame_index, thumb
/frames/pressure, temperature, timestamp   placeholders (pressure seeded NaN;
                               populated downstream by frame metadata import).
                               For HDF5/NeXus stack inputs, timestamp,
                               temperature, and /frames/pos_x, pos_y are
                               harvested from the container at reduce time.
/texture/frame, ring_r0, texture_index, po_amplitude, po_phase_deg,
         spotty_frac, coverage   optional; written by reduce/texture.py
```

The live watch-mode variant (`seriesxrd-watch` → `*_live.h5`) uses the same
schema with `live_mode=True`, resizable datasets appended in arrival order,
and no cakes or thumbnails.

## Analysis HDF5

### Created by Step 1 (`analysis/background.py`)

```
/  attrs: schema_version, seriesxrd_version, created_at, source_reduced,
          unit, wavelength, max_half_window, n_passes, use_lls, has_sigmaclip,
          robust_source, n_straightened, signal_frac_clean, spotty_sample,
          npt_1d, npt_1d_mode, npt_1d_suggested
/provenance  attrs: seriesxrd_version, schema_version, tool, created_at,
          python_version, platform, config_json, dependencies_json, and
          per-input identity: input_<name>_path/_bytes/_mtime/_sha256/
          _hash_kind (full SHA-256 up to 64 MiB, head/tail sample above)
/provenance/steps/<step>  attrs: tool, seriesxrd_version, schema_version,
          created_at — one record per appending analysis step below
/radial                        (N_bins,)
/frames/filename               (N,)  copied from the reduced file
/frames/contamination          (N,)  integrated positive spot residual
/frames/flagged                (N,)  bool, contamination > threshold (optional)
/frames/excluded               (N,)  bool, carried from the reduce stage
/frames/pressure               (N,)  GPa; carried from reduced, else parsed
                               from filenames. NaN where unknown.
/frames/pressure_sigma         (N,)  GPa per-frame uncertainty (CSV import)
/frames/temperature, timestamp (N,)  carried when present
/frames/pos_x, pos_y           (N,)  stage positions (mapping scans)
/frames/user_edited            (N,)  bool; values a human set survive
                               re-parsing and Step-1 rebuilds
/background/clean              (N, N_bins)  = robust − baseline
/background/baseline           (N, N_bins)  SNIP estimate
/background/spot_residual      (N, N_bins)  = mean − robust
/background/sigmaclip_residual (N, N_bins)  = sigmaclip − robust (optional)
```

### Appended by Step 2 (`analysis/peaks.py`)

```
/peaks  attrs: schema_version, seriesxrd_version, source, sensitivity, ...
/peaks/counts      (N_frames,)  peaks per frame
/peaks/frame       (P,)  frame index for each peak      (P = sum(counts);
/peaks/center      (P,)  position on the radial axis     ragged layout —
/peaks/amplitude   (P,)  height                          peak count varies
/peaks/fwhm        (P,)  full width at half maximum      per frame)
/peaks/eta         (P,)  Lorentzian fraction ∈ [0, 1]
/peaks/area        (P,)  integrated intensity
/peaks/chi2        (P,)  reduced chi-square of the whole joint fit — the
                         GROUP's adequacy, identical for every peak fitted
                         together. Reported for inspection; not a rejection
/peaks/chi2_local  (P,)  reduced chi-square over this peak's own span
/peaks/rel_misfit  (P,)  rms residual over that span, as a fraction of this
                         peak's own height
/peaks/flag        (P,)  int; 0 = good, else bitmask (low amplitude, bad
                         chi², center drift, width at bound, no convergence)
/peaks/center_err, amplitude_err, fwhm_err   (P,)  1σ estimated standard
                         deviations from the least-squares covariance
```

`chi2_local` and `rel_misfit` are the two measures the `FLAG_BAD_CHI2` decision
is made on, and a peak has to fail both to be flagged (`max_chi2` and
`max_rel_misfit`). They model two different error regimes: a weak peak is
limited by random noise, so its residual matters in units of the noise floor; a
bright peak exposes systematic profile mismatch, so what matters is the residual
as a fraction of its own height. One threshold on either measure alone
mis-judges the other regime.

The noise used is the pattern's MAD background floor, not per-point counting
statistics. That is deliberate: these patterns are azimuthal quantile-band
means rather than raw counts, and an intensity-dependent noise model fitted to
them is unstable at this sampling (peaks span ~4 bins, so every high-intensity
bin sits on a steep flank and the estimator measures the peak's slope instead
of its noise — measured gains ranged 1.96 to 30.9 across frames).

Both columns are exposed so a consumer can apply its own standard: a peak can be
sound enough for position-based phase attribution while being too poorly modelled
for quantitative area or width use. Nothing downstream currently tiers on them —
`identify`, `fractions` and `microstructure` all still take `flag == 0` — because
loosening what identification accepts needs a false-attribution measurement on
labelled data first.

### Appended by Step 3a (`analysis/identify.py` + `analysis/residual.py`)

```
/identify  attrs: p_min, p_max, rel_tol, pressure_window, pressure_sigma_k,
                  min_matched, intensity_k, ...
/identify/<phase>/pressure, score, confidence, recall, precision, n_matched,
                  prior_penalty, intensity_corr   (N,) per frame
/identify/<phase>  attrs: pressure_model, pressure_assumption, prior_penalized
/identify/<phase>/refl_d, refl_w, refl_hkl   cached ambient reflections
/peaks/phase                  (P,) str  phase attributed to each fitted peak
                              ("" = unexplained)
/residual/clean               (N, N_bins)  clean minus reconstructed peaks of
                              phases that cleared the evidence gate
/residual/explained_counts    (N,) int
/residual/unexplained_counts  (N,) int
/residual/peaks/counts, frame, center, amplitude, fwhm   peaks re-fitted on
                              the residual (input to Step 3c)
```

### Appended by Step 3b (`analysis/ml_rank.py`)

```
/ml/candidates  attrs: requested_source, source, resolved_source, top_k,
                method, fwhm_d, fwhm_q, fwhm_q_poly, phases, clip_negative,
                normalize, n_points
/ml/candidates/<phase>/score     (N,)  per-frame similarity to the phase
/ml/candidates/<phase>/pressure  (N,)  pressure the best score used
/ml/candidates/topk_names        (N, top_k) str  ranked candidates per frame
/ml/candidates/topk_score        (N, top_k)
```

### Later analysis groups

```
/unknowns        Step 3c (unknowns.py): obs/, tracks/, clusters/,
                 fingerprint/ — residual peaks linked into gap-tolerant
                 tracks, co-occurrence clusters, per-cluster d-fingerprints
/fractions       names (P,), fractions (N, P). fractions.py writes
                 semi-quantitative intensity shares (method =
                 intensity_share | rir).
/refinement      refine_import.py: GSAS-II names (P,), refined fractions and
                 fraction_esd (N, P); per-frame source_histogram, group_size,
                 rwp, gof, converged; cell and cell_esd (N, P, 7), with columns
                 a,b,c,alpha,beta,gamma,volume. A grouped histogram is
                 replicated to its member frames and group_size records that
                 those rows share one refinement result. Importing this group
                 does not replace the screening estimates in /fractions.
/microstructure  microstructure.py: Williamson–Hall size_A, strain, r2 per
                 frame (flagged uncorrected without an instrument profile)
/spots           spots.py: single-crystal reflections tracked in cake space
                 (written to the analysis file, or <reduced>_spots.h5 when
                 no analysis file is given). obs/ per-frame blob detections
                 with pressure and d; scans/ per-scan groups; tracks/
                 pressure-ordered (azimuth, q) links with d0 and dd_dp.
```

## Correlation HDF5 (`correlations/processing.py`)

The fourth stage reads an Analysis HDF5 and writes
`correlations_powder.h5` or `correlations_single_crystal.h5`; it never mutates
the Analysis file. The matching manifests are `manifest_powder.json` and
`manifest_single_crystal.json`. Both sample types can therefore share one
result directory without overwriting each other's numerical artifacts. Let
`M` be the retained frame count, `K` the retained all-peak/all-observation
count, `R` the radial-bin count, `W` the window count, and `L=63` the
positive-lag fingerprint length from each 64-point resampled window.

```text
/  attrs: tool, schema_version, seriesxrd_version, created_at, sample_type,
          source_requested, source_resolved, source_analysis, unit,
          n_frames, n_peaks, n_windows, all_peak_policy,
          roi_area_method, roi_area_directional, order_by, order_label
/provenance                         standard input identity + effective config
/transform attrs: method="log_squared", scale, scale_quantile, noise_floor,
          epsilon, epsilon_floor, formula, signed_formula,
          position_in_pipeline, scale_estimate, noise_estimate

/patterns/radial                   (R,)
/patterns/original_positive        (M, R)  pre-Log² positive source; waterfall height
/patterns/log_squared              (M, R)  positive-clipped bounded ROI source
/patterns/log_squared_signed       (M, R)  signed-input bounded window source

/frames/index                      (M,)    original Analysis frame indices
/frames/filename                   (M,)    UTF-8
/frames/pressure                   (M,)    GPa, NaN where unavailable
/frames/order_value                (M,)    the order_by axis value per frame
                                           (frame index when order_by="frame")

/peaks/id                          (K,)    stage-local anchor id
/peaks/source_index                (K,)    row in /peaks or /spots/obs
/peaks/frame_row                   (K,)    row in this artifact's frame arrays
/peaks/original_frame              (K,)    Analysis frame index
/peaks/local_peak                  (K,)    deterministic slot within frame
/peaks/center, width, half_width   (K,)    native radial unit
/peaks/area, pressure, track       (K,)    upstream values/provenance
/peaks/valid                       (K,)    bool; ROI support inside the radial
                                           axis with no masked bin inside it.
                                           Invalid anchors have structurally
                                           NaN score rows and no per-anchor
                                           plots

/anchor_maps/profile_coordinate    (65,)   normalized review-sampling coordinate
/anchor_maps/roi_profiles_log_squared (K, 65) review samples only
/anchor_maps/roi_feature_log_squared  (K,) single-crystal runs only: 1D radial
                                           mean approximation to the raw-pixel
                                           Log² ROI feature
/anchor_maps/roi_area              (K, K)  anchor-to-target similarity
/anchor_maps/location              (K, K)  center similarity; no intensity transform

/windows attrs: width, step                native radial unit; width <= selected span
/windows/start, end                (W,)    native radial unit
/windows/label                     (W,)    UTF-8
/windows/acf_features              (M, W, L) standardized positive-lag FFT-ACF
/windows/across_direct             (W, M, M) standardized transformed vectors
/windows/across_acf                (W, M, M) standardized positive-lag FFT-ACF
/windows/within_acf                (M, W, W) standardized positive-lag FFT-ACF

/tracks  attrs: linker="seriesxrd.analysis.unknowns.link_tracks",
          similarity="mutual_sqrt_directional_roi", exploratory=True,
          transition_rule, n_tracks, group_by, link_tol_fwhm, max_gap,
          min_track_frames, min_roi_similarity, order_by
/tracks/obs/track, peak_id         (Nobs,) linked observations; peak_id is
                                           /peaks/id
/tracks/summary/id, n_obs, first_frame_row, last_frame_row, center_first,
          center_last, axis_first, axis_last, group, mean_similarity   (T,)
/tracks/edges/track, peak_from, peak_to, similarity, center_shift,
          axis_gap                 (E,)    consecutive linked pairs; similarity
                                           is the mutual sqrt(S(A->B)*S(B->A))
/tracks/intervals/order_pos, frame_row_from, frame_row_to, axis_from,
          axis_to, group, births, deaths, n_active, median_center_shift,
          window_direct_median, transition_candidate   (per-group M_g-1 rows)
/tracks/group_label                (G,)    UTF-8 scan/folder labels
```

`--order-by` (default `frame`) stably orders the retained frames by a
`/frames` metadata axis before anything downstream sees them; frames missing
the metadata sort last and are excluded from track linking. `/tracks` is
present unless the run used `--no-tracks`; it is exploratory throughout — a
track is a linking hypothesis and a flagged interval is a coincidence of
changes worth inspecting (the exact rule is recorded in `transition_rule`),
never a confirmed transition. Older artifacts without `/peaks/valid`,
`/frames/order_value`, or `/tracks` remain fully readable.

For powder, `half_width = 0.75 × width` and `/anchor_maps/roi_area`
is a directional integrated IoU on the anchor's absolute native-radial
support. The target is zero outside its own support; profiles are not
recentered or width-normalized, so the matrix need not be symmetric. For
single crystal, every `/spots/obs` row remains independent and the score is
`min(feature_i, feature_j) / max(feature_i, feature_j)` (`both zero = 0`:
two ROIs that both carry no signal share absence, not similarity).
The feature is a one-dimensional positive-Log² radial ROI approximation; it
is not a raw-detector-pixel measurement. The track column is never used to
group, filter, or score observations. In both sample modes, target peaks from
the anchor's own frame are structural `NaN` cells and are blank in plots.

ROI processing uses
`clip(max(I, 0) / scale, 0, 1)` before squaring and applying the bounded Log²
formula. Window processing uses signed background residuals normalized by the
same pooled `scale` and `epsilon`, clips them to `[-1, 1]`, and then squares
them inside the same Log² formula. FFT-ACF fingerprints are standardized,
contain positive lags only, and exclude lag zero. The MVP does not write the
prototype's `shift_tolerant_secondary` or same-scan aggregate because the
Analysis HDF5 has no stable `scan_id`.

Review images are replaceable artifacts beside the HDF5:

```text
heatmaps/<powder|single_crystal>/
  roi_area/[pressure_*_GPa/]anchor_*.png
  location/[pressure_*_GPa/]anchor_*.png
  waterfall/[pressure_*_GPa/]anchor_*.png
  window_across/direct/window_*.png
  window_across/acf/window_*.png
  window_within/acf/frame_*.png
```

Pressure subfolders are labels only and are created when finite pressure
metadata exist. `manifest_<sample_type>.json` records the source, shared
transform parameters, matrix counts, algorithms, and the exact current PNG
list. Review images remain separated under `heatmaps/<sample_type>` when the
result directory is shared.

The arrays `/windows/across_direct`, `/windows/across_acf`, and
`/windows/within_acf` retain their complete square matrices. Their PNG review
views deliberately mask the diagonal and upper triangle, leaving only the
strict lower triangle because the omitted values are self-correlations or
mirrored duplicates. In the Results browser, across-frame images are filed
under **All pressures**; within-frame images are filed under the corresponding
frame pressure.

## JSON manifests

Every stage run also returns/writes a JSON manifest whose header is
standardized (`core/provenance.manifest_provenance`):

```json
{
  "tool": "seriesxrd.analysis.peaks",
  "seriesxrd_version": "0.2.0",
  "schema_version": "1",
  "created_at": "...",
  "...": "per-stage fields"
}
```

`seriesxrd_version` is the package version that wrote the artifact;
`schema_version` only changes when the file layout changes.
