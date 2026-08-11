"""Step 1 of the analysis pipeline: background-scattering isolation.

Implements the first "identify → record → remove" pass of the categorization
workflow (see categorization.py): separate, per frame, the two background-like
contributions of a diamond-anvil-cell pattern from the sample/marker signal.

Two distinct operations, both recorded so nothing is silently discarded:

1. Single-crystal "spot" residual (diamond anvil reflections, coarse-grain
   spottiness). Powder rings are uniform in azimuth; single-crystal spots are
   not. The reduce stage already produced both the azimuthal MEAN
   (``patterns/intensity``) and the azimuthal MEDIAN (``patterns/intensity_robust``)
   integrations, so:
       spot_residual = mean - robust          (the spot/texture contribution)
       robust        = spot-suppressed powder signal
   A per-frame ``contamination_score`` (integrated positive residual) flags
   frames dominated by diamond spots.

2. Smooth + amorphous background (Compton/air + gasket/pressure-medium humps).
   Estimated from the robust pattern with SNIP (Statistics-sensitive Non-linear
   Iterative Peak-clipping) on an LLS-transformed intensity, then:
       baseline = SNIP(robust)
       clean    = robust - baseline

Pure-numpy logic (no pyFAI). ``separate_background`` works on a single pattern;
``run_background_separation`` drives a whole reduced HDF5.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .parallel import chunk_ranges, process_map_or_serial, resolve_workers
from ..core.config import VERSION, now_iso
from ..core.provenance import manifest_provenance, write_provenance


# ---------------------------------------------------------------------------
# SNIP baseline
# ---------------------------------------------------------------------------

def _lls(y: np.ndarray) -> np.ndarray:
    """Log-Log-Sqrt operator — compresses dynamic range so strong peaks do not
    shadow the SNIP background estimate (Morhac 2000)."""
    return np.log(np.log(np.sqrt(np.clip(y, 0.0, None) + 1.0) + 1.0) + 1.0)


def _lls_inv(z: np.ndarray) -> np.ndarray:
    return (np.exp(np.exp(z) - 1.0) - 1.0) ** 2 - 1.0


def snip_baseline(y, max_half_window: int = 40, n_passes: int = 1,
                  use_lls: bool = True) -> np.ndarray:
    """Estimate a smooth baseline under the peaks of a 1D pattern via SNIP.

    For increasing half-window ``m`` = 1..max_half_window, each point is replaced
    by ``min(y[i], (y[i-m] + y[i+m]) / 2)``; anything narrower than the window
    survives as a peak, anything broader is clipped to the baseline.

    Parameters
    ----------
    y : array
        Intensity vs radial bin (work in q for uniform peak widths). NaNs are
        interpolated across for the estimate, then restored.
    max_half_window : int
        Widest feature (in BINS) treated as a peak. Set ~1.5-2x the broadest
        Bragg peak half-width; wider than this is treated as background.
    n_passes : int
        Repeat the full 1..M sweep this many times (>1 pushes the baseline
        lower; risks eroding broad peaks).
    use_lls : bool
        Apply the LLS transform (recommended when intensity spans >2 decades).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return np.full(n, np.nan)
    yf = y.copy()
    if not finite.all():
        idx = np.arange(n)
        yf = np.interp(idx, idx[finite], y[finite])

    work = _lls(yf) if use_lls else yf.copy()
    idx = np.arange(n)
    m_max = max(1, int(max_half_window))
    for _ in range(max(1, int(n_passes))):
        for m in range(1, m_max + 1):
            lo = np.clip(idx - m, 0, n - 1)
            hi = np.clip(idx + m, 0, n - 1)
            work = np.minimum(work, 0.5 * (work[lo] + work[hi]))

    base = _lls_inv(work) if use_lls else work
    base = np.minimum(base, yf)        # baseline never sits above the data
    base[~finite] = np.nan             # don't invent values where data was absent
    return base


# ---------------------------------------------------------------------------
# Per-pattern separation
# ---------------------------------------------------------------------------

def spot_residual(mean: np.ndarray, robust: np.ndarray) -> np.ndarray:
    """Single-crystal/spot contribution = azimuthal mean - azimuthal median."""
    return np.asarray(mean, float) - np.asarray(robust, float)


def contamination_score(residual: np.ndarray) -> float:
    """Integrated POSITIVE spot residual for one frame (a diamond-contamination
    metric; negative values are just noise and are ignored)."""
    r = np.asarray(residual, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float(np.sum(np.clip(r, 0.0, None)))


def _significant_bins(row: np.ndarray, k: float = 8.0) -> int:
    """Bins standing ``k``×MAD above a row's median — a crude Bragg-signal count."""
    r = np.asarray(row, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0
    med = float(np.median(r))
    mad = 1.4826 * float(np.median(np.abs(r - med)))
    if mad <= 0:
        return 0
    return int(np.sum(r > med + k * mad))


def diagnose_signal_channels(clean: np.ndarray, spots: np.ndarray,
                             excluded: "Optional[np.ndarray]" = None
                             ) -> Dict[str, Any]:
    """Where does the Bragg signal actually live — the spot-suppressed ``clean``
    or the azimuthal-mean excess?

    For a fine-grained powder the two agree (``signal_frac_clean`` ≈ 1). For a
    coarse-grained / spotty / near-single-crystal sample the rings are a few
    azimuthal spots, so the median-based channels reject the SAMPLE itself as
    outliers: ``clean`` holds only background while the whole pattern sits in
    ``spot_residual`` (``signal_frac_clean`` → 0). Fitting the default channel
    then fits noise — the correct source is ``mean``. Returns ``{ok,
    signal_frac_clean, spotty_sample, n_frames_used}``; the diagnosis is
    data-driven per dataset (no sample-type assumptions), and undecidable when
    no frame shows significant bins at all.
    """
    n = clean.shape[0]
    live = (~excluded if excluded is not None and excluded.size == n
            else np.ones(n, bool))
    fracs = []
    for i in np.nonzero(live)[0]:
        mean_row = clean[i] + spots[i]
        n_mean = _significant_bins(mean_row)
        if n_mean < 3:                       # frame carries no clear peaks at all
            continue
        fracs.append(_significant_bins(clean[i]) / float(n_mean))
    if not fracs:
        return {"ok": False, "signal_frac_clean": float("nan"),
                "spotty_sample": False, "n_frames_used": 0}
    frac = float(np.median(fracs))
    return {"ok": True, "signal_frac_clean": frac,
            "spotty_sample": bool(frac < 0.5), "n_frames_used": len(fracs)}


def separate_background(intensity, intensity_robust,
                        max_half_window: int = 40, n_passes: int = 1,
                        use_lls: bool = True) -> Dict[str, Any]:
    """Run the full Step-1 separation on one frame's pair of 1D patterns.

    Returns a dict with ``spot_residual``, ``baseline``, ``clean`` (all arrays)
    and ``contamination`` (scalar). ``clean`` is the spot-suppressed,
    background-subtracted powder signal handed to peak fitting.
    """
    mean = np.asarray(intensity, float)
    robust = np.asarray(intensity_robust, float)
    if robust.shape != mean.shape:
        raise ValueError(f"shape mismatch: intensity {mean.shape} vs robust {robust.shape}")
    spots = spot_residual(mean, robust)
    baseline = snip_baseline(robust, max_half_window=max_half_window,
                             n_passes=n_passes, use_lls=use_lls)
    clean = robust - baseline
    return {
        "spot_residual": spots,
        "baseline": baseline,
        "clean": clean,
        "contamination": contamination_score(spots),
    }


# ---------------------------------------------------------------------------
# Dataset driver (reduced HDF5 -> analysis HDF5)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"


def _previous_user_metadata(out_path: Path) -> "Optional[Dict[str, Any]]":
    """Harvest user-edited frame metadata from an existing analysis file at
    ``out_path`` before it is rebuilt. Returns ``{by_name, by_index, n}`` where
    each entry maps to ``(pressure, sigma, temperature)`` tuples, or None when
    there is nothing to carry."""
    import h5py  # type: ignore
    if not out_path.is_file():
        return None
    try:
        with h5py.File(str(out_path), "r") as h5:
            fr = h5.get("frames")
            if fr is None or "user_edited" not in fr:
                return None
            mask = np.asarray(fr["user_edited"][:], dtype=bool)
            if not mask.any():
                return None
            def _f(key):
                return (np.asarray(fr[key][:], dtype="f8")
                        if key in fr else np.full(mask.size, np.nan))
            chans = tuple(_f(k) for k in
                          ("pressure", "pressure_sigma", "temperature",
                           "pos_x", "pos_y"))
            names = ([x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray))
                      else str(x) for x in fr["filename"][:]]
                     if "filename" in fr else [])
            by_name: Dict[str, tuple] = {}
            by_index: Dict[int, tuple] = {}
            for j in np.nonzero(mask)[0]:
                rec = tuple(c[j] for c in chans)
                by_index[int(j)] = rec
                if j < len(names):
                    by_name[names[j]] = rec
            return {"by_name": by_name, "by_index": by_index, "n": int(mask.size)}
    except Exception:
        return None   # unreadable/partial file — nothing to carry


def _parse_wavelength(poni_text: str) -> float:
    """Extract the wavelength in Å from pyFAI PONI text (stored in metres)."""
    m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)", str(poni_text), re.IGNORECASE)
    if not m:
        return 0.0
    try:
        wl = float(m.group(1))
    except ValueError:
        return 0.0
    return wl * 1e10 if 0 < wl < 1e-6 else wl


def _bg_chunk(payload):
    """Worker: run separate_background over a contiguous chunk of frames."""
    mean_c, robust_c, mhw, npasses, lls = payload
    m, nb = mean_c.shape
    clean = np.full((m, nb), np.nan, "f4")
    base = np.full((m, nb), np.nan, "f4")
    spots = np.full((m, nb), np.nan, "f4")
    contam = np.zeros(m, "f8")
    for j in range(m):
        res = separate_background(mean_c[j], robust_c[j], max_half_window=mhw,
                                  n_passes=npasses, use_lls=lls)
        clean[j] = res["clean"]; base[j] = res["baseline"]
        spots[j] = res["spot_residual"]; contam[j] = res["contamination"]
    return clean, base, spots, contam


def run_background_separation(
    reduced_h5: "str | Path",
    out_h5: "Optional[str | Path]" = None,
    *,
    max_half_window: int = 40,
    n_passes: int = 1,
    use_lls: bool = True,
    contamination_threshold: "Optional[float]" = None,
    robust_source: str = "robust",
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Apply Step-1 background separation to every frame of a reduced HDF5.

    ``robust_source`` selects the spot-suppressed input the whole stage is built
    on. ``"robust"`` (default) uses the reduce-side azimuthal median
    (``patterns/intensity_robust``). ``"straightened"`` uses the cake-straightened,
    de-waved channels written by ``reduce.straighten.straighten_reduced``
    (``intensity_straightened_robust`` as the median, ``intensity_straightened``
    as the mean) so a sample-off-calibrant offset no longer splits each ring into
    a double-horned peak. Frames without a saved cake (straightened = NaN) fall
    back per-frame to the ordinary channels, and the reduce-side sigmaclip channel
    (still wavy) is skipped in this mode.

    Reads ``patterns/intensity``, ``patterns/intensity_robust``,
    ``patterns/radial`` (and ``frames/...`` for provenance). Writes a
    self-contained analysis HDF5:

        /  attrs: schema_version, seriesxrd_version, created_at, source_reduced,
                  unit, wavelength, max_half_window, n_passes, robust_source,
                  n_straightened
        /provenance  attrs: version/dependency/config/input-hash record
                     (see core/provenance.py); /provenance/steps/<step> is
                     appended by each later analysis step
        /radial                      (N_bins,)
        /frames/filename             (N,)   copied from the reduced file
        /frames/contamination        (N,)   per-frame spot score
        /frames/excluded             (N,)   bool, carried over from the reduce stage
        /frames/flagged              (N,)   contamination > threshold (if given)
        /frames/pressure             (N,)   GPa; carried from the reduced file, or
                                            parsed from the filenames when that is
                                            empty. NaN where unknown. (Step-3 prior.)
        /frames/pressure_sigma       (N,)   GPa; only when carried from a previous
                                            run's user-edited frames
        /frames/temperature          (N,)   K; carried from the reduced file (if any)
        /frames/timestamp            (N,)   str; carried from the reduced file (if any)
        /frames/user_edited          (N,)   bool; frames whose P/σ/T a human set
                                            (GUI edit or CSV import). Their values
                                            are carried forward from the previous
                                            analysis file (matched by filename) so
                                            a re-run cannot resurrect the bad value
                                            the user already corrected.
        /background/clean            (N, N_bins)
        /background/baseline         (N, N_bins)
        /background/spot_residual    (N, N_bins)
        /background/sigmaclip_residual (N, N_bins)  only if the reduced file has
                                     patterns/intensity_sigmaclip (= sigmaclip − robust)

    Returns a manifest dict (also has ``out_h5`` and per-run stats). Progress
    lines ``[ANALYSIS] <done> <total>`` go to stdout for a supervising UI.
    """
    import h5py  # type: ignore

    src = Path(reduced_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reduced HDF5 not found: {src}")
    out = Path(out_h5).expanduser().resolve() if out_h5 else src.with_name(src.stem + "_analysis.h5")
    tmp = out.with_name(out.name + ".tmp")

    with h5py.File(str(src), "r") as h5:
        pat = h5.get("patterns")
        if pat is None or "intensity" not in pat or "intensity_robust" not in pat:
            raise ValueError(
                "Reduced file lacks patterns/intensity_robust — re-run reduction "
                "with the robust (azimuthal-median) pattern enabled; it is required "
                "for diamond-spot separation.")
        mean_all = np.asarray(pat["intensity"][:], dtype=float)
        robust_all = np.asarray(pat["intensity_robust"][:], dtype=float)
        # Optional reduce-side azimuthal sigma-clipped (trimmed-mean) channel —
        # keeps azimuthally-sparse real sample intensity that the median drops,
        # while still rejecting diamond single-crystal spots. Carried into the
        # analysis file as a residual so Step 2 can fit on it.
        sigmaclip_all = (np.asarray(pat["intensity_sigmaclip"][:], dtype=float)
                         if "intensity_sigmaclip" in pat else None)
        # Cake-straightened, de-waved channels (reduce.straighten). The median is
        # spot-suppressed like intensity_robust; the mean like intensity. Only
        # consumed when robust_source="straightened" (below).
        straight_mean_all = (np.asarray(pat["intensity_straightened"][:], dtype=float)
                             if "intensity_straightened" in pat else None)
        straight_median_all = (np.asarray(pat["intensity_straightened_robust"][:], dtype=float)
                               if "intensity_straightened_robust" in pat else None)
        radial = np.asarray(pat["radial"][:], dtype=float) if "radial" in pat else None
        unit = str(h5.attrs.get("unit", ""))
        poni = h5.attrs.get("poni_text", "")
        if isinstance(poni, (bytes, bytearray)):
            poni = poni.decode("utf-8", "replace")
        wavelength = _parse_wavelength(poni)
        # Binning provenance carried forward so any analysis file shows how the
        # 1D bin count was chosen (auto/explicit/fallback + the suggestion).
        npt_prov = {k: h5.attrs[k] for k in
                    ("npt_1d", "npt_1d_mode", "npt_1d_suggested") if k in h5.attrs}
        frames = h5.get("frames")
        names = None
        if frames is not None and "filename" in frames:
            names = [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in frames["filename"][:]]
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)
        # Per-frame metadata carried straight through to the analysis file so
        # Step 3 can use pressure as a prior (see frame_metadata.py). The reduce
        # stage seeds /frames/pressure as an all-NaN placeholder; we backfill it
        # from the filenames below when it arrives empty.
        red_pressure = (np.asarray(frames["pressure"][:], dtype=float)
                        if frames is not None and "pressure" in frames else None)
        red_temperature = (np.asarray(frames["temperature"][:], dtype=float)
                           if frames is not None and "temperature" in frames else None)
        red_pos_x = (np.asarray(frames["pos_x"][:], dtype=float)
                     if frames is not None and "pos_x" in frames else None)
        red_pos_y = (np.asarray(frames["pos_y"][:], dtype=float)
                     if frames is not None and "pos_y" in frames else None)
        red_timestamp = (
            [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else str(x)
             for x in frames["timestamp"][:]]
            if frames is not None and "timestamp" in frames else None)

    # De-waved (straightened) mode: swap the median/mean inputs for the
    # cake-straightened channels so every downstream product (clean, baseline,
    # spot_residual, contamination) is computed on single, un-split rings. Per
    # frame, keep the straightened value where finite and fall back to the
    # ordinary channel elsewhere (cake-less frames stay all-NaN in straightened →
    # fully ordinary; NaN edge bins within a straightened row are filled too).
    robust_source = (robust_source or "robust").strip().lower()
    n_straightened = 0
    if robust_source == "straightened":
        if straight_median_all is None or straight_mean_all is None:
            raise ValueError(
                "robust_source='straightened' but the reduced file has no "
                "patterns/intensity_straightened_robust. Click 'Write straightened "
                "1D' on the reduce stage's Review tab first (it needs saved cakes).")
        if straight_median_all.shape != robust_all.shape:
            raise ValueError(
                "intensity_straightened_robust shape "
                f"{straight_median_all.shape} != intensity_robust "
                f"{robust_all.shape}; re-run 'Write straightened 1D'.")
        have = np.isfinite(straight_median_all).any(axis=1)
        robust_all = np.where(np.isfinite(straight_median_all) & have[:, None],
                              straight_median_all, robust_all)
        mean_all = np.where(np.isfinite(straight_mean_all) & have[:, None],
                            straight_mean_all, mean_all)
        sigmaclip_all = None          # reduce sigmaclip is still wavy — don't mix
        n_straightened = int(have.sum())
        print(f"[ANALYSIS] background source = straightened "
              f"({n_straightened}/{have.size} frames de-waved; the rest fall back "
              f"to the azimuthal median)", flush=True)

    n, nb = mean_all.shape
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)
    clean = np.full((n, nb), np.nan, dtype="f4")
    baseline = np.full((n, nb), np.nan, dtype="f4")
    spots = np.full((n, nb), np.nan, dtype="f4")
    contam = np.zeros(n, dtype="f8")
    workers = resolve_workers(num_workers)
    print(f"[ANALYSIS] background separation: {n} frames, {nb} bins, "
          f"SNIP M={max_half_window} passes={n_passes} workers={workers}", flush=True)

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(mean_all[a:b], robust_all[a:b], max_half_window, n_passes, use_lls)
                    for a, b in ranges]
        done = 0
        # Lazy iterator: the [ANALYSIS] progress lines below stream only while
        # results are consumed one chunk at a time -- do not materialize it.
        results = process_map_or_serial(
            _bg_chunk, payloads, max_workers=workers, label="ANALYSIS")
        for (a, b), (c, bs, sp, ct) in zip(ranges, results):
            clean[a:b] = c; baseline[a:b] = bs; spots[a:b] = sp; contam[a:b] = ct
            done += (b - a)
            print(f"[ANALYSIS] {done} {n}", flush=True)
    else:
        for i in range(n):
            res = separate_background(mean_all[i], robust_all[i],
                                      max_half_window=max_half_window,
                                      n_passes=n_passes, use_lls=use_lls)
            clean[i] = res["clean"]
            baseline[i] = res["baseline"]
            spots[i] = res["spot_residual"]
            contam[i] = res["contamination"]
            if (i + 1) % 25 == 0 or i + 1 == n:
                print(f"[ANALYSIS] {i + 1} {n}", flush=True)

    # sigmaclip channel as a residual on the spot-suppressed median (mirrors how
    # spot_residual = mean − robust is stored), so any source can be rebuilt as
    # clean + <residual> in Step 2 without re-reading the reduced file.
    sigmaclip_residual = None
    if sigmaclip_all is not None and sigmaclip_all.shape == robust_all.shape:
        sigmaclip_residual = (sigmaclip_all - robust_all).astype("f4")

    flagged = None
    if contamination_threshold is not None:
        flagged = contam > float(contamination_threshold)

    # Where does the Bragg signal live? A spotty/coarse-grained sample is
    # rejected by the median-based channels (clean holds only background) —
    # record the diagnosis so Step 2's source="auto" can act on DATA, not on a
    # sample-type assumption.
    diag = diagnose_signal_channels(clean, spots, excluded)
    if diag["spotty_sample"]:
        print(f"[ANALYSIS] WARNING: spotty/coarse-grained sample — only "
              f"{100 * diag['signal_frac_clean']:.0f}% of the significant bins "
              f"survive in the spot-suppressed 'clean' channel (the rest sit in "
              f"spot_residual). The azimuthal median is rejecting the SAMPLE "
              f"itself; peak fitting should use source='mean' (Step 2 auto will).",
              flush=True)

    # Resolve the per-frame pressure channel: carry the reduced value through,
    # but if it is absent / all-NaN (the usual placeholder case) parse it from
    # the filenames so phase identification has a pressure prior with no manual
    # step. A later CSV import (frame_metadata.import_csv_to_analysis) overrides.
    pressure = red_pressure if (red_pressure is not None and red_pressure.size == n) else None
    n_pressure_parsed = 0
    if (pressure is None or not np.any(np.isfinite(pressure))) and names is not None:
        from .frame_metadata import extract_pressures
        parsed = extract_pressures(names)
        if np.any(np.isfinite(parsed)):
            pressure = parsed
    if pressure is not None:
        n_pressure_parsed = int(np.sum(np.isfinite(pressure)))

    # Carry deliberate human corrections across the rebuild: re-running Step 1
    # recreates this file, and without this the re-parsed filename pressure
    # would silently resurrect exactly the value a user already fixed on the
    # Frame metadata tab (e.g. a mistyped '50p7GPa' token). Matched by
    # filename; index fallback when the frame count is unchanged.
    temperature = (red_temperature
                   if red_temperature is not None and red_temperature.size == n
                   else None)
    pressure_sigma = None
    # Stage positions carried from the reduced file (NeXus stack harvest);
    # user-edited values below override per frame.
    pos_x = (red_pos_x if red_pos_x is not None and red_pos_x.size == n else None)
    pos_y = (red_pos_y if red_pos_y is not None and red_pos_y.size == n else None)
    user_mask = None
    prev = _previous_user_metadata(out)
    if prev is not None:
        user_mask = np.zeros(n, dtype=bool)
        n_carried = 0
        for i in range(n):
            rec = None
            if names is not None and i < len(names) and names[i] in prev["by_name"]:
                rec = prev["by_name"][names[i]]
            elif not prev["by_name"] and prev["n"] == n:
                rec = prev["by_index"].get(i)
            if rec is None:
                continue
            p_val, s_val, t_val, x_val, y_val = rec
            if np.isfinite(p_val):
                if pressure is None:
                    pressure = np.full(n, np.nan, "f8")
                pressure[i] = p_val
            if np.isfinite(s_val):
                if pressure_sigma is None:
                    pressure_sigma = np.full(n, np.nan, "f8")
                pressure_sigma[i] = s_val
            if np.isfinite(t_val):
                if temperature is None:
                    temperature = np.full(n, np.nan, "f8")
                temperature[i] = t_val
            if np.isfinite(x_val):
                if pos_x is None:
                    pos_x = np.full(n, np.nan, "f8")
                pos_x[i] = x_val
            if np.isfinite(y_val):
                if pos_y is None:
                    pos_y = np.full(n, np.nan, "f8")
                pos_y[i] = y_val
            user_mask[i] = True
            n_carried += 1
        if n_carried:
            print(f"[ANALYSIS] preserved {n_carried} user-edited frame metadata "
                  f"value(s) from the previous analysis file", flush=True)
        else:
            user_mask = None

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(str(tmp), "w") as o:
            o.attrs.update({
                "tool": "seriesxrd.analysis.background", "schema_version": SCHEMA_VERSION,
                "seriesxrd_version": VERSION, "created_at": now_iso(),
                "source_reduced": str(src), "unit": unit,
                "wavelength": float(wavelength),
                "max_half_window": int(max_half_window), "n_passes": int(n_passes),
                "use_lls": bool(use_lls),
                "has_sigmaclip": bool(sigmaclip_residual is not None),
                "robust_source": robust_source,
                "n_straightened": int(n_straightened),
                "signal_frac_clean": float(diag["signal_frac_clean"]),
                "spotty_sample": bool(diag["spotty_sample"]),
            })
            o.attrs.update(npt_prov)
            if radial is not None:
                o.create_dataset("radial", data=radial)
            gf = o.create_group("frames")
            if names is not None:
                import h5py as _h5
                gf.create_dataset("filename", data=np.array(names, dtype=object),
                                  dtype=_h5.string_dtype(encoding="utf-8"))
            gf.create_dataset("contamination", data=contam)
            gf.create_dataset("excluded", data=excluded)
            if flagged is not None:
                gf.create_dataset("flagged", data=flagged)
            # Frame metadata (pressure prior + provenance). pressure is always
            # written (NaN where unknown) so Step 3 can read a consistent channel.
            gf.create_dataset("pressure",
                              data=(pressure if pressure is not None
                                    else np.full(n, np.nan, "f8")).astype("f8"))
            if pressure_sigma is not None:
                gf.create_dataset("pressure_sigma", data=pressure_sigma.astype("f8"))
            if pos_x is not None:
                gf.create_dataset("pos_x", data=pos_x.astype("f8"))
            if pos_y is not None:
                gf.create_dataset("pos_y", data=pos_y.astype("f8"))
            if user_mask is not None:
                gf.create_dataset("user_edited", data=user_mask)
            if temperature is not None and temperature.size == n:
                gf.create_dataset("temperature", data=temperature.astype("f8"))
            if red_timestamp is not None and len(red_timestamp) == n:
                import h5py as _h5t
                gf.create_dataset("timestamp",
                                  data=np.array(red_timestamp, dtype=object),
                                  dtype=_h5t.string_dtype(encoding="utf-8"))
            gb = o.create_group("background")
            gb.create_dataset("clean", data=clean, compression="gzip", compression_opts=1)
            gb.create_dataset("baseline", data=baseline, compression="gzip", compression_opts=1)
            gb.create_dataset("spot_residual", data=spots, compression="gzip", compression_opts=1)
            if sigmaclip_residual is not None:
                gb.create_dataset("sigmaclip_residual", data=sigmaclip_residual,
                                  compression="gzip", compression_opts=1)
            write_provenance(
                o, tool="seriesxrd.analysis.background",
                schema_version=SCHEMA_VERSION,
                config={
                    "max_half_window": int(max_half_window),
                    "n_passes": int(n_passes), "use_lls": bool(use_lls),
                    "contamination_threshold": contamination_threshold,
                    "robust_source": robust_source,
                    "num_workers": int(num_workers),
                },
                inputs={"reduced": src})
        import os
        os.replace(tmp, out)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    manifest = {
        **manifest_provenance("seriesxrd.analysis.background", SCHEMA_VERSION),
        "source_reduced": str(src),
        "out_h5": str(out),
        "n_frames": int(n),
        "n_bins": int(nb),
        "unit": unit,
        "wavelength": float(wavelength) or None,
        "n_excluded": int(excluded.sum()),
        "n_pressure": int(n_pressure_parsed),
        "max_half_window": int(max_half_window),
        "n_passes": int(n_passes),
        "contamination_threshold": contamination_threshold,
        "n_flagged": int(flagged.sum()) if flagged is not None else None,
        "has_sigmaclip": bool(sigmaclip_residual is not None),
        "robust_source": robust_source,
        "n_straightened": int(n_straightened),
        "signal_frac_clean": diag["signal_frac_clean"],
        "spotty_sample": diag["spotty_sample"],
        "contamination_min": float(np.min(contam)) if n else 0.0,
        "contamination_max": float(np.max(contam)) if n else 0.0,
    }
    print(f"[ANALYSIS] done -> {out}", flush=True)
    return manifest
