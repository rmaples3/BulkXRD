"""Step 2 of the analysis pipeline: peak / profile fitting.

Takes the background-separated ``clean`` patterns from Step 1 (see
``background.py``) and fits every Bragg peak with a pseudo-Voigt profile, so
each reflection is reduced to a small, physically meaningful parameter set:

    center      peak position on the radial axis (q or 2theta, as reduced)
    amplitude   peak height of the profile
    fwhm        full width at half maximum  -> crystallite size & microstrain
    eta         Lorentzian fraction in [0,1] (size vs strain character)
    area        integrated intensity (texture / phase fraction)
    chi2        reduced chi-square of the local fit (goodness)
    flag        0 = good, nonzero = a rejection reason (see FLAG_*)
    center_err, amplitude_err, fwhm_err
                1-sigma uncertainties from the fit covariance (NaN when the
                fit failed) — esd-weighting for Step-3 matching and
                Williamson-Hall error bars

Pipeline per frame:
    1. detect candidate peaks (local maxima with SNR above a MAD noise floor),
    2. group peaks whose windows overlap and fit each group jointly as a sum of
       pseudo-Voigts plus a local constant baseline (Levenberg-Marquardt),
    3. reject implausible fits, record everything.

Across a frame series the peak centers drift slowly (the lattice compresses),
so ``fit_dataset`` optionally *seeds* each frame's detection with the previous
frame's fitted centers — this keeps peak identity coherent for the Step-3
heatmap. The model evaluation is vectorized (numpy broadcasting); per-frame
fits are independent and could be parallelized, but scipy least-squares on
~10^3 frames runs in seconds, so the default backend is plain serial scipy.

Depends on numpy + scipy (scipy is imported lazily). ``run_peak_fitting``
drives a whole analysis HDF5 written by Step 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .parallel import chunk_ranges, process_map_or_serial, resolve_workers
from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance


# Rejection flags (bitwise-OR-able).
FLAG_OK = 0
FLAG_LOW_AMP = 1        # amplitude fell below the noise floor during fitting
FLAG_BAD_CHI2 = 2       # reduced chi-square above threshold (poor fit)
FLAG_CENTER_DRIFT = 4   # center pinned at its ±0.5·FWHM seed bound (ran away)
FLAG_WIDTH_BOUND = 8    # fwhm pinned to a bound (degenerate width)
FLAG_NO_CONVERGE = 16   # optimizer did not converge

# Companion to ``max_chi2``: the largest rms residual, as a FRACTION of the peak
# height, that still counts as a good fit.
#
# The fit weights every point by one scalar sigma — the MAD noise floor of the
# background — so a reduced chi-square measures the misfit in units of the
# BACKGROUND noise. For a fixed *fractional* profile mismatch that grows as
# (peak height / noise)^2, which makes ``max_chi2`` alone an SNR gate: on the
# archived DAC series the fitted log-log slope above SNR 200 is 2.1 (r = 0.87),
# and the strongest reflection of the frame was rejected in 1055 of 1288 frames
# while fitting to about 1% of its own height. Peaks measured well enough are
# rejected for being measured well.
#
# A peak is rejected only when the fit is bad on BOTH counts — bad against the
# noise AND bad relative to the peak itself. Weak peaks are noise-limited, so
# the chi-square clause still governs them; strong peaks are systematics-limited
# and are judged on relative misfit. Both are evaluated over the peak's OWN span
# and against its OWN height, not the enclosing group's: see ``_fit_group``.
#
# The tolerance is not a physical constant — it depends on sampling, overlap,
# background treatment and what the peak will be used for, so it is calibrated,
# not derived. Calibrated on 1402 peaks from 144 frames of a real DAC series,
# excluding peaks that carry a hard failure (no convergence, width or centre
# pinned at a bound) since those are rejected regardless:
#
#     SNR band     median   90th    95th        (synchrotron DAC series)
#     10-30         3.14%   6.05%   6.93%
#     30-100        1.97%   3.46%   4.25%
#     100-300       1.70%   3.27%   3.92%
#     300-1000      1.58%   2.72%   3.10%
#     1000-3000     1.98%   3.46%   3.81%
#
# Nearly flat above SNR 30, which is what makes one threshold workable there.
# Sweeping it, the rejection rate runs 14.98% / 6.56% / 2.78% / 1.43% / 1.00% at
# 2 / 3 / 4 / 5 / 6% and then flattens, so the knee is near 5%; that clears the
# 95th percentile of every band above SNR 30 and keeps the frame's strongest
# reflection in 97.9% of frames (56.9% at a 2% threshold).
#
# GENERALITY. Both measures are dimensionless, and were verified invariant to
# detector gain (x1000) and to a flat pedestal exactly, and to within 1% under a
# doubling of bins per FWHM — so neither has to be re-derived per detector. The
# NUMBER is not universal. On opXRD (2188 peaks from labelled experimental
# patterns taken on many different instruments) the same measure is larger and
# FALLS with brightness rather than staying flat — median 8.1% below SNR 10,
# 2.6% at 30-100, 0.9% at 1000-3000 — because laboratory profiles are broader,
# more asymmetric and more overlapped than a synchrotron DAC ring. The two-clause
# structure is what protects those weak peaks: one with a 10% relative misfit is
# still governed by chi-square and is rejected only if it is also inconsistent
# with the noise. There the threshold barely matters (rejection moves 9.7% ->
# 5.9% across 2-10%), because the rejections are dominated by fits that are bad
# on both counts. Re-calibrate for a very different instrument; tighten to
# 0.03-0.04 for a stricter map, where the cost falls on strong reflections.
DEFAULT_MAX_REL_MISFIT = 0.05

# Half-width, in FWHM, of the span each peak's quality is measured over (further
# restricted to the points where that peak's own profile is the group's tallest —
# see ``_fit_group``). Fixed rather than tied to ``window_factor``: both measures
# average residuals over this span, so widening it dilutes them with baseline.
# Measured on one peak at
# window_factor 2 / 3 / 5, the relative misfit reads 0.88% / 0.79% / 0.65% — a
# calibrated threshold would silently change meaning whenever the fit window
# knob moved, which is not what that knob is for.
QUALITY_WINDOW_FWHM = 2.0


# Fitness tiers. "Good peak" is not one claim: a reflection whose centre is
# solid can still be modelled too poorly for its area or width to mean anything.
# Hard failures — no convergence, width or centre pinned at a bound — are
# failures at every tier regardless of how small the residual is.
TIER_REJECT = "reject"              # unusable
TIER_POSITION = "position"          # centre is usable; area/width are not
TIER_QUANTITATIVE = "quantitative"  # profile is modelled well enough to integrate

_HARD_FLAGS = FLAG_NO_CONVERGE | FLAG_WIDTH_BOUND | FLAG_CENTER_DRIFT


def quality_tier(flag, rel_misfit,
                 max_rel_misfit: float = None):  # type: ignore[assignment]
    """What a fitted peak may be used for. Scalars or arrays.

    ``TIER_REJECT`` for any flagged peak. Otherwise ``TIER_QUANTITATIVE`` when
    the profile residual is within ``max_rel_misfit`` of the peak's own height,
    else ``TIER_POSITION`` — the centre survived the noise-limited test but the
    shape is not modelled well enough to trust an integrated intensity or a
    width from it. That case is real and invisible to ``flag`` alone: a weak
    peak passes on chi-square while its rms residual is tens of percent of its
    height.

    Nothing in the pipeline calls this yet. ``identify``, ``fractions`` and
    ``microstructure`` all still take ``flag == 0``; wiring the tiers in changes
    what those report, so it wants its own validation run rather than riding
    along with the gate change.
    """
    if max_rel_misfit is None:
        max_rel_misfit = DEFAULT_MAX_REL_MISFIT
    flag_a = np.asarray(flag, dtype=int)
    rel_a = np.asarray(rel_misfit, dtype=float)
    tier = np.where(flag_a != FLAG_OK, TIER_REJECT,
                    np.where(np.isfinite(rel_a) & (rel_a <= float(max_rel_misfit)),
                             TIER_QUANTITATIVE, TIER_POSITION))
    return tier if tier.ndim else str(tier[()])


_GAUSS_C = 4.0 * np.log(2.0)          # exp(-_GAUSS_C * (dx/fwhm)^2) has the given FWHM
_GAUSS_AREA = np.sqrt(np.pi / _GAUSS_C)   # integral of unit-height gaussian / fwhm
_LN2 = np.log(2.0)                    # gaussian exponent factor (see pseudo_voigt)


# ---------------------------------------------------------------------------
# Profile model (vectorized)
# ---------------------------------------------------------------------------

def pseudo_voigt(x, center, amplitude, fwhm, eta) -> np.ndarray:
    """Pseudo-Voigt profile ``A * (eta*L + (1-eta)*G)`` with peak height ``A``.

    All of ``center, amplitude, fwhm, eta`` may be scalars or broadcastable
    arrays; with ``x`` of shape (N,) and the parameters of shape (K,) passing
    ``center[None, :]`` etc. yields an (N, K) stack of profiles.
    """
    x = np.asarray(x, float)
    w = np.asarray(fwhm, float)
    dx = x - np.asarray(center, float)
    z = (2.0 * dx / w) ** 2                       # = 4 dx^2 / fwhm^2
    lor = 1.0 / (1.0 + z)
    gau = np.exp(-np.log(2.0) * z)                # exp(-4 ln2 dx^2 / fwhm^2)
    eta = np.asarray(eta, float)
    return np.asarray(amplitude, float) * (eta * lor + (1.0 - eta) * gau)


def pseudo_voigt_area(amplitude, fwhm, eta) -> np.ndarray:
    """Analytic integral of :func:`pseudo_voigt` over all x."""
    a = np.asarray(amplitude, float)
    w = np.asarray(fwhm, float)
    e = np.asarray(eta, float)
    lor_area = 0.5 * np.pi * w          # integral of unit-height lorentzian
    gau_area = _GAUSS_AREA * w          # integral of unit-height gaussian
    return a * (e * lor_area + (1.0 - e) * gau_area)


def pseudo_voigt_jac(x, center, amplitude, fwhm, eta
                     ) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Partial derivatives of :func:`pseudo_voigt`, in the order
    ``(d/dcenter, d/damplitude, d/dfwhm, d/deta)``.

    Each output has the broadcast shape of the inputs (so ``x[:, None]`` with
    ``(K,)`` parameters yields (N, K) derivative stacks). This is the closed-form
    least-squares Jacobian used by :func:`_fit_group`; supplying it to
    ``scipy.optimize.least_squares`` replaces the finite-difference Jacobian,
    whose ``(4K+2)`` extra model evaluations per iteration otherwise dominate the
    whole Step-2 runtime on many-peak patterns.
    """
    x = np.asarray(x, float)
    c = np.asarray(center, float)
    a = np.asarray(amplitude, float)
    w = np.asarray(fwhm, float)
    e = np.asarray(eta, float)
    dx = x - c
    inv_w2 = 1.0 / (w * w)
    z = 4.0 * dx * dx * inv_w2
    lor = 1.0 / (1.0 + z)
    gau = np.exp(-_LN2 * z)
    # d/dcenter of each component; d/dfwhm is d/dcenter times (dx/w) — both carry
    # the same z chain-rule factor, since z depends on center and fwhm only
    # through dx/w.
    dlor_dc = 8.0 * dx * lor * lor * inv_w2
    dgau_dc = 8.0 * _LN2 * dx * gau * inv_w2
    d_center = a * (e * dlor_dc + (1.0 - e) * dgau_dc)
    d_amp = e * lor + (1.0 - e) * gau
    d_fwhm = d_center * (dx / w)
    d_eta = a * (lor - gau)
    return d_center, d_amp, d_fwhm, d_eta


# ---------------------------------------------------------------------------
# Noise floor & peak detection
# ---------------------------------------------------------------------------

def mad_sigma(y) -> float:
    """Robust noise estimate: 1.4826 * MAD, computed on finite values only.

    Robust to the peaks themselves because the median ignores the (sparse)
    high-intensity bins.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return float(1.4826 * mad)


def _half_max_width(x, y, k) -> float:
    """Estimate FWHM at local-max index ``k`` from the half-height crossings."""
    n = y.size
    half = 0.5 * y[k]
    li = k
    while li > 0 and y[li] > half:
        li -= 1
    ri = k
    while ri < n - 1 and y[ri] > half:
        ri += 1
    w = abs(x[ri] - x[li])
    if not np.isfinite(w) or w <= 0:
        # fall back to a couple of bins' worth
        dx = np.median(np.abs(np.diff(x))) if n > 1 else 1.0
        w = 3.0 * float(dx)
    return float(w)


def detect_peaks(x, y, *, min_snr: float = 5.0,
                 min_prominence_snr: Optional[float] = None,
                 sigma: Optional[float] = None,
                 seed_centers: Optional[Sequence[float]] = None,
                 edge_bins: int = 0,
                 x_min: Optional[float] = None,
                 x_max: Optional[float] = None,
                 ) -> List[Dict[str, float]]:
    """Find candidate peaks and seed initial fit parameters.

    Returns a list of ``{center, amplitude, fwhm}`` dicts (one per candidate),
    sorted by center. Uses ``scipy.signal.find_peaks`` with a height threshold
    AND a prominence threshold, both in units of the MAD noise floor ``σ``:

    * ``min_snr·σ`` — minimum height (a peak must clear the noise).
    * ``min_prominence_snr·σ`` — minimum prominence. Decoupled from height
      because prominence is measured against the *taller* neighbour: a real peak
      sitting on the shoulder of a stronger one has low prominence and was being
      rejected even though its height was fine. Defaults to ``min_snr`` (the old
      coupled behaviour) when not given; set it lower to keep shoulder/adjacent
      peaks.

    Candidates (including re-added seeds) are restricted to a valid window:
    ``edge_bins`` excludes peaks within that many bins of either array end
    (kills beamstop-onset and detector-truncation artefacts), and
    ``x_min``/``x_max`` (in the radial unit) clip to a physically valid range.

    If ``seed_centers`` is given, seeds with no detection are added back so peak
    identity survives across a frame series.
    """
    from scipy.signal import find_peaks  # lazy

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yf = np.where(np.isfinite(y), y, 0.0)
    sig = float(sigma) if sigma is not None else mad_sigma(y)
    sig = sig if sig > 0 else (np.nanstd(y) or 1.0)
    prom_snr = min_snr if min_prominence_snr is None else float(min_prominence_snr)
    height = min_snr * sig
    prominence = max(prom_snr * sig, 1e-12)
    idx, _ = find_peaks(yf, height=height, prominence=prominence)

    # Valid x-window: edge-bin guard (order-agnostic) ∩ [x_min, x_max].
    n = x.size
    eb = max(int(edge_bins), 0)
    if n and eb:
        lo_i, hi_i = min(eb, n - 1), max(n - 1 - eb, 0)
        w_lo, w_hi = min(x[lo_i], x[hi_i]), max(x[lo_i], x[hi_i])
    elif n:
        w_lo, w_hi = float(np.min(x)), float(np.max(x))
    else:
        w_lo, w_hi = -np.inf, np.inf
    if x_min is not None:
        w_lo = max(w_lo, float(x_min))
    if x_max is not None:
        w_hi = min(w_hi, float(x_max))

    cands: List[Dict[str, float]] = []
    for k in idx:
        cands.append({"center": float(x[k]), "amplitude": float(yf[k]),
                      "fwhm": _half_max_width(x, yf, int(k))})

    if seed_centers is not None and len(seed_centers):
        seeds = np.sort(np.asarray(seed_centers, float))
        got = np.array([c["center"] for c in cands], float)
        # Re-add only seeds with no detection within ~half a peak width — a peak
        # that merely drifted between frames is already detected here; this
        # recovers reflections that dipped below SNR, without duplicating them.
        dx = np.median(np.abs(np.diff(x))) if x.size > 1 else 1.0
        typ_w = np.median([c["fwhm"] for c in cands]) if cands else 3.0 * float(dx)
        tol = max(3.0 * float(dx), 0.5 * float(typ_w))
        for s in seeds:
            if got.size == 0 or np.min(np.abs(got - s)) > tol:
                k = int(np.argmin(np.abs(x - s)))
                if np.isfinite(yf[k]):
                    cands.append({"center": float(x[k]),
                                  "amplitude": float(max(yf[k], height)),
                                  "fwhm": _half_max_width(x, yf, k)})
    cands = [c for c in cands if w_lo <= c["center"] <= w_hi]
    cands.sort(key=lambda c: c["center"])
    return cands


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# Joint fits scale ~O(K^2) in peaks per group; on a noisy detection a chain of
# marginal candidates can link the WHOLE pattern into one group and the fit
# effectively hangs. Groups larger than this are split at their widest
# internal center gaps (the physically most separable seams).
MAX_GROUP_SIZE = 12


def _split_group(group: "List[Dict[str, float]]", max_size: int
                 ) -> "List[List[Dict[str, float]]]":
    """Recursively split an oversized group at its largest center gap."""
    if len(group) <= max_size:
        return [group]
    gaps = [group[i + 1]["center"] - group[i]["center"]
            for i in range(len(group) - 1)]
    cut = int(np.argmax(gaps)) + 1
    if cut == 0 or cut == len(group):          # degenerate; hard split
        cut = len(group) // 2
    return (_split_group(group[:cut], max_size)
            + _split_group(group[cut:], max_size))


def _group_peaks(cands: List[Dict[str, float]], window_factor: float,
                 max_group_size: int = MAX_GROUP_SIZE
                 ) -> List[List[Dict[str, float]]]:
    """Cluster peaks whose +-window_factor*fwhm windows overlap, for joint fit.

    Groups exceeding ``max_group_size`` are split at their widest internal
    gaps — a bounded joint fit of the most-overlapped neighbours beats an
    unbounded one that never converges.
    """
    groups: List[List[Dict[str, float]]] = []
    cur: List[Dict[str, float]] = []
    cur_hi = -np.inf
    for c in cands:
        lo = c["center"] - window_factor * c["fwhm"]
        hi = c["center"] + window_factor * c["fwhm"]
        if cur and lo <= cur_hi:
            cur.append(c)
            cur_hi = max(cur_hi, hi)
        else:
            if cur:
                groups.append(cur)
            cur = [c]
            cur_hi = hi
    if cur:
        groups.append(cur)
    out: List[List[Dict[str, float]]] = []
    for g in groups:
        out.extend(_split_group(g, max(2, int(max_group_size))))
    return out


def _fit_group(x, y, group, sigma, *, window_factor: float, max_chi2: float,
               max_rel_misfit: float = DEFAULT_MAX_REL_MISFIT
               ) -> List[Dict[str, Any]]:
    """Jointly fit one cluster of peaks (sum of pseudo-Voigts + LOCAL LINEAR
    baseline).

    The baseline under each group is fitted as intercept + slope rather than a
    constant: real residual background (whatever the global SNIP left) is locally
    sloped, and — per Girgsdies — an ambiguous/biased baseline corrupts the peak
    height, FWHM and area, and a flat fit on a slope leaves a sloped residual that
    inflates chi-square (spurious bad_chi2 rejections). A local slope absorbs that.
    """
    from scipy.optimize import least_squares  # lazy

    centers = np.array([g["center"] for g in group], float)
    amps = np.array([max(g["amplitude"], 1e-9) for g in group], float)
    widths = np.array([max(g["fwhm"], 1e-9) for g in group], float)
    lo = float(centers.min() - window_factor * widths.max())
    hi = float(centers.max() + window_factor * widths.max())
    m = (x >= lo) & (x <= hi) & np.isfinite(y)
    xw, yw = x[m], y[m]
    K = len(group)
    if xw.size < 4 * K + 2:                        # too few points to fit
        return [_failed_peak(g, FLAG_NO_CONVERGE) for g in group]
    xc = float(xw.mean())                          # centre x for slope conditioning

    # parameter vector: [c,a,w,eta]*K + [b0 (intercept), b1 (slope)]
    p0, lb, ub = [], [], []
    for c, a, w in zip(centers, amps, widths):
        p0 += [c, a, w, 0.5]
        lb += [c - 0.5 * w, 0.0, 0.2 * w, 0.0]
        ub += [c + 0.5 * w, 5.0 * a + 1e-9, 5.0 * w, 1.0]
    b0_init = float(np.percentile(yw, 10))         # local floor as the intercept seed
    p0 += [b0_init, 0.0]
    lb += [-np.inf, -np.inf]
    ub += [np.inf, np.inf]
    p0 = np.array(p0); lb = np.array(lb); ub = np.array(ub)
    p0 = np.clip(p0, lb, ub)
    sig = sigma if sigma > 0 else 1.0

    xrel = xw - xc                                 # baseline-slope regressor, cached

    def model(p):
        c = p[0:4 * K:4]; a = p[1:4 * K:4]; w = p[2:4 * K:4]; e = p[3:4 * K:4]
        prof = pseudo_voigt(xw[:, None], c[None, :], a[None, :], w[None, :], e[None, :])
        return prof.sum(axis=1) + p[-2] + p[-1] * xrel        # peaks + linear baseline

    def resid(p):
        return (model(p) - yw) / sig

    # Analytic Jacobian of ``resid``. Without it, ``least_squares`` estimates the
    # (4K+2)-column Jacobian by finite differences — (4K+2) extra ``model`` evals
    # per iteration — which dominates the whole Step-2 runtime on many-peak
    # patterns (the "peak fitting takes forever" cost, seed propagation included,
    # since propagation only changes WHICH peaks are fitted, not this per-fit
    # cost). The closed form is exact (a pseudo-Voigt is smooth in every
    # parameter), so it also gives cleaner covariance esd's than the FD estimate.
    def jac(p):
        c = p[0:4 * K:4]; a = p[1:4 * K:4]; w = p[2:4 * K:4]; e = p[3:4 * K:4]
        dc, da, dw, de = pseudo_voigt_jac(
            xw[:, None], c[None, :], a[None, :], w[None, :], e[None, :])
        J = np.empty((xw.size, 4 * K + 2))
        J[:, 0:4 * K:4] = dc                                   # d/dcenter
        J[:, 1:4 * K:4] = da                                   # d/damplitude
        J[:, 2:4 * K:4] = dw                                   # d/dfwhm
        J[:, 3:4 * K:4] = de                                   # d/deta
        J[:, -2] = 1.0                                          # d/db0 (intercept)
        J[:, -1] = xrel                                         # d/db1 (slope)
        return J / sig

    try:
        sol = least_squares(resid, p0, jac=jac, bounds=(lb, ub), method="trf",
                            max_nfev=200 * (K + 1))
        converged = bool(sol.success)
        p = sol.x
    except Exception:
        return [_failed_peak(g, FLAG_NO_CONVERGE) for g in group]

    dof = max(xw.size - p.size, 1)
    chi2 = float(np.sum(resid(p) ** 2) / dof)

    # 1σ parameter uncertainties from the Gauss-Newton covariance
    # (JᵀJ)⁻¹·χ²_red — the residuals are already scaled by the noise estimate,
    # so the reduced chi-square recalibrates any mis-scaled σ. pinv tolerates
    # the singular Jacobian a bound-pinned parameter produces (its esd is then
    # meaningless anyway; the WIDTH_BOUND/CENTER_DRIFT flags mark those peaks).
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * chi2
        esd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        esd = np.full(p.size, np.nan)

    # Fit quality is judged PER PEAK, over that peak's own span, on two measures
    # (see DEFAULT_MAX_REL_MISFIT for why one is not enough):
    #
    #   chi2_local  the residual in units of the measurement noise floor, which
    #               is the meaningful test for a weak, noise-limited peak;
    #   rel_misfit  the rms residual as a fraction of THAT peak's own height,
    #               the meaningful test for a bright, systematics-limited one.
    #
    # Both are local because the group answer is a different claim. A joint
    # model can fit a region well overall and still contain one badly modelled
    # weak component, and judging every member by the tallest peak's height is
    # only ever more lenient (a_max >= a_j): measured on a 1288-frame series,
    # 121 peaks were excused by a bright neighbour while being bad on their own,
    # and none were condemned by one. The group's own reduced chi-square is
    # still reported, as ``chi2``, so group adequacy stays inspectable — it is
    # deliberately not a rejection, because making it one brings back exactly
    # the "one bad peak condemns its neighbours" behaviour this replaced.
    #
    # The residual is taken from the FULL group model, so a neighbour's
    # intensity inside the window is accounted for rather than counted as
    # misfit.
    #
    # Each peak is scored on the points it OWNS — inside its span and where its
    # own profile is the largest of the group's. Without that, neighbours' spans
    # overlap, a badly modelled component leaks into the peak beside it, and the
    # "condemned by a neighbour" failure comes straight back at a local scale.
    resid_all = model(p) - yw
    pc_j = p[0:4 * K:4]; pa_j = p[1:4 * K:4]
    pw_j = p[2:4 * K:4]; pe_j = p[3:4 * K:4]
    prof = pseudo_voigt(xw[:, None], pc_j[None, :], pa_j[None, :],
                        pw_j[None, :], pe_j[None, :])
    owner = np.argmax(prof, axis=1) if K > 1 else np.zeros(xw.size, int)

    out: List[Dict[str, Any]] = []
    for j, g in enumerate(group):
        c, a, w, e = p[4 * j:4 * j + 4]
        c_err, a_err, w_err = esd[4 * j], esd[4 * j + 1], esd[4 * j + 2]

        span = QUALITY_WINDOW_FWHM * max(abs(w), 1e-9)
        loc = (np.abs(xw - c) <= span) & (owner == j)
        if int(loc.sum()) < 5:                 # too few to judge on its own
            loc = np.abs(xw - c) <= span
        if int(loc.sum()) < 5:
            loc = np.ones_like(xw, dtype=bool)
        rj = resid_all[loc]
        # Scaled residual sum, not a strict reduced chi-square: the parameters
        # were fitted over the whole group window and the baseline pair is
        # shared, so subtracting this peak's own 4 is an approximation. It is
        # read against a calibrated threshold, never used as a p-value.
        dof_j = max(int(loc.sum()) - 4, 1)
        chi2_local = float(np.sum((rj / sig) ** 2) / dof_j)
        rms_j = float(np.sqrt(np.mean(rj ** 2)))
        rel_misfit = rms_j / a if a > 0 else np.inf
        poor_fit = (chi2_local > max_chi2) and (rel_misfit > float(max_rel_misfit))

        flag = FLAG_OK if converged else FLAG_NO_CONVERGE
        if a < 2.0 * sig:
            flag |= FLAG_LOW_AMP
        if poor_fit:
            flag |= FLAG_BAD_CHI2
        # The fit bounds already confine the center to seed ± 0.5·FWHM, so "moved
        # more than a FWHM" can never happen — a runaway center shows up as the
        # optimizer pinning c against its bound (same detection as width).
        if abs(c - g["center"]) >= 0.5 * max(g["fwhm"], 1e-9) * 0.999:
            flag |= FLAG_CENTER_DRIFT
        if w <= 0.2 * g["fwhm"] * 1.001 or w >= 5.0 * g["fwhm"] * 0.999:
            flag |= FLAG_WIDTH_BOUND
        out.append({
            "center": float(c), "amplitude": float(a), "fwhm": float(abs(w)),
            "eta": float(np.clip(e, 0.0, 1.0)),
            "area": float(pseudo_voigt_area(a, abs(w), np.clip(e, 0.0, 1.0))),
            "chi2": chi2, "chi2_local": chi2_local,
            "rel_misfit": float(rel_misfit), "flag": int(flag),
            "center_err": float(c_err), "amplitude_err": float(a_err),
            "fwhm_err": float(w_err),
        })
    return out


def _failed_peak(g: Dict[str, float], flag: int) -> Dict[str, Any]:
    return {"center": float(g["center"]), "amplitude": float(g["amplitude"]),
            "fwhm": float(g["fwhm"]), "eta": 0.5, "area": 0.0,
            "chi2": np.inf, "chi2_local": np.inf, "rel_misfit": np.inf,
            "flag": int(flag),
            "center_err": np.nan, "amplitude_err": np.nan, "fwhm_err": np.nan}


def _window_mask(x: np.ndarray, edge_bins: int,
                 fit_min: "Optional[float]", fit_max: "Optional[float]") -> np.ndarray:
    """Boolean mask of the valid fit window: edge-bin guard ∩ [fit_min, fit_max]."""
    n = x.size
    m = np.ones(n, dtype=bool)
    eb = max(int(edge_bins), 0)
    if n and eb:
        m[:min(eb, n)] = False
        m[max(n - eb, 0):] = False
    if fit_min is not None:
        m &= x >= float(fit_min)
    if fit_max is not None:
        m &= x <= float(fit_max)
    return m


def fit_pattern(x, y, *, min_snr: float = 5.0, window_factor: float = 3.0,
                max_chi2: float = 25.0,
                max_rel_misfit: float = DEFAULT_MAX_REL_MISFIT,
                min_prominence_snr: Optional[float] = None,
                edge_bins: int = 0, fit_min: Optional[float] = None,
                fit_max: Optional[float] = None, min_fwhm_bins: float = 0.0,
                local_baseline_bins: int = 0,
                sigma: Optional[float] = None,
                seed_centers: Optional[Sequence[float]] = None,
                keep_flagged: bool = True) -> List[Dict[str, Any]]:
    """Detect and fit every peak in one 1D pattern.

    Returns a list of peak dicts (``center, amplitude, fwhm, eta, area, chi2,
    flag``) sorted by center. If ``keep_flagged`` is False, peaks whose flag is
    nonzero are dropped from the result.

    The valid fit window (``edge_bins`` end-guard ∩ ``[fit_min, fit_max]``) keeps
    detection off the beamstop onset and the detector-truncation tails.
    ``min_fwhm_bins`` flags peaks narrower than that many bins
    (``FLAG_WIDTH_BOUND``) to reject single-bin quantization spikes.

    ``local_baseline_bins`` > 0 enables **local detrending for detection**: a
    morphological opening of width that many bins estimates the residual broad
    background (whatever SNIP left behind), detection runs on ``clean − that``,
    and the noise floor is the residual's MAD. Without it, a ``clean`` that isn't
    flat inflates the global σ so badly that real peaks fall under the height
    threshold (the "missing all peaks" failure). The fit itself still uses the
    original intensities. Recommended; sized larger than a peak (a few × FWHM in
    bins) but smaller than the background structure.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = _window_mask(x, edge_bins, fit_min, fit_max)

    # Detection signal: optionally detrended so the threshold tracks real noise,
    # not residual background. Fitting always uses the original y.
    det = y
    if local_baseline_bins and int(local_baseline_bins) > 1:
        from scipy.ndimage import grey_opening  # lazy
        W = int(local_baseline_bins) | 1                      # force odd
        base = grey_opening(np.nan_to_num(y, nan=0.0), size=W)
        det = y - base

    if sigma is not None:
        sig = float(sigma)
    else:
        sig = mad_sigma(det[mask]) if mask.any() else mad_sigma(det)
        if not (sig > 0):
            sig = mad_sigma(det) or (np.nanstd(det) or 1.0)
    cands = detect_peaks(x, det, min_snr=min_snr, min_prominence_snr=min_prominence_snr,
                         sigma=sig, seed_centers=seed_centers,
                         edge_bins=edge_bins, x_min=fit_min, x_max=fit_max)
    peaks: List[Dict[str, Any]] = []
    for group in _group_peaks(cands, window_factor):
        peaks.extend(_fit_group(x, y, group, sig, window_factor=window_factor,
                                max_chi2=max_chi2,
                                max_rel_misfit=max_rel_misfit))
    # Sub-resolution rejection: a real Bragg peak spans several bins.
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    min_fwhm = float(min_fwhm_bins) * dx
    if min_fwhm > 0:
        for p in peaks:
            if p["fwhm"] < min_fwhm:
                p["flag"] = int(p["flag"]) | FLAG_WIDTH_BOUND
    peaks.sort(key=lambda p: p["center"])
    if not keep_flagged:
        peaks = [p for p in peaks if p["flag"] == FLAG_OK]
    return peaks


def fit_dataset(radial, clean, *, min_snr: float = 5.0, window_factor: float = 3.0,
                max_chi2: float = 25.0,
                max_rel_misfit: float = DEFAULT_MAX_REL_MISFIT,
                min_prominence_snr: Optional[float] = None,
                edge_bins: int = 0, fit_min: Optional[float] = None,
                fit_max: Optional[float] = None, min_fwhm_bins: float = 0.0,
                local_baseline_bins: int = 0,
                propagate_seeds: bool = True,
                keep_flagged: bool = True) -> List[List[Dict[str, Any]]]:
    """Fit every frame of a (N_frames, N_bins) ``clean`` stack.

    With ``propagate_seeds`` the good centers of frame *i* seed detection for
    frame *i+1*, so a reflection keeps its identity as it drifts across the
    series (helps the Step-3 heatmap). Returns a list (per frame) of peak lists.
    """
    radial = np.asarray(radial, float)
    clean = np.asarray(clean, float)
    results: List[List[Dict[str, Any]]] = []
    seeds: Optional[List[float]] = None
    for i in range(clean.shape[0]):
        peaks = fit_pattern(radial, clean[i], min_snr=min_snr,
                            window_factor=window_factor, max_chi2=max_chi2,
                            max_rel_misfit=max_rel_misfit,
                            min_prominence_snr=min_prominence_snr,
                            edge_bins=edge_bins, fit_min=fit_min, fit_max=fit_max,
                            min_fwhm_bins=min_fwhm_bins,
                            local_baseline_bins=local_baseline_bins,
                            seed_centers=seeds, keep_flagged=keep_flagged)
        results.append(peaks)
        if propagate_seeds:
            good = [p["center"] for p in peaks if p["flag"] == FLAG_OK]
            seeds = good if good else seeds
    return results


# ---------------------------------------------------------------------------
# Fit-source selection, sensitivity presets, auto range (driver-level helpers)
# ---------------------------------------------------------------------------

# Detection-knob presets. ``window_factor``, ``max_chi2`` and the detrend window
# are deliberately NOT part of this — they are structural, not sensitivity. A
# preset value is used only for a knob the caller leaves unset (None); an
# explicit value always wins.
SENSITIVITY_PRESETS: Dict[str, Dict[str, float]] = {
    "conservative": {"min_snr": 6.0, "min_prominence_snr": 3.0, "min_fwhm_bins": 3.0, "edge_bins": 6.0},
    "normal":       {"min_snr": 5.0, "min_prominence_snr": 2.0, "min_fwhm_bins": 2.0, "edge_bins": 5.0},
    "sensitive":    {"min_snr": 3.5, "min_prominence_snr": 1.5, "min_fwhm_bins": 2.0, "edge_bins": 4.0},
}

# Selectable peak-fitting source (see :func:`build_fit_source`).
FIT_SOURCES = ("auto", "clean", "hybrid", "mean", "sigmaclip", "spots")


def resolve_sensitivity(sensitivity: "Optional[str]" = "normal", *,
                        min_snr: "Optional[float]" = None,
                        min_prominence_snr: "Optional[float]" = None,
                        min_fwhm_bins: "Optional[float]" = None,
                        edge_bins: "Optional[int]" = None) -> Dict[str, Any]:
    """Concrete detection knobs from a named preset, each overridden by any
    explicitly-given value. Unknown preset name → "normal"."""
    name = (sensitivity or "normal").strip().lower()
    base = SENSITIVITY_PRESETS.get(name, SENSITIVITY_PRESETS["normal"])
    return {
        "min_snr": float(base["min_snr"] if min_snr is None else min_snr),
        "min_prominence_snr": float(
            base["min_prominence_snr"] if min_prominence_snr is None else min_prominence_snr),
        "min_fwhm_bins": float(base["min_fwhm_bins"] if min_fwhm_bins is None else min_fwhm_bins),
        "edge_bins": int(round(base["edge_bins"] if edge_bins is None else edge_bins)),
        "preset": name if name in SENSITIVITY_PRESETS else "normal",
    }


def winsorize_excess(spot_residual, spike_bins: int = 5) -> np.ndarray:
    """The positive azimuthal-mean excess to ADD back to ``clean`` for a hybrid
    (winsorized-mean) fit source.

    ``spot_residual = mean − robust`` is the intensity the spot-suppressed median
    dropped. It mixes two things we must separate using the only discriminator a
    1-D pattern still carries — the **radial width**:

      * a diamond single-crystal reflection is *narrow* (≈instrumental), a few
        bins wide — we reject it (fall back to the median there);
      * a real but azimuthally *sparse* sample ring (texture / spotty powder /
        incomplete Debye ring) is as wide as the Bragg peak — we keep it.

    A morphological grey-opening of width ``spike_bins`` does exactly this:
    positive features narrower than the structuring element are eroded to the
    surrounding floor (diamond spikes → 0) while broader excess keeps its core
    (so a real peak is never clipped at the tip, unlike a percentile cap).
    Negative excess is noise → floored to 0; NaNs contribute 0. Accepts a 1-D
    pattern or a 2-D (N_frames, N_bins) stack (opened along the radial axis).

    Note this is only an *approximation* of a true trimmed mean; the principled
    channel is the reduce-side ``sigmaclip`` (``source="sigmaclip"``), which
    rejects spots using the real azimuthal spread per bin.
    """
    from scipy.ndimage import grey_opening  # lazy
    sr = np.asarray(spot_residual, float)
    pos = np.where(np.isfinite(sr), np.clip(sr, 0.0, None), 0.0)
    W = max(int(spike_bins), 1) | 1                              # force odd ≥ 1
    if W <= 1:
        return pos
    if sr.ndim == 1:
        return grey_opening(pos, size=W)
    return grey_opening(pos, size=(1, W))                        # per-row, radial axis


def build_fit_source(source: str, clean, *, spot_residual=None,
                     sigmaclip_residual=None, hybrid_spike_bins: int = 5
                     ) -> "Tuple[np.ndarray, str]":
    """Build the intensity to fit from the Step-1 channels; return ``(data,
    resolved_source)``.

    Every source except ``spots`` is ``clean`` plus an already-baseline-subtracted
    residual, because ``clean = robust − baseline`` and the smooth background is
    azimuthally uniform (the same baseline applies to every channel):

        clean      robust − baseline                          (conservative)
        mean       clean + spot_residual      = mean − baseline
        hybrid     clean + winsorized(spot_residual)          (analysis-side default)
        sigmaclip  clean + sigmaclip_residual  = sigmaclip − baseline (reduce-side)
        spots      spot_residual alone — the SINGLE-CRYSTAL SAMPLE channel
        auto       sigmaclip if its residual is present, else hybrid

    ``spots`` is for the case the powder pipeline is otherwise blind to: a
    single-crystal sample. Its reflections are azimuthally sparse blobs, so the
    median-based channels reject them exactly like diamond spots and the mean
    dilutes them ~N_azimuth-fold. ``spot_residual = mean − robust`` is where that
    intensity ends up, and it is already background-free (the smooth baseline is
    azimuthally uniform, so it cancels in the subtraction). Fitting it directly
    surfaces the sample reflections (plus spotty-ring excess of coarse powder
    phases, which Step 3a then attributes and removes) so their d(P) tracks reach
    the Step-3c unknowns.
    """
    clean = np.asarray(clean, float)
    s = (source or "auto").strip().lower()
    if s == "auto":
        if sigmaclip_residual is not None:
            s = "sigmaclip"
        elif spot_residual is not None:
            s = "hybrid"
        else:
            s = "clean"
    if s == "clean":
        return clean, s
    if s == "mean":
        if spot_residual is None:
            raise ValueError("source='mean' needs spot_residual.")
        return clean + np.asarray(spot_residual, float), s
    if s == "hybrid":
        if spot_residual is None:
            raise ValueError("source='hybrid' needs spot_residual.")
        return clean + winsorize_excess(spot_residual, hybrid_spike_bins), s
    if s == "sigmaclip":
        if sigmaclip_residual is None:
            raise ValueError(
                "source='sigmaclip' needs /background/sigmaclip_residual — re-run "
                "reduction with the sigma-clip channel on, then Step 1.")
        return clean + np.asarray(sigmaclip_residual, float), s
    if s == "spots":
        if spot_residual is None:
            raise ValueError("source='spots' needs /background/spot_residual.")
        return np.asarray(spot_residual, float), s
    raise ValueError(f"Unknown fit source {source!r} (choose from {FIT_SOURCES}).")


def auto_fit_range(radial, signal, *, max_trim_frac: float = 0.15,
                   noise_k: float = 4.0) -> "Tuple[Optional[float], Optional[float]]":
    """Conservatively infer the valid ``(fit_min, fit_max)`` in the radial unit.

    Trims only what is unambiguously not sample signal, and never more than
    ``max_trim_frac`` of the axis at either end (interior peaks stay safe):

      * low end  — leading non-finite bins and the monotonically *descending*
        beamstop-onset ramp (stops at the first turn-up = first real feature);
      * high end — the trailing dead/flat tail whose intensity never clears
        ``noise_k`` × MAD noise floor (detector truncation / noisy tail).

    Returns ``(lo, hi)``; an end that needs no trimming is ``None`` (= use the
    full range there). ``signal`` should be a representative 1-D pattern, e.g.
    the per-bin median across frames of the chosen fit source.
    """
    x = np.asarray(radial, float)
    y = np.asarray(signal, float)
    n = x.size
    if n < 8 or x.shape != y.shape:
        return None, None
    order = np.argsort(x)
    x, y = x[order], y[order]
    finite = np.isfinite(y)
    if not finite.any():
        return None, None
    i0 = int(np.argmax(finite))                      # first finite bin
    i1 = int(n - 1 - np.argmax(finite[::-1]))        # last finite bin
    cap_lo = i0 + int(max_trim_frac * n)
    cap_hi = i1 - int(max_trim_frac * n)

    # Threshold decisions run on a light smoothing of a NaN-free copy, so
    # point-to-point noise can't halt the descent early or look like a peak in
    # the tail. The returned bounds are still real radial values.
    from scipy.ndimage import uniform_filter1d  # lazy
    idx = np.arange(n)
    yfill = y if finite.all() else np.interp(idx, idx[finite], y[finite])
    ys = uniform_filter1d(yfill, size=max(3, n // 200) | 1)

    # Low: descend the leading beamstop ramp to its first (smoothed) local min.
    lo_i = i0
    while lo_i < cap_lo and ys[lo_i + 1] < ys[lo_i]:
        lo_i += 1

    # High: trim a trailing run sitting at the background within the noise. The
    # local background is the trailing region's median and the noise its
    # first-difference MAD — immune to a smooth slope (diff of a ramp is
    # ~constant) and to peaks — so a real signal tail isn't mistaken for a dead
    # one. The retreat stops at the first bin above floor (a real peak), keeping it.
    tail = ys[max(i1 - max(int(0.2 * n), 8), i0): i1 + 1]
    sig = mad_sigma(np.diff(tail)) / np.sqrt(2.0) if tail.size >= 3 else 0.0
    floor = float(np.median(tail)) + noise_k * sig
    hi_i = i1
    while hi_i > cap_hi and ys[hi_i] < floor:
        hi_i -= 1

    lo = float(x[lo_i]) if lo_i > i0 or i0 > 0 else None
    hi = float(x[hi_i]) if hi_i < i1 or i1 < n - 1 else None
    return lo, hi


# ---------------------------------------------------------------------------
# Dataset driver (analysis HDF5 -> peaks appended)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1"
_PEAK_COLS = ("center", "amplitude", "fwhm", "eta", "area", "chi2",
              "chi2_local", "rel_misfit", "flag",
              "center_err", "amplitude_err", "fwhm_err")
SEED_TRACKING_AXES = ("frame", "pressure", "temperature", "time")
SEED_GROUPS = ("none", "scan", "folder")


def _normalize_seed_axis(axis: "Optional[str]") -> str:
    key = (axis or "frame").strip().lower()
    if key in ("", "index", "same", "unknown", "unknowns"):
        key = "frame"
    if key not in SEED_TRACKING_AXES:
        raise ValueError(f"Unknown peak-seed tracking axis {axis!r} "
                         f"(choose from {', '.join(SEED_TRACKING_AXES)}).")
    return key


def _normalize_seed_group(group_by: "Optional[str]") -> str:
    key = (group_by or "none").strip().lower()
    if key in ("", "all", "same", "unknown", "unknowns"):
        key = "none"
    if key not in SEED_GROUPS:
        raise ValueError(f"Unknown peak-seed group {group_by!r} "
                         f"(choose from {', '.join(SEED_GROUPS)}).")
    return key


def _seed_frame_orders(n_frames: int, axis_values: np.ndarray, axis_key: str,
                       group_values: np.ndarray) -> List[np.ndarray]:
    """Frame orders used for seed propagation.

    Each returned order is one independent propagation path. Physical axes are
    sorted within their scan/folder group; frames with missing physical metadata
    are still fitted, but as one-frame paths so stale seeds are not carried into
    unknown conditions.
    """
    n = int(n_frames)
    if n <= 0:
        return []
    axis = np.asarray(axis_values, dtype=float)
    groups = np.asarray(group_values)
    if axis.size != n or groups.size != n:
        raise ValueError("Seed tracking metadata length must match frame count.")

    group_order: List[Any] = []
    seen = set()
    for raw in groups.tolist():
        if raw not in seen:
            seen.add(raw)
            group_order.append(raw)

    orders: List[np.ndarray] = []
    for group_id in group_order:
        frames = np.where(groups == group_id)[0].astype(int)
        if not frames.size:
            continue
        if axis_key == "frame":
            orders.append(frames)
            continue
        finite = frames[np.isfinite(axis[frames])]
        if finite.size:
            ordered = np.asarray(
                sorted((int(f) for f in finite), key=lambda f: (float(axis[f]), f)),
                dtype=int,
            )
            orders.append(ordered)
        for f in frames[~np.isfinite(axis[frames])]:
            orders.append(np.asarray([int(f)], dtype=int))
    return orders


def _predict_seed_centers(prev_good, prev2_good, axis_now: float,
                          use_axis_predictor: bool) -> "Optional[List[float]]":
    if prev_good is None:
        return None
    axis_prev, centers_prev = prev_good
    centers = np.asarray(centers_prev, dtype=float)
    if not centers.size:
        return None
    if (not use_axis_predictor or prev2_good is None
            or len(prev2_good[1]) != centers.size):
        return centers.tolist()
    axis_prev2, centers_prev2 = prev2_good
    if not (np.isfinite(axis_now) and np.isfinite(axis_prev)
            and np.isfinite(axis_prev2)):
        return centers.tolist()
    da = float(axis_prev) - float(axis_prev2)
    if abs(da) <= 1e-12:
        return centers.tolist()
    c0 = np.sort(np.asarray(centers_prev2, dtype=float))
    c1 = np.sort(centers)
    pred = c1 + (c1 - c0) / da * (float(axis_now) - float(axis_prev))
    pred = pred[np.isfinite(pred)]
    return pred.tolist() if pred.size else centers.tolist()


def _fit_ordered_rows(radial: np.ndarray, clean_rows: np.ndarray,
                      excluded_rows: np.ndarray, row_order: np.ndarray,
                      axis_values: np.ndarray, *,
                      min_snr: float, window_factor: float, max_chi2: float,
                      propagate: bool,
                      max_rel_misfit: float = DEFAULT_MAX_REL_MISFIT,
                      min_prominence_snr: "Optional[float]",
                      edge_bins: int, fit_min: "Optional[float]",
                      fit_max: "Optional[float]", min_fwhm_bins: float,
                      local_baseline_bins: int,
                      seed_max_axis_gap: "Optional[float]",
                      seed_axis_predictor: bool):
    m = clean_rows.shape[0]
    counts = [0] * m
    frames_local: List[int] = []
    cols: Dict[str, list] = {c: [] for c in _PEAK_COLS}
    prev_good = None
    prev2_good = None
    last_axis: "Optional[float]" = None

    for jj in np.asarray(row_order, dtype=int):
        if excluded_rows[jj]:
            continue
        axis_now = float(axis_values[jj]) if axis_values.size else float(jj)
        if (propagate and seed_max_axis_gap is not None and last_axis is not None
                and np.isfinite(axis_now) and np.isfinite(last_axis)
                and abs(axis_now - last_axis) > float(seed_max_axis_gap)):
            prev_good = None
            prev2_good = None
        seeds = (_predict_seed_centers(prev_good, prev2_good, axis_now,
                                       bool(seed_axis_predictor))
                 if propagate else None)
        peaks = fit_pattern(radial, clean_rows[jj], min_snr=min_snr,
                            window_factor=window_factor, max_chi2=max_chi2,
                            max_rel_misfit=max_rel_misfit,
                            min_prominence_snr=min_prominence_snr,
                            edge_bins=edge_bins, fit_min=fit_min, fit_max=fit_max,
                            min_fwhm_bins=min_fwhm_bins,
                            local_baseline_bins=local_baseline_bins,
                            seed_centers=seeds, keep_flagged=True)
        counts[jj] = len(peaks)
        for p in peaks:
            frames_local.append(int(jj))
            for c in _PEAK_COLS:
                cols[c].append(p[c])
        if propagate:
            good = [p["center"] for p in peaks if p["flag"] == FLAG_OK]
            if good:
                prev2_good = prev_good
                prev_good = (axis_now, good)
            last_axis = axis_now
    return counts, frames_local, cols


def _peaks_chunk(payload):
    """Worker: fit a contiguous chunk of frames with internal seed propagation.

    Returns ``(counts, frames_local, cols)`` where ``frames_local`` are indices
    within the chunk (the parent offsets them to global frame indices). Excluded
    frames are skipped (count 0) and do not seed their neighbours.
    """
    (radial, clean_c, excluded_c, min_snr, window_factor, max_chi2,
     max_rel_misfit, propagate, min_prominence_snr, edge_bins, fit_min, fit_max,
     min_fwhm_bins,
     local_baseline_bins, seed_max_axis_gap, seed_axis_predictor) = payload
    m = clean_c.shape[0]
    return _fit_ordered_rows(
        radial, clean_c, excluded_c, np.arange(m, dtype=int),
        np.arange(m, dtype=float),
        min_snr=min_snr, window_factor=window_factor, max_chi2=max_chi2,
        max_rel_misfit=max_rel_misfit,
        propagate=propagate, min_prominence_snr=min_prominence_snr,
        edge_bins=edge_bins, fit_min=fit_min, fit_max=fit_max,
        min_fwhm_bins=min_fwhm_bins, local_baseline_bins=local_baseline_bins,
        seed_max_axis_gap=seed_max_axis_gap,
        seed_axis_predictor=seed_axis_predictor,
    )


def _peaks_order_chunk(payload):
    """Worker: fit one explicit propagation path and return global frame ids."""
    (radial, clean_c, excluded_c, global_order, axis_values,
     min_snr, window_factor, max_chi2, max_rel_misfit, propagate,
     min_prominence_snr,
     edge_bins, fit_min, fit_max, min_fwhm_bins, local_baseline_bins,
     seed_max_axis_gap, seed_axis_predictor) = payload
    m = clean_c.shape[0]
    counts, frames_local, cols = _fit_ordered_rows(
        radial, clean_c, excluded_c, np.arange(m, dtype=int),
        np.asarray(axis_values, dtype=float),
        min_snr=min_snr, window_factor=window_factor, max_chi2=max_chi2,
        max_rel_misfit=max_rel_misfit,
        propagate=propagate, min_prominence_snr=min_prominence_snr,
        edge_bins=edge_bins, fit_min=fit_min, fit_max=fit_max,
        min_fwhm_bins=min_fwhm_bins, local_baseline_bins=local_baseline_bins,
        seed_max_axis_gap=seed_max_axis_gap,
        seed_axis_predictor=seed_axis_predictor,
    )
    order = np.asarray(global_order, dtype=int)
    frames_global = [int(order[j]) for j in frames_local]
    return order, counts, frames_global, cols


def run_peak_fitting(
    analysis_h5: "str | Path",
    out_h5: "Optional[str | Path]" = None,
    *,
    source: str = "auto",
    sensitivity: "Optional[str]" = None,
    auto_range: bool = True,
    hybrid_spike_bins: int = 5,
    min_snr: "Optional[float]" = None,
    window_factor: float = 3.0,
    max_chi2: float = 25.0,
    max_rel_misfit: float = DEFAULT_MAX_REL_MISFIT,
    min_prominence_snr: Optional[float] = None,
    edge_bins: "Optional[int]" = None,
    fit_min: Optional[float] = None,
    fit_max: Optional[float] = None,
    min_fwhm_bins: "Optional[float]" = None,
    local_baseline_bins: int = 0,
    propagate_seeds: bool = True,
    seed_tracking_axis: str = "frame",
    seed_group_by: str = "none",
    seed_max_axis_gap: "Optional[float]" = None,
    seed_axis_predictor: bool = True,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Fit peaks for every frame of a Step-1 analysis HDF5 and store the result.

    Reads ``/radial`` and the Step-1 ``/background`` channels (written by
    ``background.run_background_separation``) and builds the fit signal per
    ``source`` (see :func:`build_fit_source`): the median-based ``clean`` is the
    conservative channel, but ``hybrid``/``sigmaclip`` recover real peaks that
    the azimuthal median suppresses on spotty/textured/incomplete rings. With
    ``source="auto"`` the reduce-side ``sigmaclip`` channel is used when present,
    else the analysis-side ``hybrid``.

    ``sensitivity`` ("conservative"/"normal"/"sensitive") sets the detection
    knobs (min_snr, min_prominence_snr, min_fwhm_bins, edge_bins) for any left
    as ``None``; pass it ``None`` to keep the historical explicit defaults.
    ``auto_range`` fills a blank ``fit_min``/``fit_max`` from
    :func:`auto_fit_range`.

    Fit quality is judged on two tests and a peak has to fail BOTH to be
    flagged: ``max_chi2`` against the background noise, and ``max_rel_misfit``
    — the rms residual as a fraction of the peak's own height. On its own the
    chi-square test is an SNR gate, because the fit weights every point by one
    scalar noise estimate; see :data:`DEFAULT_MAX_REL_MISFIT`.

    Seed propagation defaults to historical frame order. Set
    ``seed_group_by="scan"`` (or ``"folder"``) so seeds propagate only WITHIN a
    scan and never across a scan border — scan identity is parsed from the frame
    filenames, so a reflection from the end of one scan cannot seed the start of
    the next. ``seed_tracking_axis="pressure"`` additionally orders the
    within-group path by pressure instead of frame index. Physical-axis
    propagation can also predict the next seed center from the previous local
    drift, matching the unknown-track logic used later in Step 3c.

    Writes a ``/peaks`` group with a flat, ragged layout so the per-frame peak
    count can vary:

        /peaks/counts      (N_frames,)   peaks fitted per frame
        /peaks/frame       (P,)          frame index of each peak (0..N-1)
        /peaks/center      (P,)          \\
        /peaks/amplitude   (P,)           |
        /peaks/fwhm        (P,)           |  one row per fitted peak,
        /peaks/eta         (P,)           |  grouped/ordered by frame then center
        /peaks/area        (P,)          /
        /peaks/chi2        (P,)          reduced chi-square of the whole joint
                                         fit — the GROUP's adequacy, shared by
                                         every member; reported, never a
                                         rejection on its own
        /peaks/chi2_local  (P,)          reduced chi-square over this peak's own
                                         span (the noise-limited test)
        /peaks/rel_misfit  (P,)          rms residual there / this peak's height
                                         (the systematics-limited test)
        /peaks/flag        (P,) int      0 = good, else FLAG_* bitmask
        /peaks/center_err  (P,)          1σ fit uncertainties (NaN = fit failed);
        /peaks/amplitude_err, fwhm_err   esd-weighting + Williamson-Hall errors

    where P = sum(counts). If ``out_h5`` is given the source is copied there
    first and peaks are written into the copy; otherwise ``/peaks`` is added to
    the analysis file in place (replacing any existing group). Returns a
    manifest dict; prints ``[PEAKS] <done> <total>`` progress.
    """
    import h5py  # type: ignore
    import os
    import shutil

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    seed_axis_key = _normalize_seed_axis(seed_tracking_axis)
    seed_group_key = _normalize_seed_group(seed_group_by)
    seed_axis_values: "Optional[np.ndarray]" = None
    seed_group_values: "Optional[np.ndarray]" = None
    seed_group_labels: List[str] = ["all"]

    with h5py.File(str(src), "r") as h5:
        bg = h5.get("background")
        if bg is None or "clean" not in bg:
            raise ValueError(
                "Analysis file lacks /background/clean — run Step 1 "
                "(background.run_background_separation) first.")
        clean_raw = np.asarray(bg["clean"][:], dtype=float)
        spot = np.asarray(bg["spot_residual"][:], dtype=float) if "spot_residual" in bg else None
        sigres = (np.asarray(bg["sigmaclip_residual"][:], dtype=float)
                  if "sigmaclip_residual" in bg else None)
        radial = np.asarray(h5["radial"][:], dtype=float) if "radial" in h5 else \
            np.arange(clean_raw.shape[1], dtype=float)
        unit = str(h5.attrs.get("unit", ""))
        spotty = bool(h5.attrs.get("spotty_sample", False))
        signal_frac = float(h5.attrs.get("signal_frac_clean", float("nan")))
        frames = h5.get("frames")
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)
        if propagate_seeds:
            seed_axis_values = np.arange(clean_raw.shape[0], dtype=float)
            seed_group_values = np.zeros(clean_raw.shape[0], dtype=int)
            if seed_axis_key != "frame" or seed_group_key != "none":
                from .unknowns import _tracking_groups, _tracking_values
                if seed_axis_key != "frame":
                    seed_axis_key, seed_axis_values, _ = _tracking_values(
                        h5, seed_axis_key, clean_raw.shape[0])
                if seed_group_key != "none":
                    seed_group_key, seed_group_values, seed_group_labels = _tracking_groups(
                        h5, seed_group_key, clean_raw.shape[0])

    # Pick the channel to fit on. "auto" is DATA-driven: when Step 1 diagnosed a
    # spotty/coarse-grained sample (the median-based channels rejected the sample
    # itself — signal_frac_clean << 1), the only channel that still carries the
    # Bragg signal is the azimuthal mean. Everything else keeps the normal
    # preference order (sigmaclip if present, else hybrid).
    src_req = (source or "auto").strip().lower()
    if src_req == "auto" and spotty and spot is not None:
        print(f"[PEAKS] Step-1 diagnosis: spotty sample (only "
              f"{100 * signal_frac:.0f}% of signal survives the median) — "
              f"auto source -> 'mean'.", flush=True)
        src_req = "mean"
    clean, used_source = build_fit_source(
        src_req, clean_raw, spot_residual=spot, sigmaclip_residual=sigres,
        hybrid_spike_bins=hybrid_spike_bins)

    n = clean.shape[0]
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)
    if seed_axis_values is None or seed_axis_values.size != n:
        seed_axis_values = np.arange(n, dtype=float)
    if seed_group_values is None or seed_group_values.size != n:
        seed_group_values = np.zeros(n, dtype=int)

    # Scan/folder grouping keeps seed propagation inside one series (no seeds
    # across a scan border). It is driven entirely by the frame filenames, so if
    # they carry no scan/folder tag every frame lands in one group and the
    # setting silently does nothing — warn instead of failing quietly.
    n_seed_groups = int(np.unique(seed_group_values).size)
    if propagate_seeds and seed_group_key in ("scan", "folder") and n_seed_groups <= 1:
        print(f"[PEAKS] WARNING: seed_group_by='{seed_group_key}' but only one "
              f"{seed_group_key} was found in the frame filenames — seeds will "
              f"still propagate across the WHOLE series. Check that filenames carry "
              f"a {seed_group_key} tag (e.g. 'scan001_00007.tif' for scan, or one "
              f"sub-directory per scan for folder).", flush=True)

    # Detection knobs: a sensitivity preset fills any left unset; explicit values
    # win. sensitivity=None keeps the historical defaults (prominence coupled to
    # min_snr, no min-FWHM/edge guard).
    if sensitivity is not None:
        kn = resolve_sensitivity(sensitivity, min_snr=min_snr,
                                 min_prominence_snr=min_prominence_snr,
                                 min_fwhm_bins=min_fwhm_bins, edge_bins=edge_bins)
        r_min_snr, r_prom = kn["min_snr"], kn["min_prominence_snr"]
        r_fwhm, r_edge, preset = kn["min_fwhm_bins"], kn["edge_bins"], kn["preset"]
    else:
        r_min_snr = 5.0 if min_snr is None else float(min_snr)
        r_prom = None if min_prominence_snr is None else float(min_prominence_snr)
        r_fwhm = 0.0 if min_fwhm_bins is None else float(min_fwhm_bins)
        r_edge = 0 if edge_bins is None else int(edge_bins)
        preset = ""

    # Auto valid-range when a bound is blank (conservative; never eats interior
    # peaks). Inferred from the per-bin median of the chosen fit source.
    auto_lo = auto_hi = None
    if auto_range and (fit_min is None or fit_max is None) and n:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            rep = np.nanmedian(clean, axis=0)
        auto_lo, auto_hi = auto_fit_range(radial, rep)
        if fit_min is None and auto_lo is not None:
            fit_min = auto_lo
        if fit_max is None and auto_hi is not None:
            fit_max = auto_hi

    workers = resolve_workers(num_workers)
    prom_txt = r_min_snr if r_prom is None else r_prom
    win_txt = (f"[{'' if fit_min is None else round(fit_min, 4)},"
               f"{'' if fit_max is None else round(fit_max, 4)}] edge={r_edge} "
               f"min_fwhm_bins={r_fwhm} detrend_bins={local_baseline_bins}")
    auto_txt = (f" auto_range=[{'' if auto_lo is None else round(auto_lo, 4)},"
                f"{'' if auto_hi is None else round(auto_hi, 4)}]") if auto_range else ""
    print(f"[PEAKS] fitting {n} frames ({int(excluded.sum())} excluded), "
          f"radial[{radial.size}] unit={unit or '?'} source={used_source} "
          f"preset={preset or 'off'} min_snr={r_min_snr} min_prom={prom_txt} "
          f"window={window_factor} fit_win={win_txt}{auto_txt} "
          f"workers={workers} seed_order={seed_axis_key}/{seed_group_key}"
          f"{' predict' if propagate_seeds and seed_axis_key != 'frame' and seed_axis_predictor else ''}",
          flush=True)

    counts = np.zeros(n, dtype="i4")
    cols: Dict[str, list] = {c: [] for c in _PEAK_COLS}
    frame_idx: list = []
    ordered_seed_mode = bool(
        propagate_seeds and (
            seed_axis_key != "frame" or seed_group_key != "none"
            or seed_max_axis_gap is not None
        )
    )

    def _absorb(a, result):
        cc, fl, cols_c = result
        counts[a:a + len(cc)] = cc
        frame_idx.extend(x + a for x in fl)
        for c in _PEAK_COLS:
            cols[c].extend(cols_c[c])

    def _absorb_order(result):
        order, cc, fl, cols_c = result
        counts[np.asarray(order, dtype=int)] = np.asarray(cc, dtype="i4")
        frame_idx.extend(int(x) for x in fl)
        for c in _PEAK_COLS:
            cols[c].extend(cols_c[c])

    if ordered_seed_mode:
        seed_orders = _seed_frame_orders(n, seed_axis_values, seed_axis_key, seed_group_values)
        payloads = [
            (radial, clean[order], excluded[order], order, seed_axis_values[order],
             r_min_snr, window_factor, max_chi2, max_rel_misfit,
             propagate_seeds, r_prom,
             r_edge, fit_min, fit_max, r_fwhm, local_baseline_bins,
             seed_max_axis_gap, bool(seed_axis_predictor and seed_axis_key != "frame"))
            for order in seed_orders if len(order)
        ]
        done = 0
        if workers > 1 and len(payloads) > 1:
            # Lazy iterator: keeps the [PEAKS] progress lines streaming.
            results = process_map_or_serial(
                _peaks_order_chunk, payloads,
                max_workers=workers, label="PEAKS")
            for payload, result in zip(payloads, results):
                _absorb_order(result)
                done += int(payload[3].size)
                print(f"[PEAKS] {done} {n}", flush=True)
        else:
            for payload in payloads:
                _absorb_order(_peaks_order_chunk(payload))
                done += int(payload[3].size)
                print(f"[PEAKS] {done} {n}", flush=True)
    elif workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(radial, clean[a:b], excluded[a:b], r_min_snr, window_factor,
                     max_chi2, max_rel_misfit, propagate_seeds, r_prom,
                     r_edge, fit_min, fit_max, r_fwhm,
                     local_baseline_bins, seed_max_axis_gap,
                     bool(seed_axis_predictor and seed_axis_key != "frame"))
                    for a, b in ranges]
        done = 0
        # Lazy iterator: keeps the [PEAKS] progress lines streaming.
        results = process_map_or_serial(
            _peaks_chunk, payloads, max_workers=workers, label="PEAKS")
        for (a, b), result in zip(ranges, results):
            _absorb(a, result)
            done += (b - a)
            print(f"[PEAKS] {done} {n}", flush=True)
    else:
        _absorb(0, _peaks_chunk((radial, clean, excluded, r_min_snr, window_factor,
                                 max_chi2, max_rel_misfit, propagate_seeds, r_prom,
                                 r_edge, fit_min, fit_max, r_fwhm,
                                 local_baseline_bins, seed_max_axis_gap,
                                 bool(seed_axis_predictor and seed_axis_key != "frame"))))
        print(f"[PEAKS] {n} {n}", flush=True)

    if frame_idx:
        frame_arr = np.asarray(frame_idx, dtype="i4")
        center_arr = np.asarray(cols["center"], dtype=float)
        row_order = np.lexsort((center_arr, frame_arr))
        frame_idx = frame_arr[row_order].astype("i4").tolist()
        for c in _PEAK_COLS:
            dtype = "i4" if c == "flag" else "f8"
            cols[c] = np.asarray(cols[c], dtype=dtype)[row_order].tolist()

    P = int(counts.sum())
    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "peaks" in o:
                del o["peaks"]
            gp = o.create_group("peaks")
            write_step_provenance(o, "peaks", tool="seriesxrd.analysis.peaks",
                                  schema_version=SCHEMA_VERSION)
            gp.attrs.update({"schema_version": SCHEMA_VERSION,
                             "seriesxrd_version": VERSION,
                             "source": str(used_source),
                             "sensitivity": str(preset),
                             "hybrid_spike_bins": int(hybrid_spike_bins),
                             "min_snr": float(r_min_snr),
                             "min_prominence_snr": float(
                                 r_min_snr if r_prom is None else r_prom),
                             "window_factor": float(window_factor),
                             "max_chi2": float(max_chi2),
                             "max_rel_misfit": float(max_rel_misfit),
                             "edge_bins": int(r_edge),
                             "fit_min": float(fit_min) if fit_min is not None else np.nan,
                             "fit_max": float(fit_max) if fit_max is not None else np.nan,
                             "auto_range": bool(auto_range),
                             "min_fwhm_bins": float(r_fwhm),
                             "local_baseline_bins": int(local_baseline_bins),
                             "propagate_seeds": bool(propagate_seeds),
                             "seed_tracking_axis": str(seed_axis_key),
                             "seed_group_by": str(seed_group_key),
                             "seed_group_count": int(len(seed_group_labels)),
                             "seed_axis_predictor": bool(seed_axis_predictor),
                             "seed_max_axis_gap": (
                                 float(seed_max_axis_gap)
                                 if seed_max_axis_gap is not None else np.nan)})
            gp.create_dataset("counts", data=counts)
            gp.create_dataset("frame", data=np.asarray(frame_idx, dtype="i4"))
            for c in _PEAK_COLS:
                dt = "i4" if c == "flag" else "f8"
                gp.create_dataset(c, data=np.asarray(cols[c], dtype=dt))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    flags = np.asarray(cols["flag"], dtype=int) if P else np.zeros(0, int)
    n_good = int(np.sum(flags == FLAG_OK)) if P else 0

    # Sampling adequacy — the geometric npt suggestion (~1 bin/pixel) is
    # necessary but not sufficient: very sharp peaks can still span too few
    # bins for a stable profile fit. The MEASURED median FWHM is the ground
    # truth, so feed it back as a concrete re-reduction recommendation.
    dx = float(np.median(np.abs(np.diff(radial)))) if radial.size > 1 else 1.0
    good_fwhm = (np.asarray(cols["fwhm"], float)[flags == FLAG_OK]
                 if n_good else np.zeros(0))
    median_fwhm_bins = (float(np.median(good_fwhm)) / dx) if good_fwhm.size else float("nan")
    npt_recommended = None
    if good_fwhm.size >= 10 and median_fwhm_bins < 4.0:
        import math as _math
        npt_recommended = min(int(_math.ceil(
            radial.size * 5.0 / max(median_fwhm_bins, 0.5) / 50.0) * 50), 4000)
        print(f"[PEAKS] WARNING: peaks are UNDERSAMPLED — median FWHM is only "
              f"{median_fwhm_bins:.1f} bins (want >=5 for stable profile fits). "
              f"Re-reduce with npt_1d ~ {npt_recommended} "
              f"(currently {radial.size} bins).", flush=True)
    # Per-flag rejection tally (a peak may carry several flags; counts overlap).
    flag_defs = (("low_amp", FLAG_LOW_AMP), ("bad_chi2", FLAG_BAD_CHI2),
                 ("center_drift", FLAG_CENTER_DRIFT), ("width_bound", FLAG_WIDTH_BOUND),
                 ("no_converge", FLAG_NO_CONVERGE))
    flag_counts = {name: int(np.sum((flags & bit) != 0)) for name, bit in flag_defs}
    manifest = {
        **manifest_provenance("seriesxrd.analysis.peaks", SCHEMA_VERSION),
        "source": str(src), "out_h5": str(dst),
        "n_frames": int(n), "n_peaks": P, "n_good": n_good,
        "n_flagged": P - n_good, "unit": unit, "flag_counts": flag_counts,
        "fit_source": str(used_source), "sensitivity": str(preset),
        "fit_min": float(fit_min) if fit_min is not None else None,
        "fit_max": float(fit_max) if fit_max is not None else None,
        "min_snr": float(r_min_snr), "window_factor": float(window_factor),
        "max_chi2": float(max_chi2),
        "max_rel_misfit": float(max_rel_misfit),
        "seed_tracking_axis": str(seed_axis_key),
        "seed_group_by": str(seed_group_key),
        "seed_axis_predictor": bool(seed_axis_predictor),
        "seed_max_axis_gap": (
            float(seed_max_axis_gap) if seed_max_axis_gap is not None else None),
        "peaks_per_frame_mean": float(counts.mean()) if n else 0.0,
        "median_fwhm_bins": median_fwhm_bins,
        "npt_recommended": npt_recommended,
    }
    brk = ", ".join(f"{k}={v}" for k, v in flag_counts.items() if v)
    print(f"[PEAKS] done -> {dst}  ({P} peaks, {n_good} good"
          f"{'; rejected: ' + brk if brk else ''})", flush=True)
    return manifest
