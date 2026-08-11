"""Step 3a of the analysis pipeline: deterministic EOS-based phase matching.

For each frame, the fitted peak list from Step 2 (good peaks only) is matched
against each candidate phase from the reference library (see ``phases.py``).
Pressure is the single fitting parameter:

  1. Simulate the phase's reflections once at ambient conditions (d-spacings +
     relative intensities + hkl) via pymatgen.
  2. Under (isotropic) compression every d-spacing scales by the same factor
     ``s(P) = (V(P)/V0)**(1/3)`` from the phase's Birch-Murnaghan EOS, so the
     predicted pattern at pressure P is just ``d0 * s(P)`` — no re-simulation.
  3. Convert the observed peak centers to d-spacing (wavelength-free for q-axis
     data; needs the wavelength for a 2theta axis) and score the overlap with an
     intensity-weighted Gaussian kernel on |d_pred - d_obs|.
  4. Optimize P (grid + local refine) to maximize the score → per-frame, per-
     phase best-fit pressure, match score, and a confidence (fraction of the
     in-window predicted lines that are actually present).

``run_identification`` drives a whole analysis HDF5 (the one Steps 1-2 wrote)
and appends an ``/identify`` group. Requires pymatgen for the simulation step
(optional dependency); raises an instructive error if it is missing.

Depends on numpy (+ scipy for the local refine, used if available). pymatgen is
reached only through ``phases.simulate_pattern``.
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import (Phase, simulate_pattern, compression_at_pressure,
                     pymatgen_available, has_axial_eos, has_pressure_dof, axial_scales,
                     valid_pressure_max, thermal_scale)
from .parallel import chunk_ranges, process_map_or_serial, resolve_workers
from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance

SCHEMA_VERSION = "1"

# Wavelength used only to drive pymatgen's 2theta window when extracting the
# ambient reflection list; the d-spacings returned are wavelength-independent.
# The window is derived from a d_min cutoff (see phase_reflections): simulating
# far beyond the instrument's d-range is pure waste — the old fixed 90° window
# (d >= 0.14 Å) took SECONDS per phase even for cubic cells and minutes for
# low-symmetry polymorphs. The 1.0 Å default is only a fallback: run_identification
# derives d_min from the reduction's actual q-range (a short-λ beamline reaches
# q≈11 Å⁻¹ ⇒ d≈0.55 Å, well below 1.0), so higher-order lines are modelled rather
# than mistaken for unknown phases.
_SIM_WAVELENGTH = 0.2
_SIM_D_MIN_DEFAULT = 1.0
# Strongest-N reflection cap at the reference d_min=1.0; run_identification scales
# it up with the reciprocal-space volume for wider q-ranges (see sim_max_refl).
_REFL_CAP_BASE = 40


# ---------------------------------------------------------------------------
# Axis → d-spacing
# ---------------------------------------------------------------------------

def radial_to_d(centers, unit: str, wavelength: "Optional[float]" = None) -> np.ndarray:
    """Convert peak centers on the reduced radial axis to d-spacing (Å).

    ``unit`` is the reduced/analysis ``unit`` attribute: ``q_A^-1``, ``q_nm^-1``,
    ``2th_deg`` or ``2th_rad``. The 2theta forms need ``wavelength`` (Å).
    """
    c = np.asarray(centers, dtype=float)
    u = (unit or "").strip().lower()
    if u in ("q_a^-1", "q_a-1", "q_a", "q"):          # q in Å^-1
        return 2.0 * np.pi / c
    if u in ("q_nm^-1", "q_nm-1", "q_nm"):            # q in nm^-1 → Å^-1 = /10
        return 2.0 * np.pi / (c * 0.1)
    if u in ("2th_deg", "2th_rad"):
        if wavelength is None or wavelength <= 0:
            raise ValueError(f"wavelength (Å) is required to convert {unit} to d-spacing.")
        theta = np.radians(c) / 2.0 if u == "2th_deg" else c / 2.0
        return float(wavelength) / (2.0 * np.sin(theta))
    raise ValueError(f"Unsupported unit for d-spacing conversion: {unit!r}")


def radial_err_to_d_err(centers, errs, unit: str,
                        wavelength: "Optional[float]" = None) -> np.ndarray:
    """Propagate 1σ peak-center uncertainties on the radial axis to d-spacing (Å).

    q axes: d = 2π/q ⇒ σ_d = d·σ_q/q. 2θ axes: d = λ/(2 sinθ) ⇒
    σ_d = d·cot(θ)·σ_2θ/2 (σ_2θ in radians). Non-finite/zero errors map to NaN
    (callers treat NaN as "no esd available" and fall back to rel_tol alone).
    """
    c = np.asarray(centers, dtype=float)
    e = np.asarray(errs, dtype=float)
    u = (unit or "").strip().lower()
    with np.errstate(divide="ignore", invalid="ignore"):
        if u in ("q_a^-1", "q_a-1", "q_a", "q", "q_nm^-1", "q_nm-1", "q_nm"):
            d = radial_to_d(c, unit, wavelength)
            out = d * e / c                      # unit conversion cancels in e/c
        elif u in ("2th_deg", "2th_rad"):
            if wavelength is None or wavelength <= 0:
                return np.full(c.shape, np.nan)
            theta = np.radians(c) / 2.0 if u == "2th_deg" else c / 2.0
            e_rad = np.radians(e) if u == "2th_deg" else e
            d = radial_to_d(c, unit, wavelength)
            out = d / np.tan(theta) * e_rad / 2.0
        else:
            return np.full(c.shape, np.nan)
    out = np.abs(out)
    out[~(out > 0)] = np.nan
    return out


def wavelength_from_reduced(reduced_path: "str | Path") -> "Optional[float]":
    """Best-effort read of the wavelength (Å) from a reduced file's PONI text."""
    try:
        import h5py  # type: ignore
        with h5py.File(str(reduced_path), "r") as h5:
            poni = h5.attrs.get("poni_text", "")
            if isinstance(poni, bytes):
                poni = poni.decode("utf-8", "replace")
        m = re.search(r"wavelength\s*:\s*([0-9eE.+-]+)", str(poni), re.IGNORECASE)
        if m:
            wl_m = float(m.group(1))           # pyFAI stores it in metres
            return wl_m * 1e10 if wl_m < 1e-6 else wl_m
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Reflections & pressure scaling
# ---------------------------------------------------------------------------

def phase_reflections(phase: Phase, *, max_reflections: int = _REFL_CAP_BASE,
                      min_rel_intensity: float = 0.01,
                      d_min: float = _SIM_D_MIN_DEFAULT
                      ) -> "Tuple[np.ndarray, np.ndarray, List[str]]":
    """Ambient (d, weight, hkl) reflection list for a phase, strongest first.

    ``weight`` is the relative intensity normalised to its max. Keeps the
    strongest ``max_reflections`` lines above ``min_rel_intensity`` with
    d ≥ ``d_min`` (Å). The 1.0 Å default is a conservative fallback; callers with
    data should pass a ``d_min`` matched to the reduction's q_max
    (``run_identification`` does this) so high-q higher-order lines are modelled.
    Restricting the simulation window this way also keeps pymatgen fast for
    low-symmetry cells, and stops sub-Å lines the detector can never see from
    crowding real lines out of the strongest-``max_reflections`` selection.
    Requires pymatgen (via :func:`phases.simulate_pattern`).
    """
    tt_max = 2.0 * math.degrees(math.asin(
        min(1.0, _SIM_WAVELENGTH / (2.0 * max(float(d_min), 0.05)))))
    pat = simulate_pattern(phase, _SIM_WAVELENGTH, two_theta_min=0.0,
                           two_theta_max=tt_max)
    d = np.array([p["d"] for p in pat], dtype=float)
    inten = np.array([p["intensity"] for p in pat], dtype=float)
    hkl = [p["hkl"] for p in pat]
    if d.size == 0:
        return d, inten, hkl
    w = inten / (inten.max() or 1.0)
    keep = w >= float(min_rel_intensity)
    d, w = d[keep], w[keep]
    hkl = [h for h, k in zip(hkl, keep) if k]
    order = np.argsort(w)[::-1][:max_reflections]
    return d[order], w[order], [hkl[i] for i in order]


def _simulate_one(phase: Phase, d_min: float, max_reflections: int):
    """Pool entry point for :func:`phase_reflections`.

    Module level and returning plain data so it pickles: a bound method or a
    closure would not survive the process boundary. Errors come back as a value
    rather than propagating, so one unsimulable phase cannot abort the pool.
    """
    try:
        return phase.name, phase_reflections(
            phase, d_min=d_min, max_reflections=max_reflections), None
    except Exception as e:                                # noqa: BLE001
        return phase.name, None, f"{type(e).__name__}: {e}"


def _simulate_one_payload(payload):
    """Unary adapter so reflection simulations use the shared safe map helper."""
    return _simulate_one(*payload)


def scale_at_pressure(phase: Phase, pressure: float) -> float:
    """Isotropic lattice scale factor ``s = (V(P)/V0)**(1/3)`` from the phase EOS
    (any supported type: BM2/BM3/BM4, Vinet, Murnaghan).

    Returns 1.0 at/below ambient or when the phase has no usable EOS."""
    if pressure <= 0 or not phase.has_eos():
        return 1.0
    return compression_at_pressure(phase.eos, float(pressure)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _nearest_obs(pred: np.ndarray, obs_sorted: np.ndarray
                 ) -> "Tuple[np.ndarray, np.ndarray]":
    """For each predicted d: (distance to, index of) the nearest observed d."""
    if obs_sorted.size == 0:
        return np.full(pred.shape, np.inf), np.zeros(pred.shape, dtype=int)
    idx = np.searchsorted(obs_sorted, pred)
    left = np.clip(idx - 1, 0, obs_sorted.size - 1)
    right = np.clip(idx, 0, obs_sorted.size - 1)
    gl = np.abs(pred - obs_sorted[left])
    gr = np.abs(pred - obs_sorted[right])
    nearest = np.where(gl <= gr, left, right)
    return np.minimum(gl, gr), nearest


def _nearest_gap(pred: np.ndarray, obs_sorted: np.ndarray) -> np.ndarray:
    """For each predicted d, distance to the nearest observed d (obs ascending)."""
    return _nearest_obs(pred, obs_sorted)[0]


def _parse_hkl(s) -> "Optional[Tuple[int, int, int]]":
    """Parse an hkl label like '(1, -1, 0)' or '1 1 1' to a tuple, or None."""
    nums = re.findall(r"-?\d+", str(s))
    return (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else None


def _d_from_lattice(H: np.ndarray, lattice: "Dict[str, Any]") -> np.ndarray:
    """d-spacings (Å) for hkl rows ``H`` (N×3) via the reciprocal metric tensor.

    General triclinic formula 1/d² = hᵀ G* h with G* = inv(G); valid for every
    crystal system, so anisotropic compression of a, b, c (and angles) is handled
    exactly."""
    L = lattice or {}
    a = float(L.get("a") or 0.0)
    b = float(L.get("b") or a) or a
    c = float(L.get("c") or a) or a
    al = math.radians(float(L.get("alpha", 90.0) or 90.0))
    be = math.radians(float(L.get("beta", 90.0) or 90.0))
    ga = math.radians(float(L.get("gamma", 90.0) or 90.0))
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    G = np.array([[a * a, a * b * cg, a * c * cb],
                  [a * b * cg, b * b, b * c * ca],
                  [a * c * cb, b * c * ca, c * c]], dtype=float)
    Gstar = np.linalg.inv(G)
    s = np.einsum("ni,ij,nj->n", H, Gstar, H)
    s = np.where(s > 0, s, np.nan)
    return 1.0 / np.sqrt(s)


def predicted_d(phase: Phase, d0: np.ndarray, hkls, P: float,
                temperature: "Optional[float]" = None) -> np.ndarray:
    """Predicted d-spacings (Å) at pressure ``P`` (and optionally temperature).

    Anisotropic when the phase has an axial EOS, a lattice, and parseable hkl:
    each reflection's d0 is scaled by the *ratio* of its compressed-lattice
    d-spacing to its ambient one (so it reduces exactly to d0 at P=0 and is
    independent of any small offset between the simulated d0 and the metric
    calculation). Otherwise the isotropic ``d0·s``. ``temperature`` (K) applies
    the phase's isotropic thermal expansion on top (see
    :func:`phases.thermal_scale`; 1.0 when the phase has no thermal data) —
    the seam that makes ambient-pressure temperature series work."""
    d0 = np.asarray(d0, dtype=float)
    s_T = thermal_scale(phase, temperature)
    if (P > 0 and has_axial_eos(phase) and phase.lattice
            and hkls is not None and all(h is not None for h in hkls)):
        sa, sb, sc = axial_scales(phase, P)
        Lp = dict(phase.lattice)
        for k, s in (("a", sa), ("b", sb), ("c", sc)):
            if Lp.get(k):
                Lp[k] = float(Lp[k]) * s
        H = np.asarray(hkls, dtype=float)
        ratio = _d_from_lattice(H, Lp) / _d_from_lattice(H, phase.lattice)
        dd = d0 * ratio
        bad = ~np.isfinite(dd)
        if bad.any():
            dd[bad] = d0[bad] * scale_at_pressure(phase, P)
        return dd * s_T
    return d0 * (scale_at_pressure(phase, P) * s_T)


def _match_pairs(pred: np.ndarray, obs: np.ndarray, rel_tol: float,
                 factor: float = 2.0,
                 obs_err: "Optional[np.ndarray]" = None
                 ) -> "List[Tuple[int, int, float]]":
    """Greedy ONE-TO-ONE assignment of predicted lines to observed peaks.

    A predicted line and an observed peak can be paired only once: closest pairs
    are consumed first, within ``factor·σᵢⱼ`` where
    ``σᵢⱼ = sqrt((rel_tol·predᵢ)² + obs_errⱼ²)`` — the model tolerance and the
    observed peak's own fitted center uncertainty added in quadrature (a noisy,
    weak peak legitimately sits further from its line than a sharp strong one).
    ``obs_err`` NaN/absent falls back to the model tolerance alone. This
    one-to-one rule stops a single observed peak from "explaining" several
    predicted reflections (which inflated n_matched / recall and let wrong
    phases look present in DAC data). Returns ``[(i_pred, j_obs, gap)]``.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.size == 0 or obs.size == 0:
        return []
    errs = None
    if obs_err is not None:
        errs = np.asarray(obs_err, dtype=float)
        if errs.shape != obs.shape:
            errs = None
    cand: "List[Tuple[float, int, int]]" = []
    for i, pi in enumerate(pred):
        if not np.isfinite(pi) or pi <= 0:
            continue
        sig_model = max(rel_tol * pi, 1e-9)
        for j, oj in enumerate(obs):
            e = errs[j] if errs is not None and np.isfinite(errs[j]) else 0.0
            tol = factor * math.hypot(sig_model, e)
            g = abs(pi - oj)
            if g < tol:
                cand.append((g, i, j))
    cand.sort()
    used_p: set = set()
    used_o: set = set()
    out: "List[Tuple[int, int, float]]" = []
    for g, i, j in cand:
        if i in used_p or j in used_o:
            continue
        used_p.add(i)
        used_o.add(j)
        out.append((i, j, g))
    return out


def _score_pred(obs_sorted: np.ndarray, pred: np.ndarray, weight: np.ndarray,
                rel_tol: float, strong_frac: float = 0.1,
                obs_err_sorted: "Optional[np.ndarray]" = None,
                obs_amp_sorted: "Optional[np.ndarray]" = None,
                ) -> Tuple[float, int, float, float, float]:
    """Match score, #matched, recall, precision, intensity_corr for a predicted
    d-spacing array.

    ``score`` = Σ wᵢ·exp(−½(Δdᵢ/σᵢ)²) — a smooth optimisation target for the
    pressure fit (nearest-neighbour, so it stays differentiable as P sweeps).
    σᵢ combines the model tolerance ``rel_tol·d_pred,ᵢ`` with the nearest
    observed peak's fitted center esd in quadrature (``obs_err_sorted``, from
    the Step-2 covariance) so a sharp, well-determined peak is held to a
    tighter position than a broad noisy one.

    The reported *evidence* (n_matched, recall, precision), by contrast, uses a
    hard ONE-TO-ONE assignment so a wrong phase cannot reuse one peak many times:

    * ``recall`` — intensity-weighted fraction of the phase's *strong, observable*
      lines (≥ ``strong_frac`` of the strongest in-window line) that are matched
      one-to-one.
    * ``precision`` — fraction of the *observed* peaks in the predicted range that
      a predicted line claims (matched-pairs / observed-in-range, ≤ 1).
    * ``intensity_corr`` — cosine similarity between the predicted relative
      intensities and the observed amplitudes over the one-to-one matched pairs
      (NaN when ``obs_amp_sorted`` is absent or fewer than 3 pairs matched).
      DAC texture scrambles intensities, so this is a *soft* figure of merit —
      see ``conservative_confidence(intensity_k=...)`` for how gently it is
      folded in.
    """
    pred = np.asarray(pred, dtype=float)
    sigma_model = np.maximum(rel_tol * pred, 1e-9)
    gap, near = _nearest_obs(pred, obs_sorted)
    sigma = sigma_model
    if obs_err_sorted is not None and obs_sorted.size:
        e = np.asarray(obs_err_sorted, dtype=float)[near]
        e = np.where(np.isfinite(e), e, 0.0)
        sigma = np.sqrt(sigma_model ** 2 + e ** 2)
    kernel = np.exp(-0.5 * (gap / sigma) ** 2)
    score = float(np.sum(weight * kernel))
    pairs = _match_pairs(pred, obs_sorted, rel_tol, factor=2.0,
                         obs_err=obs_err_sorted)
    n_matched = len(pairs)
    matched_pred = np.zeros(pred.size, dtype=bool)
    for i, _j, _g in pairs:
        matched_pred[i] = True
    recall = 0.0
    precision = 0.0
    intensity_corr = float("nan")
    if obs_sorted.size and pred.size:
        span = 2.0 * float(sigma.max())
        # Recall over the strong in-window predicted lines (matched one-to-one).
        in_win = (pred >= obs_sorted[0] - span) & (pred <= obs_sorted[-1] + span)
        w_win = weight[in_win]
        if w_win.size:
            strong = w_win >= strong_frac * float(w_win.max())
            denom = float(np.sum(w_win[strong]))
            num = float(np.sum((weight * kernel * matched_pred)[in_win][strong]))
            recall = num / denom if denom > 0 else 0.0
        # Precision: matched pairs / observed peaks in the predicted d-range.
        pred_sorted = np.sort(pred)
        lo, hi = pred_sorted[0] - span, pred_sorted[-1] + span
        n_obs_in = int(np.sum((obs_sorted >= lo) & (obs_sorted <= hi)))
        if n_obs_in:
            precision = min(1.0, n_matched / float(n_obs_in))
    # Intensity agreement over the matched pairs (predicted weight vs observed
    # amplitude, both non-negative → cosine in [0, 1]).
    if obs_amp_sorted is not None and n_matched >= 3:
        amp = np.asarray(obs_amp_sorted, dtype=float)
        wp = np.array([weight[i] for i, _j, _g in pairs], dtype=float)
        ao = np.array([amp[j] for _i, j, _g in pairs], dtype=float)
        ok = np.isfinite(wp) & np.isfinite(ao) & (ao >= 0)
        if int(ok.sum()) >= 3:
            wp, ao = wp[ok], ao[ok]
            nw, na = float(np.linalg.norm(wp)), float(np.linalg.norm(ao))
            if nw > 0 and na > 0:
                intensity_corr = float(np.dot(wp, ao) / (nw * na))
    return score, n_matched, recall, precision, intensity_corr


def score_at_scale(obs_sorted: np.ndarray, d0: np.ndarray, weight: np.ndarray,
                   s: float, rel_tol: float,
                   strong_frac: float = 0.1) -> Tuple[float, int, float, float]:
    """Isotropic convenience wrapper: score the predictions ``d0·s``.

    Back-compat: returns the classic ``(score, n_matched, recall, precision)``
    4-tuple (no esd/intensity inputs here — use :func:`_score_pred` for those).
    """
    return _score_pred(obs_sorted, np.asarray(d0, dtype=float) * s, weight,
                       rel_tol, strong_frac)[:4]


DEFAULT_MIN_MATCHED = 3
# Floor on the half-width of the pressure search/penalty window (GPa), so a
# near-zero metadata sigma can't collapse the window to nothing.
_MIN_PRESSURE_WINDOW = 0.3


def pressure_window_halfwidth(sigma: "Optional[float]", default_window: float,
                              sigma_k: float = 2.0) -> float:
    """Half-width (GPa) of the pressure prior window for one frame.

    Uses ``sigma_k·sigma`` when a per-frame pressure uncertainty is known,
    otherwise the ``default_window``; floored at :data:`_MIN_PRESSURE_WINDOW`."""
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        w = float(sigma_k) * float(sigma)
    else:
        w = float(default_window)
    return max(w, _MIN_PRESSURE_WINDOW)


def prior_range_offenders(prior_pressure, windows, p_min: float, p_max: float,
                          names: "Optional[Sequence[str]]" = None,
                          max_report: int = 5) -> "list[str]":
    """Describe the frames whose metadata pressure (± window) forces the search
    range beyond ``[p_min, p_max]`` — usually one bad value (a mistyped
    filename token or CSV row), so name it instead of only reporting the
    widened range. A frame far outside the spread of the other priors gets an
    explicit likely-outlier hint. Returns printable lines (empty = no
    offenders)."""
    pr = np.asarray(prior_pressure, dtype=float)
    w = np.asarray(windows, dtype=float)
    fin = np.isfinite(pr)
    # Mirror the widening logic: the low side is clamped at 0 GPa, so a small
    # prior minus its window only offends when p_min itself is above 0.
    idx = np.nonzero(fin & ((np.maximum(0.0, pr - w) < p_min)
                            | (pr + w > p_max)))[0]
    lines: "list[str]" = []
    if idx.size == 0:
        return lines
    med = float(np.median(pr[fin]))
    mad = float(np.median(np.abs(pr[fin] - med)))
    outlier_bar = max(5.0, 8.0 * 1.4826 * mad)
    for j in idx[:max_report]:
        name = ""
        if names is not None and j < len(names) and names[j]:
            name = f" ({Path(str(names[j])).name})"
        tag = ""
        if abs(pr[j] - med) > outlier_bar:
            tag = (f" — far from the series median ({med:g} GPa); likely a bad "
                   "metadata value. Fix it on the Frame metadata tab (manual "
                   "edits persist across re-runs) or in the pressure CSV.")
        lines.append(f"  frame {int(j)}{name}: prior {pr[j]:g} GPa{tag}")
    if idx.size > max_report:
        lines.append(f"  ... and {int(idx.size) - max_report} more frame(s)")
    return lines


PRESSURE_ASSUMPTIONS = ("eos_based", "ambient_reference", "eos_missing", "ignore_prior")


def pressure_model(phase: Phase) -> str:
    """The *math* available to move a phase's peaks with pressure: ``axial_eos``
    (per-axis EOS, anisotropic), ``eos`` (isotropic volume EOS), or ``no_eos``
    (none — scored at 0 GPa). This says nothing about whether 0 GPa is the right
    interpretation; see :func:`pressure_assumption`."""
    if has_axial_eos(phase):
        return "axial_eos"
    if phase.has_eos():
        return "eos"
    return "no_eos"


def pressure_assumption(phase: Phase) -> str:
    """How the pressure prior is *interpreted* for a phase (vs. the raw model).

    A ``no_eos`` phase is ambiguous — it may be a genuine ambient-only reference
    or simply lack high-P EOS data — and a user may want to exempt some phases
    from the prior entirely. Resolves the phase's ``pressure_assumption`` field
    (``ambient_reference`` | ``eos_missing`` | ``ignore_prior``) or, when unset,
    defaults to ``eos_based`` for EOS phases and ``eos_missing`` for ``no_eos``.
    Only ``ignore_prior`` changes behaviour (no penalty); the rest are labels."""
    raw = str(getattr(phase, "pressure_assumption", "") or "").strip().lower()
    if raw in ("ambient_reference", "eos_missing", "ignore_prior"):
        return raw
    return "eos_based" if pressure_model(phase) != "no_eos" else "eos_missing"


def conservative_confidence(recall: float, precision: float, n_matched: int,
                            *, min_matched: int = DEFAULT_MIN_MATCHED,
                            prior_penalty: float = 1.0,
                            intensity_corr: float = float("nan"),
                            intensity_k: float = 0.0) -> float:
    """Combine the figures of merit into ONE conservative confidence in [0, 1].

    Replaces the old ``max(recall, precision)`` — which let a phase explaining a
    couple of the busiest peaks score 1.0 — with multiplicative factors:

    * ``balanced`` = F1(recall, precision): demands the phase both shows its own
      strong lines *and* accounts for what is observed (neither alone suffices).
    * ``evidence`` = min(1, n_matched / min_matched): penalises matches resting on
      too few reflections (the DARA/RADAR-PD "minimum evidence" lesson).
    * ``prior_penalty`` ∈ (0, 1]: Gaussian falloff when the fitted pressure
      disagrees with the frame's metadata pressure (1.0 when no prior).
    * intensity factor = ``1 − intensity_k·(1 − intensity_corr)``: a *soft* pull
      toward matches whose observed amplitudes track the predicted relative
      intensities (DARA-style). Deliberately gentle (``intensity_k`` small, and
      skipped entirely when ``intensity_corr`` is NaN) because DAC texture,
      spotty rings and preferred orientation legitimately scramble intensities —
      position evidence stays primary; intensities only nudge.
    """
    r = max(0.0, float(recall))
    p = max(0.0, float(precision))
    balanced = (2.0 * r * p / (r + p)) if (r + p) > 0 else 0.0
    mm = max(1, int(min_matched))
    evidence = min(1.0, max(0, int(n_matched)) / float(mm))
    conf = balanced * evidence * max(0.0, min(1.0, prior_penalty))
    k = max(0.0, min(1.0, float(intensity_k)))
    if k > 0 and intensity_corr == intensity_corr:          # corr is not NaN
        conf *= 1.0 - k * (1.0 - max(0.0, min(1.0, float(intensity_corr))))
    return float(conf)


def fit_pressure_for_phase(obs_d, phase: Phase,
                           refl: "Optional[Tuple[np.ndarray, np.ndarray, List[str]]]" = None,
                           *, p_min: float = 0.0, p_max: float = 100.0,
                           n_grid: int = 300, rel_tol: float = 0.01,
                           p_prior: "Optional[float]" = None,
                           p_window: "Optional[float]" = None,
                           min_matched: int = DEFAULT_MIN_MATCHED,
                           obs_err=None, obs_amp=None,
                           temperature: "Optional[float]" = None,
                           intensity_k: float = 0.3) -> Dict[str, Any]:
    """Best-fit pressure for one phase against one frame's observed d-spacings.

    Returns ``{pressure, score, confidence, recall, precision, n_matched,
    intensity_corr, n_pred}``. Compression is anisotropic when the phase carries
    an ``axial_eos`` (per-axis), else isotropic from the volume EOS; with
    neither, the phase is only scored at ambient (pressure 0). ``refl`` may be
    precomputed (see :func:`phase_reflections`) to avoid re-simulating across
    frames.

    ``obs_err`` (per-peak d-spacing esd, from the Step-2 fit covariance) widens
    the match tolerance for noisy peaks in quadrature; ``obs_amp`` (per-peak
    amplitude) enables the soft intensity-agreement factor (``intensity_k``,
    see :func:`conservative_confidence`); ``temperature`` (K) applies the
    phase's thermal expansion so an ambient-pressure temperature series is
    matched at the right lattice.

    Pressure prior — the key DAC fix. When ``p_prior`` (a frame's metadata
    pressure, GPa) is given, the search is *confined* to ``p_prior ± p_window``
    instead of the whole ``[p_min, p_max]`` range, so a wrong phase can no longer
    slide along pressure until a few lines happen to coincide. The fitted
    pressure is additionally weighed against the prior by a Gaussian penalty in
    :func:`conservative_confidence` (tolerance = ``p_window``). The penalty also
    applies to no-EOS phases (scored at ambient, so they are dragged down on a
    high-pressure frame).

    Phases whose ``pressure_assumption`` is ``ignore_prior`` are exempt from the
    prior ENTIRELY — no confidence penalty AND a free pressure search over the
    whole ``[p_min, p_max]``. This is the seam for material that is genuinely
    NOT at the chamber pressure: e.g. a second copy of the pressure-marker/gasket
    metal picked up at the gasket flank or bridged to an anvil, which sits tens
    of GPa away from the sample chamber and would otherwise be unmatchable
    (and pollute the unknowns) because the confined search can never reach it.
    """
    if refl is None:
        refl = phase_reflections(phase)
    d0, weight, hkl_raw = refl
    d0 = np.asarray(d0, dtype=float)
    weight = np.asarray(weight, dtype=float)
    hkls = [_parse_hkl(h) for h in hkl_raw] if hkl_raw is not None else None
    obs_raw = np.asarray(obs_d, dtype=float)
    valid = np.isfinite(obs_raw) & (obs_raw > 0)
    order = np.argsort(obs_raw[valid])
    obs = obs_raw[valid][order]

    def _aligned(aux):
        """Sort an optional per-peak array the same way as obs (else None)."""
        if aux is None:
            return None
        a = np.asarray(aux, dtype=float)
        if a.shape != obs_raw.shape:
            return None
        return a[valid][order]

    err_s = _aligned(obs_err)
    amp_s = _aligned(obs_amp)
    out = {"pressure": float("nan"), "score": 0.0, "confidence": 0.0,
           "recall": 0.0, "precision": 0.0, "n_matched": 0, "prior_penalty": 1.0,
           "intensity_corr": float("nan"), "n_pred": int(d0.size)}
    if d0.size == 0 or obs.size == 0:
        out["pressure"] = 0.0
        return out

    has_prior = (p_prior is not None and np.isfinite(p_prior))
    has_eos = has_pressure_dof(phase)
    # The pressure-prior penalty applies whenever a frame pressure is known —
    # including for no-EOS phases (scored at ambient, so they are dragged down on
    # a high-pressure frame rather than falsely matching as ambient) — unless the
    # phase is explicitly exempted via pressure_assumption == "ignore_prior".
    apply_pen = has_prior and pressure_assumption(phase) != "ignore_prior"
    pen_prior = float(p_prior) if apply_pen else None
    pen_tol = float(p_window) if (p_window and p_window > 0) else None

    def _score_P(P):
        return _score_pred(obs, predicted_d(phase, d0, hkls, P, temperature),
                           weight, rel_tol,
                           obs_err_sorted=err_s, obs_amp_sorted=amp_s)

    def _record(p_val):
        score, nm, rec, prec, icorr = _score_P(p_val)
        prior_pen = 1.0
        if pen_prior is not None and pen_tol:
            prior_pen = float(np.exp(-0.5 * ((p_val - pen_prior) / pen_tol) ** 2))
        out.update({"pressure": p_val, "score": score, "n_matched": nm,
                    "recall": rec, "precision": prec, "prior_penalty": prior_pen,
                    "intensity_corr": icorr,
                    "confidence": conservative_confidence(
                        rec, prec, nm, min_matched=min_matched,
                        prior_penalty=prior_pen,
                        intensity_corr=icorr, intensity_k=intensity_k)})

    if not has_eos:
        _record(0.0)
        return out

    # Confine the search to the prior window when we have one, and never past
    # the phase's validity ceiling (eos['p_max'], e.g. a phase transition) —
    # extrapolating the EOS into a regime where the phase does not exist lets a
    # stability-limited entry "match" data it cannot physically produce.
    # ignore_prior phases search the full range: they model material that is NOT
    # at the chamber pressure (gasket-flank / anvil-bridged marker metal), which
    # a prior-confined search could never reach (see the docstring).
    p_valid = valid_pressure_max(phase)
    lo_b, hi_b = float(p_min), min(float(p_max), p_valid)
    if apply_pen and p_window and p_window > 0:
        lo_b = max(lo_b, float(p_prior) - float(p_window))
        hi_b = min(hi_b, float(p_prior) + float(p_window))
        if hi_b <= lo_b:                       # prior sits outside the valid range
            lo_b = hi_b = min(max(float(p_prior), float(p_min)), float(p_max), p_valid)

    if hi_b <= lo_b:
        _record(lo_b)
        return out

    Ps = np.linspace(lo_b, hi_b, int(max(n_grid, 2)))
    scores = np.array([_score_P(P)[0] for P in Ps])
    best = int(np.argmax(scores))
    p_opt = float(Ps[best])
    # Local refine within the bracketing grid cell.
    lo = float(Ps[max(best - 1, 0)])
    hi = float(Ps[min(best + 1, Ps.size - 1)])
    if hi > lo:
        try:
            from scipy.optimize import minimize_scalar  # lazy
            r = minimize_scalar(lambda P: -_score_P(P)[0],
                                bounds=(lo, hi), method="bounded")
            if r.success and -r.fun >= scores[best]:
                p_opt = float(r.x)
        except Exception:
            pass
    _record(p_opt)
    return out


# ---------------------------------------------------------------------------
# Dataset driver (analysis HDF5 -> /identify appended)
# ---------------------------------------------------------------------------

def _h5_safe(name: str) -> str:
    return name.replace("/", "_")


def _identify_chunk(payload):
    """Worker: fit pressure for every phase over a contiguous chunk of frames.

    Reflections are precomputed in the parent and passed in, so the workers need
    no pymatgen. Excluded frames are left as NaN/zero.
    """
    (phases, refls, obs_chunk, excluded_chunk, prior_chunk, window_chunk,
     p_min, p_max, rel_tol, min_matched,
     amp_chunk, err_chunk, temp_chunk, intensity_k) = payload
    m = len(obs_chunk)
    res = {ph.name: {"pressure": np.full(m, np.nan, "f8"), "score": np.zeros(m, "f8"),
                     "confidence": np.zeros(m, "f8"), "recall": np.zeros(m, "f8"),
                     "precision": np.zeros(m, "f8"), "n_matched": np.zeros(m, "i4"),
                     "prior_penalty": np.ones(m, "f8"),
                     "intensity_corr": np.full(m, np.nan, "f8")}
           for ph in phases}
    for j in range(m):
        if excluded_chunk[j]:
            continue
        obs = obs_chunk[j]
        pp = None
        if prior_chunk is not None and np.isfinite(prior_chunk[j]):
            pp = float(prior_chunk[j])
        pw = float(window_chunk[j]) if (window_chunk is not None and pp is not None) else None
        temp = None
        if temp_chunk is not None and np.isfinite(temp_chunk[j]):
            temp = float(temp_chunk[j])
        for ph, refl in zip(phases, refls):
            r = fit_pressure_for_phase(obs, ph, refl, p_min=p_min, p_max=p_max,
                                       rel_tol=rel_tol, p_prior=pp, p_window=pw,
                                       min_matched=min_matched,
                                       obs_err=(err_chunk[j] if err_chunk is not None else None),
                                       obs_amp=(amp_chunk[j] if amp_chunk is not None else None),
                                       temperature=temp, intensity_k=intensity_k)
            rr = res[ph.name]
            rr["pressure"][j] = r["pressure"]; rr["score"][j] = r["score"]
            rr["confidence"][j] = r["confidence"]; rr["recall"][j] = r["recall"]
            rr["precision"][j] = r["precision"]; rr["n_matched"][j] = r["n_matched"]
            rr["prior_penalty"][j] = r["prior_penalty"]
            rr["intensity_corr"][j] = r["intensity_corr"]
    return res


def _read_good_peaks_by_frame(h5) -> "Tuple[int, List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]":
    """Good peaks per frame: ``(n, centers, amplitudes, center_errs)``.

    ``amplitudes`` / ``center_errs`` are per-frame arrays aligned with
    ``centers``, or None when the file predates those columns (older Step-2
    output) — identification then runs position-only, as before.
    """
    pk = h5.get("peaks")
    if pk is None or "center" not in pk or "frame" not in pk:
        raise ValueError("Analysis file lacks /peaks — run Step 2 (peak fitting) first.")
    frame = np.asarray(pk["frame"][:], dtype=int)
    center = np.asarray(pk["center"][:], dtype=float)
    flag = np.asarray(pk["flag"][:], dtype=int) if "flag" in pk else np.zeros_like(center, int)
    amp = np.asarray(pk["amplitude"][:], dtype=float) if "amplitude" in pk else None
    cerr = np.asarray(pk["center_err"][:], dtype=float) if "center_err" in pk else None
    if "counts" in pk:
        n = int(np.asarray(pk["counts"][:]).size)
    else:
        bg = h5.get("background")
        n = int(bg["clean"].shape[0]) if bg is not None and "clean" in bg else (int(frame.max()) + 1 if frame.size else 0)
    good = flag == 0
    centers = [center[good & (frame == i)] for i in range(n)]
    amps = ([amp[good & (frame == i)] for i in range(n)]
            if amp is not None and amp.shape == center.shape else None)
    cerrs = ([cerr[good & (frame == i)] for i in range(n)]
             if cerr is not None and cerr.shape == center.shape else None)
    return n, centers, amps, cerrs


def run_identification(
    analysis_h5: "str | Path",
    phases: "Sequence[Phase]",
    *,
    wavelength: "Optional[float]" = None,
    p_min: float = 0.0,
    p_max: float = 100.0,
    rel_tol: float = 0.01,
    out_h5: "Optional[str | Path]" = None,
    num_workers: int = 1,
    pressure_by_frame: "Optional[Sequence[float]]" = None,
    pressure_sigma_by_frame: "Optional[Sequence[float]]" = None,
    use_frame_pressure: bool = True,
    pressure_window: float = 2.0,
    pressure_sigma_k: float = 2.0,
    min_matched: int = DEFAULT_MIN_MATCHED,
    seen_conf: float = 0.5,
    marker_prior: bool = False,
    intensity_k: float = 0.3,
    use_frame_temperature: bool = True,
) -> Dict[str, Any]:
    """Match every frame's good peaks against each candidate phase and store the
    per-frame best-fit pressure / score / confidence under ``/identify``.

        /identify  attrs: schema_version, unit, wavelength, p_min, p_max, rel_tol,
                          phases, pressure_window, pressure_sigma_k, min_matched,
                          intensity_k, sim_d_min, sim_max_refl, n_temperature
        /identify/<phase>/pressure    (N,)  best-fit pressure (GPa)
        /identify/<phase>/score       (N,)  match score
        /identify/<phase>/confidence  (N,)  conservative confidence (see
                                            conservative_confidence)
        /identify/<phase>/recall, /precision (N,)
        /identify/<phase>/n_matched   (N,) int  one-to-one matched reflections
        /identify/<phase>/prior_penalty (N,)    Gaussian pressure-prior factor in
                                                (0,1] applied to confidence (1=no prior)
        /identify/<phase>/intensity_corr (N,)   cosine of predicted vs observed
                                                intensities over matched pairs
                                                (NaN = too few pairs / no amps)
                          attrs: name, n_pred, has_eos, category, pressure_model
                                 (eos|axial_eos|no_eos), pressure_assumption,
                                 prior_penalized

    Position evidence is esd-weighted when Step 2 stored ``center_err`` (the
    match tolerance widens in quadrature with each observed peak's own fitted
    uncertainty), and observed amplitudes feed a soft intensity-agreement
    factor (``intensity_k``; 0 disables — see ``conservative_confidence``).
    ``use_frame_temperature`` applies ``/frames/temperature`` through each
    phase's thermal expansion (``Phase.thermal``), the ambient-pressure
    temperature-series analog of the pressure prior.

    Pressure prior (the DAC accuracy fix): if ``pressure_by_frame`` is given — or
    the analysis file already carries ``/frames/pressure`` (from filenames or a
    CSV import, see ``frame_metadata.py``) — each phase is fitted only within
    ``±window`` of that frame's pressure rather than the full ``[p_min, p_max]``.
    The window is ``pressure_sigma_k·σ`` where a per-frame ``pressure_sigma`` is
    known, else ``pressure_window`` (GPa). With ``marker_prior=True`` and no
    metadata pressure, a first pass over the marker-category phases estimates the
    per-frame pressure, which then primes the full pass.

    ``phases`` are Phase objects (resolve names via ``phases.load_library``).
    Requires pymatgen. Prints ``[IDENTIFY] <done> <total>`` progress.
    """
    import h5py  # type: ignore

    if not pymatgen_available():
        raise RuntimeError(
            "pymatgen is required for Step 3a phase matching (pip install pymatgen).")
    phases = [p for p in phases]
    if not phases:
        raise ValueError("No candidate phases supplied — enable some on the Phases tab.")
    # Open-set mode sweeps the whole library, which can include phases with no
    # simulatable structure (e.g. a He pressure medium or a gasket alloy entry).
    # Drop them up front so one un-simulatable phase can't abort the whole run.
    no_struct = [p.name for p in phases if not p.has_structure()]
    phases = [p for p in phases if p.has_structure()]
    if no_struct:
        print(f"[IDENTIFY] skipped (no structure to simulate): {no_struct}", flush=True)
    if not phases:
        raise ValueError(
            "None of the selected phases have a simulatable structure "
            "(need a CIF, or space group + lattice + atoms). Add structure on the "
            "Phases tab, or enable phases that have one.")

    import os
    import shutil

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")
    dst = Path(out_h5).expanduser().resolve() if out_h5 else src

    with h5py.File(str(src), "r") as h5:
        unit = str(h5.attrs.get("unit", ""))
        stored_wl = float(h5.attrs.get("wavelength", 0.0) or 0.0)
        radial_axis = (np.asarray(h5["radial"][:], dtype=float)
                       if "radial" in h5 else None)
        source_reduced = h5.attrs.get("source_reduced", "")
        if isinstance(source_reduced, bytes):
            source_reduced = source_reduced.decode("utf-8", "replace")
        n, peaks_by_frame, amps_by_frame, cerrs_by_frame = _read_good_peaks_by_frame(h5)
        frames = h5.get("frames")
        excluded = (np.asarray(frames["excluded"][:], dtype=bool)
                    if frames is not None and "excluded" in frames else None)
        # Frame pressure prior (+ optional per-frame uncertainty) — written by
        # Step 1 from filenames, or by a frame_metadata CSV import. An explicit
        # pressure_by_frame argument overrides whatever is on disk.
        file_pressure = (np.asarray(frames["pressure"][:], dtype=float)
                         if frames is not None and "pressure" in frames else None)
        file_sigma = (np.asarray(frames["pressure_sigma"][:], dtype=float)
                      if frames is not None and "pressure_sigma" in frames else None)
        file_temperature = (np.asarray(frames["temperature"][:], dtype=float)
                            if frames is not None and "temperature" in frames else None)
        file_names = ([x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray))
                       else str(x) for x in frames["filename"][:]]
                      if frames is not None and "filename" in frames else None)
    if excluded is None or excluded.size != n:
        excluded = np.zeros(n, dtype=bool)

    def _as_n(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=float)
        return a if a.size == n else None

    prior_pressure = _as_n(pressure_by_frame if pressure_by_frame is not None
                           else (file_pressure if use_frame_pressure else None))
    prior_sigma = _as_n(pressure_sigma_by_frame if pressure_sigma_by_frame is not None
                        else (file_sigma if use_frame_pressure else None))
    frame_temperature = _as_n(file_temperature if use_frame_temperature else None)
    if frame_temperature is not None and not np.any(np.isfinite(frame_temperature)):
        frame_temperature = None

    # Wavelength is only needed for a 2theta axis: prefer the value stored at
    # Step 1, then an explicit arg, then the reduced PONI.
    if wavelength is None and stored_wl > 0:
        wavelength = stored_wl
    if wavelength is None and unit.strip().lower() in ("2th_deg", "2th_rad"):
        wavelength = wavelength_from_reduced(source_reduced)
        if wavelength is None:
            raise ValueError(
                f"Data is on a {unit} axis but no wavelength was found. "
                "Re-run Step 1 (which stores it) or pass wavelength explicitly.")

    # Observed d per frame (convert once). Center esd's propagate to d-space so
    # the matcher can widen tolerances per peak (position-only when Step 2
    # predates the *_err columns).
    obs_d_by_frame = [radial_to_d(c, unit, wavelength) if c.size else np.zeros(0)
                      for c in peaks_by_frame]
    obs_err_by_frame = None
    if cerrs_by_frame is not None:
        obs_err_by_frame = [
            radial_err_to_d_err(c, e, unit, wavelength) if c.size else np.zeros(0)
            for c, e in zip(peaks_by_frame, cerrs_by_frame)]

    # Reflection d-range: cover THIS reduction's actual q-range instead of a
    # fixed 1.0 Å guess. Short-wavelength (high-energy) beamlines reach q≈11 Å⁻¹
    # (d≈0.55 Å); the old fixed d_min=1.0 (q≤6.3) left every phase's higher-order
    # lines unmodelled, so real peaks past q≈6.3 — e.g. tungsten 321/400/422, and
    # because W barely compresses they sit at nearly fixed q across the series —
    # were harvested as false "unknowns". radial.max() is the detector's q_max →
    # the smallest visible d. Take the more inclusive of the default and the data
    # edge so low-q data never over-simulates; floor guards a corrupt axis.
    sim_d_min = _SIM_D_MIN_DEFAULT
    if radial_axis is not None and radial_axis.size:
        try:
            d_axis = radial_to_d(radial_axis, unit, wavelength)
            finite = d_axis[np.isfinite(d_axis) & (d_axis > 0)]
            if finite.size:
                sim_d_min = max(0.25, min(_SIM_D_MIN_DEFAULT, float(finite.min())))
        except Exception:
            pass

    # Strongest-N cap scales with reciprocal-space volume (∝ d_min⁻³): the 40 lines
    # tuned for d_min=1.0 must grow with the d-range, or a low-symmetry phase's
    # higher-order lines get crowded out of the top-N — and because the d_min fix
    # extends the *predicted range* to q_max, an in-range-but-unmodelled observed
    # peak counts against precision (see _score_pred). Bounded so a large cell can't
    # explode the predicted comb (which would erode precision on wrong phases).
    sim_max_refl = int(min(400, max(_REFL_CAP_BASE,
                                    round(_REFL_CAP_BASE * (1.0 / sim_d_min) ** 3))))

    workers = resolve_workers(num_workers)
    print(f"[IDENTIFY] {len(phases)} phase(s), {n} frames "
          f"({int(excluded.sum())} excluded), unit={unit or '?'} "
          f"P=[{p_min},{p_max}] GPa workers={workers}", flush=True)
    no_eos = [p.name for p in phases if not p.has_eos()]
    if no_eos:
        print(f"[IDENTIFY] WARNING: no Birch-Murnaghan EOS for {no_eos} — these "
              f"phases are evaluated at ambient only (pressure fixed at 0). Add "
              f"V0/K0/K0' on the Phases tab to fit pressure.", flush=True)

    # Reflections simulated once in the parent (needs pymatgen); workers only
    # score. Guard each simulation so one phase that pymatgen chokes on (bad CIF,
    # exotic element) is skipped with a warning rather than killing the run.
    #
    # This step, not the scoring, dominates on a large library: pymatgen's cost
    # climbs steeply with cell size (measured on a 879-phase library: ~0.4 s at
    # 24 atoms, ~11 s at 56, ~23 s at 315). Serially that is hours before a
    # single frame is scored, and `workers` does not touch it because the fan-out
    # happens afterwards. So simulate in a pool too. Phases are independent here
    # and results are keyed by name, so this changes timing only.
    refl_cache = {}
    t_sim = time.time()
    simulation_payloads = [
        (ph, sim_d_min, sim_max_refl) for ph in phases
    ]
    if workers > 1 and len(phases) > 1:
        simulation_results = process_map_or_serial(
            _simulate_one_payload, simulation_payloads,
            max_workers=workers, label="IDENTIFY")
    else:
        simulation_results = [
            _simulate_one_payload(payload) for payload in simulation_payloads
        ]
    for name, refl, err in simulation_results:
        if refl is not None:
            refl_cache[name] = refl
        else:
            print(f"[IDENTIFY] skipped {name!r}: simulation failed ({err})",
                  flush=True)
    kept = [ph for ph in phases if ph.name in refl_cache]
    print(f"[IDENTIFY] simulated {len(kept)}/{len(phases)} phases in "
          f"{time.time() - t_sim:.0f}s", flush=True)
    phases = kept
    if not phases:
        raise ValueError("No phases could be simulated for identification.")
    refls = [refl_cache[ph.name] for ph in phases]

    # Marker-derived prior: when no metadata pressure exists but the user asked
    # for it, fit only the pressure-marker phases first and adopt the best
    # marker's per-frame pressure as the prior that primes the full pass.
    if marker_prior and (prior_pressure is None or not np.any(np.isfinite(prior_pressure))):
        markers = [ph for ph in phases
                   if ph.category == "marker" and has_pressure_dof(ph)]
        if markers:
            print(f"[IDENTIFY] marker-prior pass over {[m.name for m in markers]}", flush=True)
            est = np.full(n, np.nan, "f8")
            best_conf = np.zeros(n, "f8")
            for j in range(n):
                if excluded[j]:
                    continue
                obs = obs_d_by_frame[j]
                if obs.size == 0:
                    continue
                for m in markers:
                    r = fit_pressure_for_phase(
                        obs, m, refl_cache[m.name], p_min=p_min,
                        p_max=p_max, rel_tol=rel_tol, min_matched=min_matched,
                        obs_err=(obs_err_by_frame[j] if obs_err_by_frame is not None else None),
                        obs_amp=(amps_by_frame[j] if amps_by_frame is not None else None),
                        temperature=(float(frame_temperature[j])
                                     if frame_temperature is not None
                                     and np.isfinite(frame_temperature[j]) else None),
                        intensity_k=intensity_k)
                    if (r["n_matched"] >= min_matched and np.isfinite(r["pressure"])
                            and r["confidence"] > best_conf[j]):
                        best_conf[j] = r["confidence"]
                        est[j] = r["pressure"]
            if np.any(np.isfinite(est)):
                prior_pressure = est
                print(f"[IDENTIFY] marker prior set for "
                      f"{int(np.sum(np.isfinite(est)))}/{n} frames", flush=True)

    # Per-frame search/penalty window (half-width, GPa); None when no prior.
    n_prior = int(np.sum(np.isfinite(prior_pressure))) if prior_pressure is not None else 0
    if prior_pressure is not None and n_prior:
        windows = np.array([
            pressure_window_halfwidth(
                (prior_sigma[j] if (prior_sigma is not None and np.isfinite(prior_sigma[j]))
                 else None),
                pressure_window, pressure_sigma_k)
            for j in range(n)], dtype="f8")
        print(f"[IDENTIFY] pressure prior on {n_prior}/{n} frames "
              f"(window +/-{pressure_window} GPa or {pressure_sigma_k}*sigma)", flush=True)
        # Auto-widen the global search range to cover the metadata pressures (+window).
        # Otherwise a frame whose prior sits outside [p_min, p_max] would clamp its
        # search to the boundary while the prior penalty still measures against the
        # true prior — collapsing confidence to ~0 for an otherwise-correct phase.
        finite_prior = prior_pressure[np.isfinite(prior_pressure)]
        wmax = float(np.nanmax(windows))
        need_lo = max(0.0, float(finite_prior.min()) - wmax)
        need_hi = float(finite_prior.max()) + wmax
        if need_lo < p_min or need_hi > p_max:
            new_lo, new_hi = min(p_min, need_lo), max(p_max, need_hi)
            print(f"[IDENTIFY] WARNING: widening pressure range "
                  f"[{p_min:g},{p_max:g}] -> [{new_lo:g},{new_hi:g}] GPa to cover the "
                  f"frame-metadata pressures (+window). Frames responsible:", flush=True)
            for line in prior_range_offenders(prior_pressure, windows,
                                              p_min, p_max, file_names):
                print(f"[IDENTIFY] {line}", flush=True)
            p_min, p_max = new_lo, new_hi
    else:
        prior_pressure = None
        windows = None

    results: Dict[str, Dict[str, np.ndarray]] = {
        ph.name: {"pressure": np.full(n, np.nan, "f8"), "score": np.zeros(n, "f8"),
                  "confidence": np.zeros(n, "f8"), "recall": np.zeros(n, "f8"),
                  "precision": np.zeros(n, "f8"), "n_matched": np.zeros(n, "i4"),
                  "prior_penalty": np.ones(n, "f8"),
                  "intensity_corr": np.full(n, np.nan, "f8")}
        for ph in phases}

    def _absorb(a, chunk_res):
        for name, rr in chunk_res.items():
            for k, v in rr.items():
                results[name][k][a:a + len(v)] = v

    def _slice(arr, a, b):
        return arr[a:b] if arr is not None else None

    if workers > 1 and n > 1:
        ranges = chunk_ranges(n, workers)
        payloads = [(phases, refls, obs_d_by_frame[a:b], excluded[a:b],
                     _slice(prior_pressure, a, b), _slice(windows, a, b),
                     p_min, p_max, rel_tol, min_matched,
                     _slice(amps_by_frame, a, b), _slice(obs_err_by_frame, a, b),
                     _slice(frame_temperature, a, b), intensity_k)
                    for a, b in ranges]
        done = 0
        chunk_results = process_map_or_serial(
            _identify_chunk, payloads, max_workers=workers, label="IDENTIFY")
        for (a, b), chunk_res in zip(ranges, chunk_results):
            _absorb(a, chunk_res)
            done += (b - a)
            print(f"[IDENTIFY] {done} {n}", flush=True)
    else:
        _absorb(0, _identify_chunk((phases, refls, obs_d_by_frame, excluded,
                                    prior_pressure, windows,
                                    p_min, p_max, rel_tol, min_matched,
                                    amps_by_frame, obs_err_by_frame,
                                    frame_temperature, intensity_k)))
        print(f"[IDENTIFY] {n} {n}", flush=True)

    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            if "identify" in o:
                del o["identify"]
            gid = o.create_group("identify")
            write_step_provenance(o, "identify",
                                  tool="seriesxrd.analysis.identify",
                                  schema_version=SCHEMA_VERSION)
            gid.attrs.update({
                "schema_version": SCHEMA_VERSION,
                "seriesxrd_version": VERSION, "unit": unit,
                "wavelength": float(wavelength) if wavelength else 0.0,
                "p_min": float(p_min), "p_max": float(p_max), "rel_tol": float(rel_tol),
                "phases": ", ".join(p.name for p in phases),
                "pressure_window": float(pressure_window),
                "pressure_sigma_k": float(pressure_sigma_k),
                "min_matched": int(min_matched),
                "seen_conf": float(seen_conf),
                "n_pressure_prior": int(n_prior),
                "intensity_k": float(intensity_k),
                "sim_d_min": float(sim_d_min),
                "sim_max_refl": int(sim_max_refl),
                "n_temperature": int(np.sum(np.isfinite(frame_temperature))
                                     if frame_temperature is not None else 0),
            })
            for ph in phases:
                g = gid.create_group(_h5_safe(ph.name))
                # prior_penalized: did the pressure prior actually pull this phase's
                # confidence down on any live frame? (penalty noticeably below 1.)
                pp_live = results[ph.name]["prior_penalty"][~excluded] if (~excluded).any() \
                    else results[ph.name]["prior_penalty"]
                g.attrs.update({"name": ph.name, "n_pred": int(refl_cache[ph.name][0].size),
                                "has_eos": bool(ph.has_eos()), "category": ph.category,
                                "pressure_model": pressure_model(ph),
                                "pressure_assumption": pressure_assumption(ph),
                                "prior_penalized": bool(np.any(pp_live < 0.999))})
                for k, v in results[ph.name].items():
                    g.create_dataset(k, data=v)
                # Cache the ambient reflection list so GUI overlays (reflection
                # tracks / phase layers) never need to re-run pymatgen — that
                # simulation, on the main thread, froze the app for low-symmetry
                # phases. d-spacings are wavelength-independent; tracks scale them
                # by the per-frame pressure.
                d0c, wc, hklc = refl_cache[ph.name]
                g.create_dataset("refl_d", data=np.asarray(d0c, "f8"))
                g.create_dataset("refl_w", data=np.asarray(wc, "f8"))
                g.create_dataset(
                    "refl_hkl",
                    data=np.array([str(h) for h in hklc], dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    # Per-phase summary. "Seen" requires both a modest confidence bar (a DAC
    # pattern rarely shows every strong line) AND minimum reflection evidence
    # (≥ min_matched one-to-one matches), so a phase can't be called present off
    # one or two coincidental peaks. The richer stats let the user judge partial
    # matches instead of collapsing them to a single 0.
    SEEN_CONF = max(0.0, min(1.0, float(seen_conf)))
    summary = {}
    live = ~excluded
    for ph in phases:
        res = results[ph.name]
        conf = res["confidence"]
        nm = res["n_matched"]
        pp = res["prior_penalty"]
        conf_live = conf[live] if live.any() else conf
        seen = (conf > SEEN_CONF) & (nm >= int(min_matched)) & live
        pp_live = pp[live] if live.any() else pp
        summary[ph.name] = {
            "mean_confidence": float(np.mean(conf_live)) if conf_live.size else 0.0,
            "max_confidence": float(np.max(conf_live)) if conf_live.size else 0.0,
            "max_recall": float(np.max(res["recall"][live])) if live.any() else 0.0,
            "max_precision": float(np.max(res["precision"][live])) if live.any() else 0.0,
            "seen_conf": SEEN_CONF,
            "seen_min_matched": int(min_matched),
            "n_frames_seen": int(np.sum(seen)),
            "n_frames_matched": int(np.sum((nm > 0) & live)),
            "max_matched": int(np.max(nm)) if nm.size else 0,
            "pressure_median": float(np.nanmedian(res["pressure"][seen])) if np.any(seen) else float("nan"),
            "has_eos": bool(ph.has_eos()),
            "pressure_model": pressure_model(ph),
            "pressure_assumption": pressure_assumption(ph),
            "prior_penalized": bool(np.any(pp_live < 0.999)),
            "n_frames_penalized": int(np.sum(pp_live < 0.999)),
        }

    manifest = {
        **manifest_provenance("seriesxrd.analysis.identify", SCHEMA_VERSION),
        "source": str(src), "out_h5": str(dst),
        "n_frames": int(n), "unit": unit,
        "wavelength": float(wavelength) if wavelength else None,
        "p_min": float(p_min), "p_max": float(p_max), "rel_tol": float(rel_tol),
        "phases": [p.name for p in phases], "summary": summary,
    }
    print(f"[IDENTIFY] done -> {dst}", flush=True)
    return manifest
