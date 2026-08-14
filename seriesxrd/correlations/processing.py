"""Headless correlation processing for an Analysis HDF5.

The implementation intentionally fixes the intensity transform to Log-squared.
Every compared frame shares one pooled-positive scale and one epsilon estimated
from the original signed source. ROI comparisons use positive intensity, while
window comparisons preserve the signed source until the deliberate squaring
step::

    z_roi = clip(max(I, 0) / scale, 0, 1)
    z_window = clip(I / scale, -1, 1)
    Log2(z) = log1p(z**2 / epsilon) / log1p(1 / epsilon)

Location maps are geometric and therefore do not use transformed intensity.
Waterfall traces use the original positive source; only their colors come from
the positive Log-squared ROI-area map.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.config import VERSION, now_iso, read_json, write_json
from ..core.provenance import manifest_provenance, write_provenance

SCHEMA_VERSION = "1"
TOOL = "seriesxrd.correlations"
SAMPLE_TYPES = ("powder", "single_crystal")
DEFAULT_SCALE_QUANTILE = 0.995
DEFAULT_EPSILON_FLOOR = 1.0e-12
DEFAULT_WINDOW_WIDTH = 5.0
DEFAULT_WINDOW_STEP = 1.0
DEFAULT_LOCATION_TOLERANCE = 0.02
ROI_PROFILE_POINTS = 65
WINDOW_PROFILE_POINTS = 64


@dataclass(frozen=True)
class TransformParameters:
    """Frozen parameters shared by every intensity comparison in one run."""

    method: str
    scale: float
    scale_quantile: float
    noise_floor: float
    epsilon: float
    epsilon_floor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "scale": self.scale,
            "scale_quantile": self.scale_quantile,
            "noise_floor": self.noise_floor,
            "epsilon": self.epsilon,
            "epsilon_floor": self.epsilon_floor,
            "formula": (
                "z=clip(max(I,0)/scale,0,1); "
                "log1p(z**2/epsilon)/log1p(1/epsilon)"
            ),
            "signed_formula": (
                "z=clip(I/scale,-1,1); "
                "log1p(z**2/epsilon)/log1p(1/epsilon)"
            ),
        }


@dataclass(frozen=True)
class PeakTable:
    """All retained peak observations; rows are never collapsed by track."""

    source_index: np.ndarray
    frame_row: np.ndarray
    original_frame: np.ndarray
    local_peak: np.ndarray
    center: np.ndarray
    width: np.ndarray
    half_width: np.ndarray
    area: np.ndarray
    pressure: np.ndarray
    track: np.ndarray

    @property
    def size(self) -> int:
        return int(self.center.size)


def reusable_artifact(
    artifact: Path,
    analysis_path: Path,
    compute_config: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Whether an existing artifact can stand in for recomputing it.

    No new bookkeeping: ``/provenance`` already records the input hash and
    the effective config, which is exactly what decides this. Render-only
    settings are ignored, so asking for different figures never forces the
    numbers to be recomputed.
    """

    import h5py  # type: ignore

    from ..core.provenance import file_fingerprint

    if not Path(artifact).is_file():
        return False, "no existing artifact"
    try:
        with h5py.File(str(artifact), "r") as h5:
            group = h5.get("provenance")
            if group is None:
                return False, "artifact has no /provenance"
            stored = json.loads(str(group.attrs.get("config_json", "{}")))
            stored_hash = str(group.attrs.get("input_analysis_sha256", ""))
    except (OSError, ValueError) as exc:
        return False, f"artifact unreadable ({exc})"

    current_hash = str(file_fingerprint(analysis_path).get("sha256", ""))
    if not stored_hash or not current_hash:
        return False, "input hash unavailable"
    if stored_hash != current_hash:
        return False, "the Analysis HDF5 changed"
    for key, value in compute_config.items():
        if stored.get(key) != value:
            return False, f"{key} changed ({stored.get(key)!r} -> {value!r})"
    return True, "reused the existing artifact"


def _export_from_existing(
    artifact: Path,
    destination: Path,
    prior: Mapping[str, Any],
    *,
    plot_families: Sequence[str],
    max_anchor_plots: Optional[int],
    export_csv: bool,
    export_matrix_csv: bool,
) -> Dict[str, Any]:
    """Re-drive export against an artifact that is already up to date.

    The previous run's manifest is carried forward and only its export
    fields are refreshed, so provenance of the numbers stays attached to
    the run that produced them.
    """

    import h5py  # type: ignore

    base: Dict[str, Any] = dict(prior)
    if not base:
        # No manifest survived; rebuild the summary from the artifact.
        with h5py.File(str(artifact), "r") as h5:
            base = {
                **manifest_provenance(TOOL, SCHEMA_VERSION),
                "correlations_h5": str(artifact),
                "out_dir": str(destination),
                **{
                    key: _decode(h5.attrs[key])
                    if isinstance(h5.attrs[key], (bytes, str))
                    else h5.attrs[key].item()
                    for key in (
                        "sample_type", "unit", "source_requested",
                        "source_resolved", "order_by", "order_label",
                        "n_frames", "n_peaks", "n_windows",
                    )
                    if key in h5.attrs
                },
                "manifest_rebuilt_from_artifact": True,
            }

    plot_files: Sequence[str] = ()
    if plot_families:
        from .plots import render_all

        plot_files = render_all(
            artifact,
            destination / "heatmaps",
            max_anchor_plots=max_anchor_plots,
            families=plot_families,
            resume=True,
        )
    csv_files: list = []
    if export_csv or export_matrix_csv:
        from . import export as _export

        csv_dir = destination / "csv"
        if export_csv:
            csv_files.extend(_export.export_summary_csvs(artifact, csv_dir))
        if export_matrix_csv:
            csv_files.extend(_export.export_matrices(artifact, csv_dir))

    manifest = base
    manifest.update(
        {
            "resumed": True,
            "resumed_at": now_iso(),
            "plots": list(plot_families),
            "anchor_plot_cap": (
                None if max_anchor_plots is None else int(max_anchor_plots)
            ),
            "plots_written": len(plot_files),
            "plot_files": list(plot_files),
            "csv_files": [
                path.relative_to(destination).as_posix() for path in csv_files
            ],
        }
    )
    sample = str(manifest.get("sample_type", "powder"))
    write_json(destination / f"manifest_{sample}.json", manifest)
    return manifest


def resolve_plot_families(
    plots: "Optional[Sequence[str]]" = None,
    make_plots: Optional[bool] = None,
) -> Tuple[str, ...]:
    """Which figure families a run should write to PNG.

    Bulk rendering is **off by default**: every number is already in the
    artifact and the GUI draws figures on demand, so a routine run has no
    reason to emit thousands of files. Ask for them explicitly with
    ``plots=("all",)`` or a list of families. ``make_plots`` is the
    deprecated boolean spelling and is honored when ``plots`` is not given.
    """

    from .plots import FAMILIES

    if plots is None:
        return tuple(FAMILIES) if make_plots else ()
    names = tuple(str(name).strip().lower() for name in plots if str(name).strip())
    if not names or "none" in names:
        return ()
    if "all" in names:
        return tuple(FAMILIES)
    unknown = [name for name in names if name not in FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown figure families {unknown}; choose from "
            f"{['all', 'none', *FAMILIES]}"
        )
    # De-duplicate while keeping the canonical render order.
    return tuple(name for name in FAMILIES if name in names)


def _progress(done: int, total: int) -> None:
    """House 3-token progress protocol; GUIs parse ``[CORRELATIONS] d t``."""
    print(f"[CORRELATIONS] {int(done)} {int(total)}", flush=True)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _estimate_noise_floor(values: np.ndarray) -> float:
    """Robust white-noise estimate from first differences, pooled by frame.

    Deliberately NOT consolidated with peaks.mad_sigma or
    stackplot.mad_noise: those are value-MADs (they include background
    structure), while a first-difference MAD isolates the white-noise sigma
    that the Log2 epsilon floor needs; staying local also keeps this module
    import-light.
    """

    array = np.atleast_2d(np.asarray(values, dtype=float))
    estimates = []
    for row in array:
        pair_ok = np.isfinite(row[:-1]) & np.isfinite(row[1:])
        differences = np.diff(row)[pair_ok]
        if differences.size < 3:
            continue
        middle = float(np.median(differences))
        sigma = 1.4826 * float(np.median(np.abs(differences - middle))) / math.sqrt(2.0)
        if np.isfinite(sigma) and sigma >= 0.0:
            estimates.append(sigma)
    return float(np.median(estimates)) if estimates else 0.0


def _bounded_log_squared(z: np.ndarray, epsilon: float) -> np.ndarray:
    """``log1p(z^2/eps) / log1p(1/eps)``, saturating exactly at 1.

    The normalization constant exists precisely so that ``|z| = 1`` maps to
    1, but the numerator goes through numpy's (possibly SIMD) ``log1p`` and
    the denominator through the scalar one, and the two implementations may
    disagree by an ULP for the same argument — which they do on some numpy
    builds, returning 0.9999999999999998 at saturation. Pinning the
    saturation point keeps the documented [0, 1] bound exact and keeps the
    artifact reproducible across numpy versions rather than only within one.
    """

    scaled = np.log1p((z * z) / epsilon) / math.log1p(1.0 / epsilon)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.where(np.abs(z) >= 1.0, 1.0, scaled)


def log_squared_transform(
    values: np.ndarray,
    *,
    scale: Optional[float] = None,
    noise_floor: Optional[float] = None,
    scale_quantile: float = DEFAULT_SCALE_QUANTILE,
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
) -> Tuple[np.ndarray, TransformParameters]:
    """Apply the fixed, bounded Log-squared intensity transform.

    ``scale`` defaults to one pooled positive finite quantile over *all* input
    rows. ``noise_floor`` defaults to a first-difference MAD estimate. Masked or
    non-finite values become NaN; finite negative values map to exact zero.
    """

    array = np.asarray(values, dtype=float)
    if not (0.0 < float(scale_quantile) <= 1.0):
        raise ValueError("scale_quantile must be in (0, 1]")
    if not np.isfinite(epsilon_floor) or epsilon_floor <= 0.0:
        raise ValueError("epsilon_floor must be finite and positive")
    if scale is None:
        positive = _finite_positive(array)
        if positive.size == 0:
            raise ValueError("intensity source has no positive finite values")
        scale = float(np.quantile(positive, float(scale_quantile)))
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    if noise_floor is None:
        noise_floor = _estimate_noise_floor(array)
    if not np.isfinite(noise_floor) or float(noise_floor) < 0.0:
        raise ValueError("noise_floor must be finite and non-negative")

    epsilon = float(max((float(noise_floor) / float(scale)) ** 2, epsilon_floor))
    output = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array)
    z = np.clip(np.maximum(array[valid], 0.0) / float(scale), 0.0, 1.0)
    output[valid] = _bounded_log_squared(z, epsilon)
    params = TransformParameters(
        method="log_squared",
        scale=float(scale),
        scale_quantile=float(scale_quantile),
        noise_floor=float(noise_floor),
        epsilon=epsilon,
        epsilon_floor=float(epsilon_floor),
    )
    return output, params


def _signed_log_squared_transform(
    values: np.ndarray,
    parameters: TransformParameters,
) -> np.ndarray:
    """Apply Log-squared to signed values with an already-fitted transform.

    The sign is retained through normalization and clipping, then deliberately
    erased by squaring. Reusing the parameters guarantees that ROI and window
    products share the same pooled scale and noise-derived epsilon.
    """

    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array)
    z = np.clip(array[valid] / parameters.scale, -1.0, 1.0)
    output[valid] = _bounded_log_squared(z, parameters.epsilon)
    return output


def integrated_iou(
    left: np.ndarray,
    right: np.ndarray,
    coordinate: Optional[np.ndarray] = None,
) -> float:
    """Continuous min/max IoU for two finite, non-negative profiles.

    A finite zero denominator (both profiles carry no signal) returns 0.0,
    matching :func:`directional_anchor_iou`; NaN is reserved for structural
    invalidity (fewer than two finite samples or a non-finite integral).
    """

    from scipy.integrate import trapezoid

    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("profiles must have equal length >= 2")
    x = np.arange(a.size, dtype=float) if coordinate is None else np.asarray(
        coordinate, dtype=float
    ).reshape(-1)
    if x.shape != a.shape:
        raise ValueError("coordinate must match profile shape")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(x)
    if np.count_nonzero(valid) < 2:
        return math.nan
    aa, bb, xx = a[valid], b[valid], x[valid]
    if np.any(aa < 0.0) or np.any(bb < 0.0):
        raise ValueError("ROI profiles must be non-negative")
    order = np.argsort(xx, kind="stable")
    aa, bb, xx = aa[order], bb[order], xx[order]
    numerator = float(trapezoid(np.minimum(aa, bb), xx))
    denominator = float(trapezoid(np.maximum(aa, bb), xx))
    if not np.isfinite(denominator):
        return math.nan
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def directional_anchor_iou(
    radial: np.ndarray,
    anchor_profile: np.ndarray,
    target_profile: np.ndarray,
    *,
    anchor_support: Tuple[float, float],
    target_support: Tuple[float, float],
) -> float:
    """Powder ROI IoU on the anchor's absolute native-radial support.

    The target is exactly zero outside its own physical support.  The two
    profiles are never recentered or width-normalized, so ``A -> B`` can differ
    from ``B -> A``. A finite zero denominator denotes zero signal and returns
    zero; NaN is reserved for a structurally unusable profile/support.
    """

    from scipy.integrate import trapezoid

    x = np.asarray(radial, dtype=float).reshape(-1)
    anchor = np.asarray(anchor_profile, dtype=float).reshape(-1)
    target = np.asarray(target_profile, dtype=float).reshape(-1)
    if x.shape != anchor.shape or x.shape != target.shape or x.size < 2:
        raise ValueError("radial and directional ROI profiles must be equal-shape 1D")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("radial must be finite and strictly increasing")
    a_lo, a_hi = map(float, anchor_support)
    t_lo, t_hi = map(float, target_support)
    if not all(np.isfinite(v) for v in (a_lo, a_hi, t_lo, t_hi)):
        return math.nan
    if a_lo >= a_hi or t_lo >= t_hi:
        return math.nan
    if a_lo < x[0] or a_hi > x[-1]:
        return math.nan
    finite_a = np.isfinite(anchor)
    finite_t = np.isfinite(target)
    if np.count_nonzero(finite_a) < 2 or np.count_nonzero(finite_t) < 2:
        return math.nan
    # A masked/native NaN bin is structural, not a value this stage may
    # silently reconstruct (same rule as the window path): inside the anchor
    # support it voids the anchor, inside the support overlap it voids the
    # target. The interpolation below then only ever draws boundary values
    # from bins outside the integration domain, which is legitimate.
    inside_anchor = (x > a_lo) & (x < a_hi)
    if np.any(inside_anchor & ~finite_a):
        return math.nan
    overlap_lo = max(a_lo, t_lo)
    overlap_hi = min(a_hi, t_hi)
    if overlap_lo < overlap_hi:
        inside_overlap = (x > overlap_lo) & (x < overlap_hi)
        if np.any(inside_overlap & ~finite_t):
            return math.nan

    knots = [a_lo, a_hi]
    knots.extend(x[(x > a_lo) & (x < a_hi)].tolist())
    if a_lo < t_lo < a_hi:
        knots.append(t_lo)
    if a_lo < t_hi < a_hi:
        knots.append(t_hi)
    grid = np.unique(np.asarray(knots, dtype=float))
    if grid.size < 2:
        return math.nan
    # Integrate one interval at a time. The physical support creates a hard
    # discontinuity at its boundary; treating that jump as a line segment would
    # add a spurious half-bin triangle. Midpoint support selection keeps the
    # discontinuity measure-zero, while an extra knot solves any crossing of
    # the two piecewise-linear profiles exactly within the interval.
    numerator = 0.0
    denominator = 0.0
    for left, right in zip(grid[:-1], grid[1:], strict=True):
        if right <= left:
            continue
        local_x = np.asarray([left, right], dtype=float)
        anchor_values = np.clip(
            np.interp(local_x, x[finite_a], anchor[finite_a]), 0.0, None
        )
        midpoint = 0.5 * (float(left) + float(right))
        if t_lo < midpoint < t_hi:
            target_values = np.clip(
                np.interp(local_x, x[finite_t], target[finite_t]), 0.0, None
            )
        else:
            target_values = np.zeros(2, dtype=float)
        difference = anchor_values - target_values
        if difference[0] * difference[1] < 0.0:
            fraction = float(-difference[0] / (difference[1] - difference[0]))
            crossing = float(left + fraction * (right - left))
            local_x = np.asarray([left, crossing, right], dtype=float)
            anchor_values = np.interp(
                local_x, [left, right], anchor_values
            )
            target_values = np.interp(
                local_x, [left, right], target_values
            )
        numerator += float(trapezoid(np.minimum(anchor_values, target_values), local_x))
        denominator += float(trapezoid(np.maximum(anchor_values, target_values), local_x))
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return math.nan
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def relative_feature_similarity(
    left: np.ndarray | float,
    right: np.ndarray | float,
) -> np.ndarray:
    """Single-crystal scalar ROI similarity ``min/max``; both-zero is zero.

    Two ROIs that both carry no signal share absence, not similarity -- a
    masked or empty region must never score as a perfect match. Zero signal
    in a finite denominator therefore scores 0.0, the same convention as
    :func:`directional_anchor_iou`; NaN is reserved for non-finite features.
    """

    a, b = np.broadcast_arrays(
        np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    )
    result = np.full(a.shape, np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0.0) & (b >= 0.0)
    both_zero = valid & (a == 0.0) & (b == 0.0)
    result[both_zero] = 0.0
    positive = valid & ~both_zero
    maximum = np.maximum(a, b)
    result[positive] = np.minimum(a[positive], b[positive]) / maximum[positive]
    return result


def location_similarity(
    left: np.ndarray | float,
    right: np.ndarray | float,
    tolerance: float = DEFAULT_LOCATION_TOLERANCE,
) -> np.ndarray:
    """Linear location similarity in the native radial unit."""

    if not np.isfinite(tolerance) or float(tolerance) <= 0.0:
        raise ValueError("location tolerance must be finite and positive")
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    result = 1.0 - np.abs(a - b) / float(tolerance)
    result = np.clip(result, 0.0, 1.0)
    return np.asarray(np.where(np.isfinite(a - b), result, np.nan), dtype=float)


def _decode(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(
        value, (bytes, bytearray)
    ) else str(value)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        return math.nan
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.nan
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return math.nan
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _pairwise_correlations(rows: np.ndarray) -> np.ndarray:
    """All-pairs Pearson matrix, matching :func:`_pearson` per cell.

    One centered-rows Gram matrix replaces the O(n^2) Python loop; rows with
    any non-finite value stay NaN everywhere, and a pair whose norm product
    is not usable stays NaN, exactly as the scalar reference decides.
    """

    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlation input must be a two-dimensional array")
    count, length = values.shape
    result = np.full((count, count), np.nan, dtype=float)
    if length < 2 or count == 0:
        return result
    finite = np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        return result
    index = np.nonzero(finite)[0]
    centered = values[index] - values[index].mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    gram = centered @ centered.T
    denominator = norms[:, None] * norms[None, :]
    bad_pair = ~np.isfinite(denominator) | (
        denominator <= np.finfo(float).eps
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.clip(gram / denominator, -1.0, 1.0)
    scores[bad_pair] = np.nan
    result[np.ix_(index, index)] = scores
    return result


def _acf_batch(rows: np.ndarray) -> np.ndarray:
    """Row-batched :func:`_acf`: one FFT call along the last axis."""

    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("ACF batch input must be a two-dimensional array")
    count, size = values.shape
    output = np.full((count, max(size - 1, 0)), np.nan, dtype=float)
    if size < 3 or count == 0:
        return output
    eps = np.finfo(float).eps
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ok = np.all(np.isfinite(values), axis=1)
        centered = values - np.mean(values, axis=1, keepdims=True)
        scale = np.std(centered, axis=1)
        magnitude = np.maximum(np.max(np.abs(centered), axis=1), 1.0)
        ok &= np.isfinite(scale) & (scale > eps * magnitude)
        standardized = centered / np.where(ok, scale, 1.0)[:, None]
        standardized = np.where(ok[:, None], standardized, 0.0)
        spectrum = np.fft.rfft(standardized, n=2 * size, axis=1)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=2 * size, axis=1
        )[:, :size]
        zero_lag = correlation[:, 0]
        ok &= np.isfinite(zero_lag) & (zero_lag > eps)
        lags = correlation[:, 1:] / np.where(ok, zero_lag, 1.0)[:, None]
        ok &= np.all(np.isfinite(lags) | ~ok[:, None], axis=1)
        lags = lags - np.mean(lags, axis=1, keepdims=True)
        fingerprint_scale = np.std(lags, axis=1)
        ok &= np.isfinite(fingerprint_scale) & (fingerprint_scale > 1.0e-12)
        lags = lags / np.where(ok, fingerprint_scale, 1.0)[:, None]
    output[ok] = lags[ok]
    return output


def _acf(values: np.ndarray) -> np.ndarray:
    """Standardized full positive-lag autocorrelation fingerprint via FFT."""

    row = np.asarray(values, dtype=float).reshape(-1)
    return _acf_batch(row[None, :])[0]


def _window_bounds(
    radial: np.ndarray,
    width: float,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("window_width must be finite and positive")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("window_step must be finite and positive")
    lo, hi = float(radial[0]), float(radial[-1])
    span = hi - lo
    if span <= 0.0:
        raise ValueError("radial axis has no positive span")
    tolerance = 1.0e-12 * max(1.0, abs(span))
    if width > span + tolerance:
        raise ValueError(
            "window_width cannot exceed the selected radial span "
            f"({width:g} > {span:g})"
        )
    if width >= span - tolerance:
        return np.asarray([lo]), np.asarray([hi])
    count = int(math.floor((span - width) / step + 1.0 + 1.0e-12))
    starts = lo + step * np.arange(max(count, 1), dtype=float)
    ends = starts + width
    keep = ends <= hi + 1.0e-10 * max(1.0, abs(hi))
    return starts[keep], np.minimum(ends[keep], hi)


def _resample_windows(
    radial: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    points: int = WINDOW_PROFILE_POINTS,
) -> np.ndarray:
    frames = values.shape[0]
    output = np.full((frames, starts.size, int(points)), np.nan, dtype=float)
    rows = np.asarray(values, dtype=float)
    for window, (start, end) in enumerate(zip(starts, ends, strict=True)):
        grid = np.linspace(float(start), float(end), int(points))
        left_index = max(int(np.searchsorted(radial, start, side="right")) - 1, 0)
        right_index = min(
            int(np.searchsorted(radial, end, side="left")), radial.size - 1
        )
        sub_x = radial[left_index : right_index + 1]
        if sub_x.size < 2:
            continue
        # One interpolation-weight vector serves every frame of this window.
        j = np.clip(
            np.searchsorted(sub_x, grid, side="right") - 1, 0, sub_x.size - 2
        )
        weight = (grid - sub_x[j]) / (sub_x[j + 1] - sub_x[j])
        sampled = (
            rows[:, left_index + j] * (1.0 - weight)
            + rows[:, left_index + j + 1] * weight
        )
        sampled[:, (grid < sub_x[0]) | (grid > sub_x[-1])] = np.nan
        # A masked/native NaN bin is structural, not a value that the
        # correlation stage may silently reconstruct by interpolation.
        row_ok = np.all(
            np.isfinite(rows[:, left_index : right_index + 1]), axis=1
        )
        output[row_ok, window] = sampled[row_ok]
    return output


def compute_window_correlations(
    radial: np.ndarray,
    transformed: np.ndarray,
    *,
    window_width: float = DEFAULT_WINDOW_WIDTH,
    window_step: float = DEFAULT_WINDOW_STEP,
) -> Dict[str, np.ndarray]:
    """Direct/ACF across-frame and ACF within-frame correlations."""

    x = np.asarray(radial, dtype=float)
    y = np.asarray(transformed, dtype=float)
    if y.ndim != 2 or y.shape[1] != x.size:
        raise ValueError("transformed patterns must have shape (frames, radial bins)")
    starts, ends = _window_bounds(x, float(window_width), float(window_step))
    signals = _resample_windows(x, y, starts, ends)
    frames, windows, points = signals.shape
    acf_features = _acf_batch(
        signals.reshape(frames * windows, points)
    ).reshape(frames, windows, max(points - 1, 0))

    across_direct = np.full((windows, frames, frames), np.nan, dtype=float)
    across_acf = np.full_like(across_direct, np.nan)
    for window in range(windows):
        across_direct[window] = _pairwise_correlations(signals[:, window, :])
        across_acf[window] = _pairwise_correlations(acf_features[:, window, :])

    within_acf = np.full((frames, windows, windows), np.nan, dtype=float)
    for frame in range(frames):
        within_acf[frame] = _pairwise_correlations(acf_features[frame])
    return {
        "start": starts,
        "end": ends,
        "signals": signals,
        "acf_features": acf_features,
        "across_direct": across_direct,
        "across_acf": across_acf,
        "within_acf": within_acf,
    }


def _read_frames(h5, n_frames: int, order_by: str = "frame") -> Dict[str, Any]:
    frames = h5.get("frames")
    names = [f"frame_{index:04d}" for index in range(n_frames)]
    pressure = np.full(n_frames, np.nan, dtype=float)
    excluded = np.zeros(n_frames, dtype=bool)
    if frames is not None:
        if "filename" in frames:
            raw = frames["filename"][:]
            if len(raw) != n_frames:
                raise ValueError("/frames/filename length does not match pattern frames")
            names = [_decode(value) for value in raw]
        if "pressure" in frames:
            pressure = np.asarray(frames["pressure"][:], dtype=float)
            if pressure.shape != (n_frames,):
                raise ValueError("/frames/pressure length does not match pattern frames")
        if "excluded" in frames:
            excluded = np.asarray(frames["excluded"][:], dtype=bool)
            if excluded.shape != (n_frames,):
                raise ValueError("/frames/excluded length does not match pattern frames")
    keep = ~excluded
    if not np.any(keep):
        raise ValueError("all analysis frames are excluded")

    from ..analysis.series import tracking_values

    axis_key, axis_values, axis_label = tracking_values(h5, order_by, n_frames)
    original_index = np.nonzero(keep)[0].astype(np.int32)
    order_values = np.asarray(axis_values, dtype=float)[original_index]
    if axis_key != "frame":
        # Stable physical ordering: finite axis values ascending, frames
        # without the metadata last, original file order as the tie-break.
        order = np.lexsort(
            (original_index, order_values, np.isnan(order_values))
        )
        original_index = original_index[order]
        order_values = order_values[order]
    old_to_new = np.full(n_frames, -1, dtype=np.int32)
    old_to_new[original_index] = np.arange(original_index.size, dtype=np.int32)
    return {
        "keep": keep,
        "original_index": original_index,
        "old_to_new": old_to_new,
        "filename": [names[index] for index in original_index],
        "pressure": pressure[original_index],
        "excluded_count": int(np.count_nonzero(excluded)),
        "order_by": axis_key,
        "order_values": order_values,
        "order_label": axis_label,
    }


def _dataset_or_default(group, name: str, count: int, default: float) -> np.ndarray:
    if name not in group:
        return np.full(count, default, dtype=float)
    values = np.asarray(group[name][:], dtype=float).reshape(-1)
    if values.size != count:
        raise ValueError(f"/{group.name.strip('/')}/{name} has inconsistent length")
    return values


def _read_peaks(
    h5,
    sample_type: str,
    frames: Mapping[str, Any],
    radial: np.ndarray,
) -> PeakTable:
    if sample_type == "powder":
        group = h5.get("peaks")
        if group is None or "frame" not in group or "center" not in group:
            raise ValueError(
                "powder correlations need /peaks/{frame,center}; "
                "run Analysis Step 2 (peak fitting) first"
            )
        center_name, width_name = "center", "fwhm"
    else:
        spots = h5.get("spots")
        group = spots.get("obs") if spots is not None else None
        if group is None or "frame" not in group or "q" not in group:
            raise ValueError(
                "single_crystal correlations need /spots/obs/{frame,q}; "
                "run SeriesXRD spot tracking first"
            )
        center_name, width_name = "q", "q_width"

    source_frame = np.asarray(group["frame"][:], dtype=int).reshape(-1)
    center = np.asarray(group[center_name][:], dtype=float).reshape(-1)
    count = center.size
    if source_frame.size != count:
        raise ValueError(f"/{group.name.strip('/')} frame/center lengths differ")
    width = _dataset_or_default(group, width_name, count, math.nan)
    area = _dataset_or_default(group, "area", count, math.nan)
    if np.all(~np.isfinite(area)) and "intensity" in group:
        area = _dataset_or_default(group, "intensity", count, math.nan)
    track = (
        np.asarray(group["track"][:], dtype=int).reshape(-1)
        if "track" in group
        else np.full(count, -1, dtype=int)
    )
    if track.size != count:
        raise ValueError(f"/{group.name.strip('/')} track length differs")
    good = np.ones(count, dtype=bool)
    if sample_type == "powder" and "flag" in group:
        flag = np.asarray(group["flag"][:], dtype=int).reshape(-1)
        if flag.size != count:
            raise ValueError("/peaks/flag length differs")
        good &= flag == 0
    good &= (source_frame >= 0) & (source_frame < frames["old_to_new"].size)
    mapped = np.full(count, -1, dtype=int)
    valid_source = (source_frame >= 0) & (source_frame < frames["old_to_new"].size)
    mapped[valid_source] = frames["old_to_new"][source_frame[valid_source]]
    good &= mapped >= 0
    good &= np.isfinite(center) & (center >= radial[0]) & (center <= radial[-1])

    source_index = np.nonzero(good)[0]
    if source_index.size == 0:
        raise ValueError("no usable all-peak observations remain inside the radial range")
    frame_row = mapped[good]
    centers = center[good]
    widths = np.abs(width[good])
    spacing = float(np.median(np.diff(radial)))
    fallback_width = max(2.0 * spacing, np.finfo(float).eps)
    widths[~np.isfinite(widths) | (widths <= 0.0)] = fallback_width
    if sample_type == "powder":
        # Frozen powder definition: [center - 0.75*width, center + 0.75*width].
        half_width = 0.75 * widths
    else:
        # The raw-pixel prototype's curated ROI is unavailable in an Analysis
        # HDF5. Use one fitted /spots/obs/q_width on each side as the explicit
        # one-dimensional support approximation.
        half_width = widths.copy()

    order = np.lexsort((source_index, centers, frame_row))
    source_index = source_index[order].astype(np.int32)
    frame_row = frame_row[order].astype(np.int32)
    centers = centers[order]
    widths = widths[order]
    half_width = half_width[order]
    original_frame = frames["original_index"][frame_row].astype(np.int32)
    local_peak = np.zeros(order.size, dtype=np.int32)
    running: Dict[int, int] = {}
    for row, frame in enumerate(frame_row):
        local_peak[row] = running.get(int(frame), 0)
        running[int(frame)] = int(local_peak[row]) + 1

    peak_pressure = np.asarray(frames["pressure"], float)[frame_row]
    if sample_type == "single_crystal" and "pressure" in group:
        obs_pressure = _dataset_or_default(group, "pressure", count, math.nan)[good][order]
        peak_pressure = np.where(np.isfinite(peak_pressure), peak_pressure, obs_pressure)
    return PeakTable(
        source_index=source_index,
        frame_row=frame_row,
        original_frame=original_frame,
        local_peak=local_peak,
        center=centers,
        width=widths,
        half_width=half_width,
        area=area[good][order],
        pressure=peak_pressure,
        track=track[good][order].astype(np.int32),
    )


def _roi_profiles(
    radial: np.ndarray,
    transformed: np.ndarray,
    peaks: PeakTable,
) -> Tuple[np.ndarray, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, ROI_PROFILE_POINTS)
    profiles = np.full((peaks.size, coordinate.size), np.nan, dtype=float)
    if radial.size < 2 or not np.all(np.isfinite(radial)):
        return coordinate, profiles
    for index in range(peaks.size):
        grid = peaks.center[index] + coordinate * peaks.half_width[index]
        row = transformed[int(peaks.frame_row[index])]
        # Interpolate the full row: a masked/native NaN bin is structural and
        # poisons the samples adjacent to it instead of being bridged.
        profiles[index] = np.interp(
            grid, radial, row, left=np.nan, right=np.nan
        )
    return coordinate, profiles


def _anchor_validity(
    radial: np.ndarray,
    transformed: np.ndarray,
    peaks: PeakTable,
) -> Dict[str, np.ndarray]:
    """Which anchors' ROI supports the scoring kernels can actually use.

    ``edge`` anchors have a support crossing the radial boundary; ``masked``
    anchors have a non-finite native bin strictly inside their own support.
    Either condition makes the whole anchor row structurally NaN, so such
    anchors are flagged (and their per-anchor plots skipped) instead of
    silently producing empty maps.
    """

    lo = peaks.center - peaks.half_width
    hi = peaks.center + peaks.half_width
    in_bounds = (lo >= radial[0]) & (hi <= radial[-1])
    nonfinite = ~np.isfinite(np.asarray(transformed, dtype=float))
    csum = np.zeros(
        (nonfinite.shape[0], nonfinite.shape[1] + 1), dtype=np.int64
    )
    np.cumsum(nonfinite, axis=1, out=csum[:, 1:])
    left = np.searchsorted(radial, lo, side="right")
    right = np.maximum(np.searchsorted(radial, hi, side="left"), left)
    rows = peaks.frame_row.astype(int)
    interior_bad = (csum[rows, right] - csum[rows, left]) > 0
    return {
        "valid": in_bounds & ~interior_bad,
        "edge": ~in_bounds,
        "masked": in_bounds & interior_bad,
    }


def _powder_roi_matrix(
    radial: np.ndarray,
    transformed: np.ndarray,
    peaks: PeakTable,
) -> np.ndarray:
    """Vectorized K x K directional ROI matrix.

    Reproduces :func:`directional_anchor_iou` cell for cell (the scalar stays
    as the public reference and comparator): the same integration partition --
    native bins inside the anchor support, the support boundaries, the
    target-support cuts, and the min/max crossing points -- with every piece
    integrated in closed form, evaluated for all K targets of one anchor in
    a single vectorized pass instead of K^2 Python calls.
    """

    x = np.asarray(radial, dtype=float)
    profiles = np.asarray(transformed, dtype=float)
    count = peaks.size
    result = np.full((count, count), np.nan, dtype=float)
    if count == 0 or x.size < 2:
        return result
    lo = peaks.center - peaks.half_width
    hi = peaks.center + peaks.half_width
    frame = peaks.frame_row.astype(int)
    eps = np.finfo(float).eps

    finite = np.isfinite(profiles)
    frame_usable = finite.sum(axis=1) >= 2
    # The scalar interpolates every value over each frame's FINITE bins, so a
    # NaN bin outside the structural-gate regions is bridged, not propagated.
    # Pre-bridging each frame once reproduces that everywhere.
    bridged = profiles.copy()
    for row in np.nonzero(~finite.all(axis=1))[0]:
        if frame_usable[row]:
            bridged[row] = np.interp(x, x[finite[row]], profiles[row][finite[row]])
    nonfinite_csum = np.zeros(
        (profiles.shape[0], profiles.shape[1] + 1), dtype=np.int64
    )
    np.cumsum(~finite, axis=1, out=nonfinite_csum[:, 1:])

    validity = _anchor_validity(x, profiles, peaks)
    support_ok = np.isfinite(lo) & np.isfinite(hi) & (lo < hi)
    anchor_ok = validity["valid"] & support_ok & frame_usable[frame]

    # One (K, R) gather up front instead of one per anchor iteration.
    target_rows = bridged[frame]

    def _boundary_values(value: float) -> np.ndarray:
        """Every peak's bridged frame profile interpolated at one value."""
        base = int(np.clip(np.searchsorted(x, value, side="right") - 1,
                           0, x.size - 2))
        weight = (value - x[base]) / (x[base + 1] - x[base])
        return (
            target_rows[:, base] * (1.0 - weight)
            + target_rows[:, base + 1] * weight
        )

    # Structural gate, target side: a non-finite native bin strictly inside
    # the overlap of the two supports voids that pair.
    target_bad_base = ~(support_ok & frame_usable[frame])

    anchors_todo = np.nonzero(anchor_ok)[0]
    chunk = max(1, anchors_todo.size // 20)
    for done, i in enumerate(anchors_todo, start=1):
        if done % chunk == 0 or done == anchors_todo.size:
            _progress(done, anchors_todo.size)
            from . import checkpoint as _stop

            _stop.check_stop()
        a_lo, a_hi = float(lo[i]), float(hi[i])
        s0 = int(np.searchsorted(x, a_lo, side="right"))
        s1 = int(np.searchsorted(x, a_hi, side="left"))
        grid = np.concatenate(([a_lo], x[s0:s1], [a_hi]))
        left = grid[:-1]
        right = grid[1:]
        width_all = right - left
        boundary_lo = _boundary_values(a_lo)
        boundary_hi = _boundary_values(a_hi)
        a_raw = np.empty(grid.size)
        a_raw[0] = boundary_lo[i]
        a_raw[-1] = boundary_hi[i]
        a_raw[1:-1] = bridged[frame[i], s0:s1]
        t_raw = np.empty((count, grid.size))
        t_raw[:, 0] = boundary_lo
        t_raw[:, -1] = boundary_hi
        t_raw[:, 1:-1] = target_rows[:, s0:s1]

        a0, a1 = a_raw[:-1], a_raw[1:]
        clip_a0 = np.clip(a0, 0.0, None)
        clip_a1 = np.clip(a1, 0.0, None)
        t0, t1 = t_raw[:, :-1], t_raw[:, 1:]
        cut1 = np.clip(lo[:, None], left, right)
        cut2 = np.clip(hi[:, None], left, right)
        frac1 = (cut1 - left) / width_all
        frac2 = (cut2 - left) / width_all
        # Raw linear values at the cut points, then clip -- the scalar samples
        # clip(raw interpolant) at every knot it integrates between.
        a_cut1 = np.clip(a0 + frac1 * (a1 - a0), 0.0, None)
        a_cut2 = np.clip(a0 + frac2 * (a1 - a0), 0.0, None)
        t_cut1 = np.clip(t0 + frac1 * (t1 - t0), 0.0, None)
        t_cut2 = np.clip(t0 + frac2 * (t1 - t0), 0.0, None)
        # Outside the target support T = 0: min contributes nothing (A >= 0),
        # max integrates the anchor alone.
        den_zero = (
            (clip_a0[None, :] + a_cut1) * 0.5 * (cut1 - left)
            + (a_cut2 + clip_a1[None, :]) * 0.5 * (right - cut2)
        )
        overlap_width = cut2 - cut1
        diff0 = a_cut1 - t_cut1
        diff1 = a_cut2 - t_cut2
        min0 = np.minimum(a_cut1, t_cut1)
        min1 = np.minimum(a_cut2, t_cut2)
        max0 = np.maximum(a_cut1, t_cut1)
        max1 = np.maximum(a_cut2, t_cut2)
        crossing = (diff0 * diff1) < 0.0
        cross_frac = np.where(
            crossing, -diff0 / np.where(crossing, diff1 - diff0, 1.0), 0.0
        )
        value_at_cross = a_cut1 + cross_frac * (a_cut2 - a_cut1)
        num_mid = np.where(
            crossing,
            (min0 + value_at_cross) * 0.5 * cross_frac * overlap_width
            + (value_at_cross + min1) * 0.5 * (1.0 - cross_frac) * overlap_width,
            (min0 + min1) * 0.5 * overlap_width,
        )
        den_mid = np.where(
            crossing,
            (max0 + value_at_cross) * 0.5 * cross_frac * overlap_width
            + (value_at_cross + max1) * 0.5 * (1.0 - cross_frac) * overlap_width,
            (max0 + max1) * 0.5 * overlap_width,
        )
        empty = overlap_width <= 0.0
        num = np.where(empty, 0.0, num_mid).sum(axis=1)
        den = (den_zero + np.where(empty, 0.0, den_mid)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(
                den <= eps, 0.0, np.clip(num / den, 0.0, 1.0)
            )

        overlap_lo = np.maximum(a_lo, lo)
        overlap_hi = np.minimum(a_hi, hi)
        has_overlap = overlap_lo < overlap_hi
        gate_left = np.searchsorted(x, overlap_lo, side="right")
        gate_right = np.maximum(
            np.searchsorted(x, overlap_hi, side="left"), gate_left
        )
        masked_overlap = np.zeros(count, dtype=bool)
        rows_with = np.nonzero(has_overlap)[0]
        masked_overlap[rows_with] = (
            nonfinite_csum[frame[rows_with], gate_right[rows_with]]
            - nonfinite_csum[frame[rows_with], gate_left[rows_with]]
        ) > 0
        score[target_bad_base | masked_overlap] = np.nan
        result[i] = score
    return result


def _single_roi_features(
    roi_profiles: np.ndarray,
) -> np.ndarray:
    """1D approximation to each spot's mean transformed-pixel ROI feature.

    Fixed-size interpolated ROI samples avoid native-bin-count bias when spot
    widths differ. A feature remains NaN when fewer than two support samples
    are structurally available.
    """

    profiles = np.asarray(roi_profiles, dtype=float)
    finite = np.isfinite(profiles)
    counts = finite.sum(axis=1)
    clipped = np.where(finite, np.clip(profiles, 0.0, None), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = clipped.sum(axis=1) / counts
    return np.where(counts >= 2, means, np.nan)


def _single_roi_matrix(features: np.ndarray) -> np.ndarray:
    return relative_feature_similarity(features[:, None], features[None, :])


def _location_matrix(peaks: PeakTable, tolerance: float) -> np.ndarray:
    return location_similarity(
        peaks.center[:, None], peaks.center[None, :], tolerance=float(tolerance)
    )


def _mask_same_frame(matrix: np.ndarray, peaks: PeakTable) -> np.ndarray:
    """Blank comparisons between observations from the same source frame."""

    result = np.asarray(matrix, dtype=float).copy()
    expected = (peaks.size, peaks.size)
    if result.shape != expected:
        raise ValueError(f"anchor matrix must have shape {expected}")
    same_frame = peaks.frame_row[:, None] == peaks.frame_row[None, :]
    result[same_frame] = np.nan
    return result


def _create_dataset(group, name: str, data: np.ndarray, **kwargs):
    array = np.asarray(data)
    options = dict(kwargs)
    if array.size and array.ndim and array.dtype.kind not in ("O", "U", "S"):
        options.setdefault("compression", "gzip")
        options.setdefault("shuffle", True)
    return group.create_dataset(name, data=array, **options)


def _write_h5(
    path: Path,
    *,
    analysis_path: Path,
    sample_type: str,
    source_requested: str,
    source_resolved: str,
    unit: str,
    radial: np.ndarray,
    original_positive: np.ndarray,
    transformed_positive: np.ndarray,
    transformed_signed: np.ndarray,
    transform: TransformParameters,
    frames: Mapping[str, Any],
    peaks: PeakTable,
    peak_valid: np.ndarray,
    profile_coordinate: np.ndarray,
    roi_profiles: np.ndarray,
    roi_feature: Optional[np.ndarray],
    roi_area: np.ndarray,
    roi_algorithm: str,
    location: np.ndarray,
    windows: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    tracks_bundle: Optional[Mapping[str, Any]] = None,
    intervals: Optional[Mapping[str, np.ndarray]] = None,
    track_group_labels: Optional[Sequence[str]] = None,
) -> None:
    import h5py  # type: ignore

    # Unique per run so two concurrent same-sample runs into one directory
    # cannot clobber each other's staging file. (core.config.write_json still
    # uses a fixed .json.tmp -- same latent race, tracked separately.)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with h5py.File(str(tmp), "w") as h5:
            h5.attrs.update(
                {
                    "tool": TOOL,
                    "schema_version": SCHEMA_VERSION,
                    "seriesxrd_version": VERSION,
                    "created_at": now_iso(),
                    "sample_type": sample_type,
                    "source_requested": source_requested,
                    "source_resolved": source_resolved,
                    "source_analysis": str(analysis_path),
                    "unit": unit,
                    "n_frames": int(original_positive.shape[0]),
                    "n_peaks": int(peaks.size),
                    "n_windows": int(windows["start"].size),
                    "all_peak_policy": "one anchor per retained observation; never collapse by track",
                    "roi_area_method": roi_algorithm,
                    "roi_area_directional": sample_type == "powder",
                    "order_by": str(frames["order_by"]),
                    "order_label": str(frames["order_label"]),
                }
            )
            write_provenance(
                h5,
                tool=TOOL,
                schema_version=SCHEMA_VERSION,
                config=dict(config),
                inputs={"analysis": analysis_path},
            )

            transform_group = h5.create_group("transform")
            transform_group.attrs.update(transform.to_dict())
            transform_group.attrs["position_in_pipeline"] = (
                "after Analysis source reconstruction; positive path before ROI "
                "correlations and signed-input path before window correlations"
            )
            transform_group.attrs["scale_estimate"] = (
                "pooled positive finite source intensity quantile"
            )
            transform_group.attrs["noise_estimate"] = (
                "first-difference MAD on original signed source patterns"
            )

            patterns = h5.create_group("patterns")
            _create_dataset(patterns, "radial", radial)
            _create_dataset(patterns, "original_positive", original_positive)
            _create_dataset(patterns, "log_squared", transformed_positive)
            _create_dataset(patterns, "log_squared_signed", transformed_signed)
            patterns["log_squared"].attrs["role"] = "positive ROI intensity"
            patterns["log_squared_signed"].attrs["role"] = (
                "signed-input window residual; sign deliberately erased by squaring"
            )
            patterns.attrs["waterfall_height_source"] = "original_positive"
            patterns.attrs["waterfall_color_source"] = "anchor_maps/roi_area"

            frame_group = h5.create_group("frames")
            _create_dataset(frame_group, "index", frames["original_index"])
            frame_group.create_dataset(
                "filename",
                data=np.asarray(frames["filename"], dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            _create_dataset(frame_group, "pressure", frames["pressure"])
            _create_dataset(frame_group, "order_value", frames["order_values"])
            frame_group["order_value"].attrs["axis"] = str(frames["order_by"])
            frame_group["order_value"].attrs["label"] = str(
                frames["order_label"]
            )

            peak_group = h5.create_group("peaks")
            _create_dataset(peak_group, "id", np.arange(peaks.size, dtype=np.int32))
            for name in (
                "source_index",
                "frame_row",
                "original_frame",
                "local_peak",
                "center",
                "width",
                "half_width",
                "area",
                "pressure",
                "track",
            ):
                _create_dataset(peak_group, name, getattr(peaks, name))
            _create_dataset(peak_group, "valid", np.asarray(peak_valid, bool))
            peak_group["valid"].attrs["meaning"] = (
                "ROI support lies inside the radial axis with no masked native "
                "bin inside it; an invalid anchor's score row is structurally "
                "NaN and its per-anchor plots are skipped"
            )
            peak_group.attrs["all_peak"] = True
            peak_group.attrs["track_used_for_grouping"] = False

            anchors = h5.create_group("anchor_maps")
            _create_dataset(anchors, "profile_coordinate", profile_coordinate)
            _create_dataset(anchors, "roi_profiles_log_squared", roi_profiles)
            anchors["roi_profiles_log_squared"].attrs["role"] = (
                "per-peak support samples for review; powder scoring remains on "
                "the absolute native-radial anchor support"
            )
            if roi_feature is not None:
                _create_dataset(anchors, "roi_feature_log_squared", roi_feature)
                anchors["roi_feature_log_squared"].attrs["method"] = (
                    "one_dimensional_radial_mean_approximation_of_raw_pixel_roi_feature"
                )
            _create_dataset(anchors, "roi_area", roi_area)
            _create_dataset(anchors, "location", location)
            anchors["roi_area"].attrs["range"] = (
                "[0,1], NaN=same-frame comparison or structurally unavailable "
                "observation/support"
            )
            anchors["roi_area"].attrs["zero_denominator"] = "0"
            anchors["roi_area"].attrs["algorithm"] = roi_algorithm
            anchors["roi_area"].attrs["directional"] = sample_type == "powder"
            anchors["roi_area"].attrs["same_frame_policy"] = "NaN"
            anchors["location"].attrs["range"] = (
                "[0,1], NaN=same-frame comparison or non-finite center"
            )
            anchors["location"].attrs["intensity_transform"] = "none"
            anchors["location"].attrs["same_frame_policy"] = "NaN"

            window_group = h5.create_group("windows")
            for name in (
                "start",
                "end",
                "acf_features",
                "across_direct",
                "across_acf",
                "within_acf",
            ):
                _create_dataset(window_group, name, windows[name])
            labels = [
                f"{start:g}-{end:g}" for start, end in zip(
                    windows["start"], windows["end"], strict=True
                )
            ]
            window_group.create_dataset(
                "label",
                data=np.asarray(labels, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            window_group.attrs["width"] = float(config["window_width"])
            window_group.attrs["step"] = float(config["window_step"])
            window_group.attrs["intensity_source"] = "patterns/log_squared_signed"
            window_group.attrs["acf_method"] = (
                "standardized_positive_lag_fft_acf"
            )
            window_group.attrs["acf_lag_policy"] = (
                "all_positive_lags;zero_lag_excluded"
            )
            window_group.attrs["acf_lag_count"] = int(
                windows["acf_features"].shape[-1]
            )
            window_group.attrs["invalid_policy"] = (
                "structurally invalid signals/fingerprints remain NaN"
            )

            if tracks_bundle is not None:
                from .tracks import TRANSITION_RULE

                track_h5 = h5.create_group("tracks")
                track_h5.attrs.update(
                    {
                        "linker": "seriesxrd.analysis.unknowns.link_tracks",
                        "similarity": "mutual_sqrt_directional_roi",
                        "exploratory": True,
                        "transition_rule": TRANSITION_RULE,
                        "n_tracks": int(
                            tracks_bundle["summary"]["id"].size
                        ),
                        "group_by": str(config.get("track_group_by", "none")),
                    }
                )
                track_h5.attrs.update(tracks_bundle["settings"])
                obs_h5 = track_h5.create_group("obs")
                for name, values in tracks_bundle["obs"].items():
                    _create_dataset(obs_h5, name, values)
                summary_h5 = track_h5.create_group("summary")
                for name, values in tracks_bundle["summary"].items():
                    _create_dataset(summary_h5, name, values)
                edges_h5 = track_h5.create_group("edges")
                for name, values in tracks_bundle["edges"].items():
                    _create_dataset(edges_h5, name, values)
                if intervals is not None:
                    interval_h5 = track_h5.create_group("intervals")
                    for name, values in intervals.items():
                        _create_dataset(interval_h5, name, values)
                if track_group_labels:
                    track_h5.create_dataset(
                        "group_label",
                        data=np.asarray(list(track_group_labels), object),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )

        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def run_correlations(
    analysis_h5: str | Path,
    out_dir: str | Path,
    *,
    sample_type: str,
    source: Optional[str] = None,
    radial_min: Optional[float] = None,
    radial_max: Optional[float] = None,
    window_width: float = DEFAULT_WINDOW_WIDTH,
    window_step: float = DEFAULT_WINDOW_STEP,
    location_tolerance: float = DEFAULT_LOCATION_TOLERANCE,
    scale_quantile: float = DEFAULT_SCALE_QUANTILE,
    plots: "Optional[Sequence[str]]" = None,
    make_plots: Optional[bool] = None,
    max_anchor_plots: Optional[int] = None,
    order_by: str = "frame",
    make_tracks: bool = True,
    track_min_similarity: float = 0.2,
    track_min_frames: int = 3,
    track_link_tol_fwhm: float = 1.5,
    track_max_gap: int = 2,
    track_group_by: str = "none",
    export_csv: bool = False,
    export_matrix_csv: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Generate the correlation artifact, and optionally figures and CSVs.

    ``order_by`` orders the retained frames by a /frames metadata axis
    (``frame`` | ``pressure`` | ``temperature`` | ``time``) before anything
    downstream sees them, so waterfall rows, window frame axes, and peak
    ``frame_row`` all follow the physical series. The default keeps the
    Analysis file order exactly as before.

    ``plots`` selects which figure families to write as PNG; the default is
    **none**, because the artifact already holds every number and figures
    are drawn on demand. See :func:`resolve_plot_families`.
    """

    import h5py  # type: ignore

    plot_families = resolve_plot_families(plots, make_plots)
    resume = bool(resume)
    analysis_path = Path(analysis_h5).expanduser().resolve()
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {analysis_path}")
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sample = str(sample_type).strip().lower()
    if sample not in SAMPLE_TYPES:
        raise ValueError(f"sample_type must be one of {SAMPLE_TYPES}")
    source_requested = (
        "fit" if sample == "powder" else "spots"
    ) if source is None or not str(source).strip() else str(source).strip().lower()
    if not np.isfinite(location_tolerance) or location_tolerance <= 0.0:
        raise ValueError("location_tolerance must be finite and positive")

    if resume:
        # Compute is cheap; figure export is not. When the artifact already
        # matches the requested settings, skip straight to exporting so an
        # interrupted export resumes instead of recomputing first.
        existing = destination / f"correlations_{sample}.h5"
        manifest_path = destination / f"manifest_{sample}.json"
        compute_key = {
            "sample_type": sample,
            "source": source_requested,
            "radial_min_requested": (
                None if radial_min is None else float(radial_min)
            ),
            "radial_max_requested": (
                None if radial_max is None else float(radial_max)
            ),
            "window_width": float(window_width),
            "window_step": float(window_step),
            "location_tolerance": float(location_tolerance),
            "scale_quantile": float(scale_quantile),
            "order_by": str(order_by),
            "make_tracks": bool(make_tracks),
            "track_min_similarity": float(track_min_similarity),
            "track_min_frames": int(track_min_frames),
            "track_link_tol_fwhm": float(track_link_tol_fwhm),
            "track_max_gap": int(track_max_gap),
        }
        ok, reason = reusable_artifact(existing, analysis_path, compute_key)
        if ok:
            print(f"[CORRELATIONS] {reason}; exporting only", flush=True)
            # A run interrupted during export never got to write its
            # manifest — the very case resume exists for — so fall back to
            # the artifact's own attributes when it is missing.
            return _export_from_existing(
                existing,
                destination,
                read_json(manifest_path),
                plot_families=plot_families,
                max_anchor_plots=max_anchor_plots,
                export_csv=export_csv,
                export_matrix_csv=export_matrix_csv,
            )
        print(f"[CORRELATIONS] recomputing ({reason})", flush=True)

    from ..analysis.refine_export import _pattern_source

    with h5py.File(str(analysis_path), "r") as source_h5:
        if "radial" not in source_h5:
            raise ValueError("Analysis HDF5 has no /radial axis")
        patterns, source_resolved = _pattern_source(source_h5, source_requested)
        patterns = np.asarray(patterns, dtype=float)
        radial = np.asarray(source_h5["radial"][:], dtype=float).reshape(-1)
        if patterns.ndim != 2 or patterns.shape[1] != radial.size:
            raise ValueError("Analysis source must have shape (frames, radial bins)")
        if radial.size < 4 or np.any(~np.isfinite(radial)):
            raise ValueError("/radial must contain at least four finite values")
        if np.all(np.diff(radial) < 0.0):
            radial = radial[::-1]
            patterns = patterns[:, ::-1]
        elif np.any(np.diff(radial) <= 0.0):
            raise ValueError("/radial must be strictly monotonic")
        frames = _read_frames(source_h5, patterns.shape[0], order_by)
        patterns = patterns[frames["original_index"]]
        lower = float(radial[0]) if radial_min is None else float(radial_min)
        upper = float(radial[-1]) if radial_max is None else float(radial_max)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("radial bounds must be finite with min < max")
        radial_keep = (radial >= lower) & (radial <= upper)
        if np.count_nonzero(radial_keep) < 4:
            raise ValueError("radial bounds retain fewer than four bins")
        radial = radial[radial_keep]
        patterns = patterns[:, radial_keep]
        unit = _decode(source_h5.attrs.get("unit", "radial"))
        peaks = _read_peaks(source_h5, sample, frames, radial)
        track_group_key, track_group_ids, track_group_labels = (
            "none", None, ["all"]
        )
        if make_tracks:
            from ..analysis.series import tracking_groups

            track_group_key, group_ids_all, track_group_labels = (
                tracking_groups(
                    source_h5, track_group_by, int(frames["keep"].size)
                )
            )
            track_group_ids = np.asarray(group_ids_all, dtype=int)[
                frames["original_index"]
            ]

    original_positive = np.where(
        np.isfinite(patterns), np.clip(patterns, 0.0, None), np.nan
    )
    transformed_positive, transform = log_squared_transform(
        patterns, scale_quantile=float(scale_quantile)
    )
    transformed_signed = _signed_log_squared_transform(patterns, transform)
    validity = _anchor_validity(radial, transformed_positive, peaks)
    profile_coordinate, roi_profiles = _roi_profiles(
        radial, transformed_positive, peaks
    )
    if sample == "powder":
        roi_feature = None
        roi_area = _powder_roi_matrix(radial, transformed_positive, peaks)
        roi_algorithm = (
            "directional_absolute_native_radial_anchor_support_integrated_iou;"
            "target_zero_outside_own_support;half_width=0.75*peak_width;"
            "zero_denominator=0;no_recentering_or_width_normalization"
        )
    else:
        roi_feature = _single_roi_features(roi_profiles)
        roi_area = _single_roi_matrix(roi_feature)
        roi_algorithm = (
            "min(feature_i,feature_j)/max(feature_i,feature_j);both_zero=0;"
            "feature=one_dimensional_radial_mean_approximation_of_raw_pixel_"
            "positive_log_squared_roi_mean"
        )
    roi_area = _mask_same_frame(roi_area, peaks)
    location = _mask_same_frame(
        _location_matrix(peaks, float(location_tolerance)), peaks
    )
    windows = compute_window_correlations(
        radial,
        transformed_signed,
        window_width=float(window_width),
        window_step=float(window_step),
    )
    tracks_bundle = None
    intervals = None
    if make_tracks:
        from .tracks import build_tracks, transition_summary

        n_kept = int(original_positive.shape[0])
        axis_used = (
            np.arange(n_kept, dtype=float)
            if frames["order_by"] == "frame"
            else np.asarray(frames["order_values"], dtype=float)
        )
        tracks_bundle = build_tracks(
            peaks,
            roi_area,
            n_frames=n_kept,
            order_key=frames["order_by"],
            order_values=frames["order_values"],
            group_ids=track_group_ids,
            link_tol_fwhm=float(track_link_tol_fwhm),
            max_gap=int(track_max_gap),
            min_track_frames=int(track_min_frames),
            min_roi_similarity=float(track_min_similarity),
        )
        intervals = transition_summary(
            tracks_bundle,
            n_frames=n_kept,
            order_key=frames["order_by"],
            order_values=axis_used,
            group_ids=track_group_ids,
            across_direct=windows["across_direct"],
        )
    config = {
        "sample_type": sample,
        "source": source_requested,
        "source_resolved": source_resolved,
        "radial_min": float(radial[0]),
        "radial_max": float(radial[-1]),
        "radial_min_requested": (
            None if radial_min is None else float(radial_min)
        ),
        "radial_max_requested": (
            None if radial_max is None else float(radial_max)
        ),
        "window_width": float(window_width),
        "window_step": float(window_step),
        "location_tolerance": float(location_tolerance),
        "scale_quantile": float(scale_quantile),
        "plots": list(plot_families),
        "max_anchor_plots": (
            None if max_anchor_plots is None else int(max_anchor_plots)
        ),
        "order_by": frames["order_by"],
        "make_tracks": bool(make_tracks),
        "track_min_similarity": float(track_min_similarity),
        "track_min_frames": int(track_min_frames),
        "track_link_tol_fwhm": float(track_link_tol_fwhm),
        "track_max_gap": int(track_max_gap),
        "track_group_by": str(track_group_key),
    }
    h5_path = destination / f"correlations_{sample}.h5"
    _write_h5(
        h5_path,
        analysis_path=analysis_path,
        sample_type=sample,
        source_requested=source_requested,
        source_resolved=source_resolved,
        unit=unit,
        radial=radial,
        original_positive=original_positive,
        transformed_positive=transformed_positive,
        transformed_signed=transformed_signed,
        transform=transform,
        frames=frames,
        peaks=peaks,
        peak_valid=validity["valid"],
        profile_coordinate=profile_coordinate,
        roi_profiles=roi_profiles,
        roi_feature=roi_feature,
        roi_area=roi_area,
        roi_algorithm=roi_algorithm,
        location=location,
        windows=windows,
        config=config,
        tracks_bundle=tracks_bundle,
        intervals=intervals,
        track_group_labels=track_group_labels,
    )

    plot_files: Sequence[str] = ()
    if plot_families:
        from .plots import render_all

        plot_files = render_all(
            h5_path,
            destination / "heatmaps",
            max_anchor_plots=max_anchor_plots,
            families=plot_families,
        )
    csv_files: list = []
    if export_csv or export_matrix_csv:
        from . import export as _export

        csv_dir = destination / "csv"
        if export_csv:
            csv_files.extend(
                _export.export_summary_csvs(h5_path, csv_dir)
            )
        if export_matrix_csv:
            csv_files.extend(_export.export_matrices(h5_path, csv_dir))
    manifest = {
        **manifest_provenance(TOOL, SCHEMA_VERSION),
        "analysis_h5": str(analysis_path),
        "out_dir": str(destination),
        "correlations_h5": str(h5_path),
        "sample_type": sample,
        "source_requested": source_requested,
        "source_resolved": source_resolved,
        "unit": unit,
        "radial_min": float(radial[0]),
        "radial_max": float(radial[-1]),
        "n_frames": int(original_positive.shape[0]),
        "n_excluded_frames": int(frames["excluded_count"]),
        "order_by": frames["order_by"],
        "order_label": frames["order_label"],
        "n_peaks": int(peaks.size),
        "n_anchors_valid": int(np.count_nonzero(validity["valid"])),
        "n_anchors_edge": int(np.count_nonzero(validity["edge"])),
        "n_anchors_masked": int(np.count_nonzero(validity["masked"])),
        "n_windows": int(windows["start"].size),
        "all_peak": True,
        "track_collapsed": False,
        "transform": transform.to_dict(),
        "roi_area_method": roi_algorithm,
        "same_frame_policy": "NaN",
        "location_method": "linear_native_axis_tolerance",
        "location_tolerance": float(location_tolerance),
        "window_width": float(window_width),
        "window_step": float(window_step),
        "window_intensity_source": "signed_log_squared",
        "window_acf_method": "standardized_positive_lag_fft_acf",
        "window_acf_lag_policy": "all_positive_lags;zero_lag_excluded",
        "window_acf_lag_count": int(windows["acf_features"].shape[-1]),
        "products": [
            "roi_area",
            "location",
            "window_across_direct",
            "window_across_acf",
            "window_within_acf",
        ]
        + (["tracks"] if tracks_bundle is not None else []),
        "plots": list(plot_families),
        "anchor_plot_cap": (
            None if max_anchor_plots is None else int(max_anchor_plots)
        ),
        "tracks": (
            None
            if tracks_bundle is None
            else {
                "n_tracks": int(tracks_bundle["summary"]["id"].size),
                "n_track_obs": int(tracks_bundle["obs"]["track"].size),
                "n_transition_candidates": (
                    0
                    if intervals is None
                    else int(
                        np.count_nonzero(intervals["transition_candidate"])
                    )
                ),
                **tracks_bundle["settings"],
                "group_by": str(track_group_key),
                "exploratory": True,
            }
        ),
        "plots_written": len(plot_files),
        "plot_files": list(plot_files),
        "csv_files": [
            path.relative_to(destination).as_posix() for path in csv_files
        ],
    }
    write_json(destination / f"manifest_{sample}.json", manifest)
    return manifest


__all__ = [
    "DEFAULT_EPSILON_FLOOR",
    "DEFAULT_LOCATION_TOLERANCE",
    "DEFAULT_SCALE_QUANTILE",
    "DEFAULT_WINDOW_STEP",
    "DEFAULT_WINDOW_WIDTH",
    "SAMPLE_TYPES",
    "SCHEMA_VERSION",
    "TransformParameters",
    "compute_window_correlations",
    "directional_anchor_iou",
    "integrated_iou",
    "location_similarity",
    "log_squared_transform",
    "relative_feature_similarity",
    "run_correlations",
]
