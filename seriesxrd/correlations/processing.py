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

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.config import VERSION, now_iso, write_json
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


def _finite_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _estimate_noise_floor(values: np.ndarray) -> float:
    """Robust white-noise estimate from first differences, pooled by frame."""

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
    output[valid] = np.log1p((z * z) / epsilon) / math.log1p(1.0 / epsilon)
    output[valid] = np.clip(output[valid], 0.0, 1.0)
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
    output[valid] = np.log1p((z * z) / parameters.epsilon) / math.log1p(
        1.0 / parameters.epsilon
    )
    output[valid] = np.clip(output[valid], 0.0, 1.0)
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
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlation input must be a two-dimensional array")
    count = values.shape[0]
    result = np.full((count, count), np.nan, dtype=float)
    for left in range(count):
        for right in range(left, count):
            score = _pearson(values[left], values[right])
            result[left, right] = result[right, left] = score
    return result


def _acf(values: np.ndarray) -> np.ndarray:
    """Standardized full positive-lag autocorrelation fingerprint via FFT."""

    row = np.asarray(values, dtype=float).reshape(-1)
    length = max(row.size - 1, 0)
    invalid = np.full(length, np.nan, dtype=float)
    if row.size < 3 or not np.all(np.isfinite(row)):
        return invalid
    row = row - float(np.mean(row))
    signal_scale = float(np.std(row))
    magnitude = max(float(np.max(np.abs(row))), 1.0)
    if not np.isfinite(signal_scale) or signal_scale <= np.finfo(float).eps * magnitude:
        return invalid
    standardized_signal = row / signal_scale
    count = standardized_signal.size
    transformed = np.fft.rfft(standardized_signal, n=2 * count)
    correlation = np.fft.irfft(
        transformed * np.conjugate(transformed), n=2 * count
    )[:count]
    zero_lag = float(correlation[0])
    if not np.isfinite(zero_lag) or zero_lag <= np.finfo(float).eps:
        return invalid
    positive_lags = np.asarray(correlation[1:] / zero_lag, dtype=float)
    if not np.all(np.isfinite(positive_lags)):
        return invalid
    positive_lags -= float(np.mean(positive_lags))
    fingerprint_scale = float(np.std(positive_lags))
    if not np.isfinite(fingerprint_scale) or fingerprint_scale <= 1.0e-12:
        return invalid
    return positive_lags / fingerprint_scale


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
    for window, (start, end) in enumerate(zip(starts, ends, strict=True)):
        grid = np.linspace(float(start), float(end), int(points))
        left_index = max(int(np.searchsorted(radial, start, side="right")) - 1, 0)
        right_index = min(
            int(np.searchsorted(radial, end, side="left")), radial.size - 1
        )
        for frame in range(frames):
            support = np.asarray(
                values[frame, left_index : right_index + 1], dtype=float
            )
            # A masked/native NaN bin is structural, not a value that the
            # correlation stage may silently reconstruct by interpolation.
            if support.size < 2 or not np.all(np.isfinite(support)):
                continue
            output[frame, window] = np.interp(
                grid,
                radial[left_index : right_index + 1],
                support,
                left=np.nan,
                right=np.nan,
            )
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
    frames, windows, _ = signals.shape
    acf_features = np.full(
        (frames, windows, signals.shape[-1] - 1), np.nan, dtype=float
    )
    for frame in range(frames):
        for window in range(windows):
            acf_features[frame, window] = _acf(signals[frame, window])

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


def _read_frames(h5, n_frames: int) -> Dict[str, Any]:
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
    original_index = np.nonzero(keep)[0].astype(np.int32)
    old_to_new = np.full(n_frames, -1, dtype=np.int32)
    old_to_new[original_index] = np.arange(original_index.size, dtype=np.int32)
    return {
        "keep": keep,
        "original_index": original_index,
        "old_to_new": old_to_new,
        "filename": [names[index] for index in original_index],
        "pressure": pressure[keep],
        "excluded_count": int(np.count_nonzero(excluded)),
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


def _powder_roi_matrix(
    radial: np.ndarray,
    transformed: np.ndarray,
    peaks: PeakTable,
) -> np.ndarray:
    result = np.full((peaks.size, peaks.size), np.nan, dtype=float)
    supports = np.column_stack(
        (peaks.center - peaks.half_width, peaks.center + peaks.half_width)
    )
    for anchor in range(peaks.size):
        anchor_profile = transformed[int(peaks.frame_row[anchor])]
        for target in range(peaks.size):
            result[anchor, target] = directional_anchor_iou(
                radial,
                anchor_profile,
                transformed[int(peaks.frame_row[target])],
                anchor_support=tuple(supports[anchor]),
                target_support=tuple(supports[target]),
            )
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
    features = np.full(profiles.shape[0], np.nan, dtype=float)
    for index, profile in enumerate(profiles):
        finite = np.isfinite(profile)
        if np.count_nonzero(finite) >= 2:
            features[index] = float(
                np.mean(np.clip(profile[finite], 0.0, None))
            )
    return features


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
    profile_coordinate: np.ndarray,
    roi_profiles: np.ndarray,
    roi_feature: Optional[np.ndarray],
    roi_area: np.ndarray,
    roi_algorithm: str,
    location: np.ndarray,
    windows: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
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
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Generate the complete MVP correlation artifact and optional figures."""

    import h5py  # type: ignore

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
        frames = _read_frames(source_h5, patterns.shape[0])
        patterns = patterns[frames["keep"]]
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

    original_positive = np.where(
        np.isfinite(patterns), np.clip(patterns, 0.0, None), np.nan
    )
    transformed_positive, transform = log_squared_transform(
        patterns, scale_quantile=float(scale_quantile)
    )
    transformed_signed = _signed_log_squared_transform(patterns, transform)
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
    config = {
        "sample_type": sample,
        "source": source_requested,
        "source_resolved": source_resolved,
        "radial_min": float(radial[0]),
        "radial_max": float(radial[-1]),
        "window_width": float(window_width),
        "window_step": float(window_step),
        "location_tolerance": float(location_tolerance),
        "scale_quantile": float(scale_quantile),
        "make_plots": bool(make_plots),
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
        profile_coordinate=profile_coordinate,
        roi_profiles=roi_profiles,
        roi_feature=roi_feature,
        roi_area=roi_area,
        roi_algorithm=roi_algorithm,
        location=location,
        windows=windows,
        config=config,
    )

    plot_files: Sequence[str] = ()
    if make_plots:
        from .plots import render_all

        plot_files = render_all(h5_path, destination / "heatmaps")
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
        "n_peaks": int(peaks.size),
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
        ] + (["waterfall"] if make_plots else []),
        "plots_written": len(plot_files),
        "plot_files": list(plot_files),
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
