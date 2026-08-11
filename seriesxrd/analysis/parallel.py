"""Small helpers for parallelizing the per-frame analysis drivers.

The per-frame work in Steps 1-3 is independent (Step 2's seed propagation is the
only cross-frame coupling, and is preserved by processing *contiguous* chunks
serially inside each worker). These helpers pick a worker count and split the
frame range into contiguous chunks; the drivers fall back to a plain serial loop
when only one worker is requested or the host cannot create a process pool.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Sequence, Tuple, TypeVar


_PayloadT = TypeVar("_PayloadT")
_ResultT = TypeVar("_ResultT")


def resolve_workers(num_workers) -> int:
    """Map a user setting to a concrete worker count.

    ``None``/1 → serial (1); ``0`` or negative → auto (CPU count − 1, min 1);
    otherwise the requested count (min 1).
    """
    if num_workers is None:
        return 1
    try:
        n = int(num_workers)
    except (TypeError, ValueError):
        return 1
    if n == 1:
        return 1
    if n <= 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return max(1, n)


def chunk_ranges(n: int, k: int) -> List[Tuple[int, int]]:
    """Split ``range(n)`` into at most ``k`` contiguous ``(start, stop)`` ranges."""
    if n <= 0:
        return []
    k = max(1, min(int(k), n))
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(k) if bounds[i + 1] > bounds[i]]


def process_map_or_serial(
    worker: Callable[[_PayloadT], _ResultT],
    payloads: Sequence[_PayloadT],
    *,
    max_workers: int,
    label: str,
) -> List[_ResultT]:
    """Map independent payloads in processes, with an ordered serial fallback.

    Some restricted containers and Python builds cannot create the semaphore
    used internally by :class:`~concurrent.futures.ProcessPoolExecutor`.  That
    is an execution-environment limitation, not an analysis failure.  Materialize
    the complete parallel result before returning it so a pool failure can never
    leave a caller with a partially absorbed result, then repeat the *same*
    payload sequence serially.

    Worker exceptions are retried serially as well.  A genuine calculation
    error is therefore raised by the serial call with its direct traceback;
    only infrastructure failures disappear after a successful fallback.
    """
    items = list(payloads)
    if max_workers <= 1 or len(items) <= 1:
        return [worker(payload) for payload in items]

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(worker, items))
    except Exception as exc:
        print(
            f"[{label}] process pool unavailable "
            f"({type(exc).__name__}: {exc}); falling back to serial",
            flush=True,
        )
        return [worker(payload) for payload in items]
