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
from typing import Callable, Iterator, List, Sequence, Tuple, TypeVar


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
) -> Iterator[_ResultT]:
    """Map independent payloads in processes, yielding results in payload order.

    Results are yielded lazily as chunks complete, so callers that print
    ``[LABEL] <done> <total>`` progress lines while absorbing keep streaming
    them to the GUI instead of seeing everything arrive at once.

    Some restricted containers and Python builds cannot create the semaphores
    and worker processes :class:`~concurrent.futures.ProcessPoolExecutor`
    needs.  That is an execution-environment limitation, not an analysis
    failure, and it surfaces as :class:`OSError` (including
    :class:`PermissionError`) while the pool is being constructed or while the
    payloads are being submitted -- before any result has been yielded.  Only
    that failure falls back to the ordered serial loop.

    Exceptions raised by ``worker`` itself, and a pool broken mid-run (a
    worker process killed by a segfault or the OOM killer), propagate
    unchanged: re-running a computation that just crashed natively inside the
    parent process could take the whole session down with it.
    """
    items = list(payloads)
    if max_workers <= 1 or len(items) <= 1:
        for payload in items:
            yield worker(payload)
        return

    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        # Executor.map submits every payload eagerly, so fork/spawn failures
        # land here rather than while the results are consumed below.
        results = executor.map(worker, items)
    except OSError as exc:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        print(
            f"[{label}] process pool unavailable "
            f"({type(exc).__name__}: {exc}); falling back to serial",
            flush=True,
        )
        for payload in items:
            yield worker(payload)
        return

    try:
        yield from results
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
