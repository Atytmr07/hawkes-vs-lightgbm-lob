"""Linearized trade-size marks for M3 (PROJECT_SPEC §2.4, §3.2).

m_k = Volume_k / Median(Volume)_{[t-1h, t)}, asset-internal, rolling, and
strictly causal: the median is taken over trades strictly before event k
within the trailing window, excluding k itself (consistent with the rest
of this codebase's "state before the event" convention, e.g.
``ewma_bank.state_before_selected_events``). With no prior trade history
in the window (window start), the mark defaults to the neutral value 1.0
rather than an undefined ratio.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # keeps diagnostics importable before requirements are installed
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True, nogil=True)
def _fenwick_kth_bucket(tree: np.ndarray, distinct_values: int, top: int, k: int) -> int:
    """0-indexed bucket of the k-th smallest (1-indexed) active element."""
    pos = 0
    remaining = k
    step = top
    while step > 0:
        next_pos = pos + step
        if next_pos <= distinct_values and tree[next_pos] < remaining:
            pos = next_pos
            remaining -= tree[pos]
        step //= 2
    return pos


@njit(cache=True, nogil=True)
def _fenwick_rolling_median_buckets_before(
    value_bucket: np.ndarray,
    times_ms: np.ndarray,
    distinct_values: int,
    window_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower- and upper-median order-statistic buckets of the trailing
    window, evaluated strictly before inserting the current element (the two
    are equal for an odd active count; averaging their mapped values gives
    the true interpolated median for an even count). Both are -1 where the
    trailing window (excluding the current element) is still empty.
    """
    tree = np.zeros(distinct_values + 1, np.int64)
    lower = np.full(len(value_bucket), -1, np.int64)
    upper = np.full(len(value_bucket), -1, np.int64)
    left = 0
    active_count = 0
    top = 1
    while top * 2 <= distinct_values:
        top *= 2
    for right in range(len(value_bucket)):
        while left < right and times_ms[right] - times_ms[left] > window_ms:
            index = value_bucket[left] + 1
            while index <= distinct_values:
                tree[index] -= 1
                index += index & -index
            active_count -= 1
            left += 1
        if active_count > 0:
            k_lower = (active_count + 1) // 2
            k_upper = active_count // 2 + 1
            lower[right] = _fenwick_kth_bucket(tree, distinct_values, top, k_lower)
            upper[right] = (
                lower[right] if k_upper == k_lower
                else _fenwick_kth_bucket(tree, distinct_values, top, k_upper)
            )
        index = value_bucket[right] + 1
        while index <= distinct_values:
            tree[index] += 1
            index += index & -index
        active_count += 1
    return lower, upper


def rolling_median_before(
    values: np.ndarray,
    times_ms: np.ndarray,
    window_ms: int = 3_600_000,
) -> np.ndarray:
    """Causal trailing-window median, excluding the current element.

    True (interpolated) median: for an even-sized window, averages the two
    middle order statistics, matching ``numpy.median``. ``times_ms`` must be
    nondecreasing. NaN where the window has no prior element yet.
    """
    values = np.asarray(values, dtype=np.float64)
    times_ms = np.asarray(times_ms, dtype=np.int64)
    unique = np.unique(values)
    bucket = np.searchsorted(unique, values).astype(np.int64)
    lower, upper = _fenwick_rolling_median_buckets_before(bucket, times_ms, len(unique), window_ms)
    valid = lower >= 0
    result = np.full(len(values), np.nan)
    result[valid] = 0.5 * (unique[lower[valid]] + unique[upper[valid]])
    return result


def trade_size_marks(
    trade_times_ms: np.ndarray,
    trade_volumes: np.ndarray,
    window_ms: int = 3_600_000,
) -> np.ndarray:
    """m_k for a chronologically sorted stream of trade events only."""
    median = rolling_median_before(trade_volumes, trade_times_ms, window_ms)
    neutral = np.isnan(median) | (median <= 0.0)
    return np.where(neutral, 1.0, trade_volumes / np.where(neutral, 1.0, median))
