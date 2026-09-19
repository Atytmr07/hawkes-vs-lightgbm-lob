"""Causal L1 state features: OFI, micro-price, spread, queues and ranks."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


M0_NAMES = [
    "ofi_l1", "microprice_basis", "spread", "bid_qty", "ask_qty",
    "bid_queue_rank_1h", "ask_queue_rank_1h",
]


@njit(cache=True, nogil=True)
def _fenwick_ranks(
    value_indices: np.ndarray,
    sample_time_ms: np.ndarray,
    active: np.ndarray,
    distinct_values: int,
    window_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = np.zeros(distinct_values + 1, np.int64)
    ranks = np.ones(len(value_indices), np.float64)
    counts = np.zeros(len(value_indices), np.int64)
    left = 0
    active_count = 0
    for right in range(len(value_indices)):
        while left < right and sample_time_ms[right] - sample_time_ms[left] > window_ms:
            if active[left]:
                index = value_indices[left] + 1
                while index <= distinct_values:
                    tree[index] -= 1
                    index += index & -index
                active_count -= 1
            left += 1
        if active[right]:
            index = value_indices[right] + 1
            while index <= distinct_values:
                tree[index] += 1
                index += index & -index
            active_count += 1
            total_le = 0
            index = value_indices[right] + 1
            while index > 0:
                total_le += tree[index]
                index -= index & -index
            ranks[right] = total_le / active_count
        counts[right] = active_count
    return ranks, counts


def _rolling_rank(values: np.ndarray, times: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(values)
    indices = np.searchsorted(unique, values).astype(np.int64)
    return _fenwick_ranks(indices, times, active, len(unique), 3_600_000)


def l1_features_at_samples(
    quote_time_ms: np.ndarray,
    bid_price: np.ndarray,
    bid_qty: np.ndarray,
    ask_price: np.ndarray,
    ask_qty: np.ndarray,
    sample_time_ms: np.ndarray,
    *,
    samples_per_hour: int,
    max_quote_age_ms: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(quote_time_ms, sample_time_ms, side="right") - 1
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    valid &= sample_time_ms - quote_time_ms[indices] <= max_quote_age_ms
    bp = bid_price[indices]
    bq = bid_qty[indices]
    ap = ask_price[indices]
    aq = ask_qty[indices]
    spread = ap - bp
    mid = (ap + bp) / 2.0
    micro = (ap * bq + bp * aq) / np.maximum(bq + aq, 1e-12)

    ofi = np.zeros(len(indices), np.float64)
    if len(indices) > 1:
        ofi[1:] = (
            (bp[1:] >= bp[:-1]) * bq[1:]
            - (bp[1:] <= bp[:-1]) * bq[:-1]
            - (ap[1:] <= ap[:-1]) * aq[1:]
            + (ap[1:] >= ap[:-1]) * aq[:-1]
        )
    bid_rank, active_count = _rolling_rank(bq, sample_time_ms, valid)
    ask_rank, _ = _rolling_rank(aq, sample_time_ms, valid)
    valid &= active_count >= min(60, samples_per_hour)
    features = np.column_stack((ofi, micro - mid, spread, bq, aq, bid_rank, ask_rank))
    return features, valid
