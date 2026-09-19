"""Causal buffered labels for adverse price moves and pure queue depletion."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True, nogil=True)
def target_a(
    quote_time_ms: np.ndarray,
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    sample_time_ms: np.ndarray,
    buffer_ms: int,
    horizon_ms: int,
) -> np.ndarray:
    out = np.zeros(len(sample_time_ms), np.uint8)
    for row in range(len(sample_time_ms)):
        now = np.searchsorted(quote_time_ms, sample_time_ms[row], side="right") - 1
        left = np.searchsorted(quote_time_ms, sample_time_ms[row] + buffer_ms, side="left")
        right = np.searchsorted(quote_time_ms, sample_time_ms[row] + buffer_ms + horizon_ms, side="right")
        if now < 0:
            continue
        for q in range(left, right):
            if bid_price[q] < bid_price[now] or ask_price[q] > ask_price[now]:
                out[row] = 1
                break
    return out


@njit(cache=True, nogil=True)
def target_b(
    quote_time_ms: np.ndarray,
    bid_price: np.ndarray,
    bid_qty: np.ndarray,
    ask_price: np.ndarray,
    ask_qty: np.ndarray,
    sample_time_ms: np.ndarray,
    depletion_fraction: float,
    buffer_ms: int,
    horizon_ms: int,
) -> np.ndarray:
    out = np.zeros(len(sample_time_ms), np.uint8)
    for row in range(len(sample_time_ms)):
        now = np.searchsorted(quote_time_ms, sample_time_ms[row], side="right") - 1
        left = np.searchsorted(quote_time_ms, sample_time_ms[row] + buffer_ms, side="left")
        right = np.searchsorted(quote_time_ms, sample_time_ms[row] + buffer_ms + horizon_ms, side="right")
        if now < 0 or left >= right:
            continue
        pure = True
        depleted = False
        bid_threshold = bid_qty[now] * (1.0 - depletion_fraction)
        ask_threshold = ask_qty[now] * (1.0 - depletion_fraction)
        for q in range(left, right):
            if bid_price[q] != bid_price[now] or ask_price[q] != ask_price[now]:
                pure = False
                break
            if bid_qty[q] < bid_threshold or ask_qty[q] < ask_threshold:
                depleted = True
        if pure and depleted:
            out[row] = 1
    return out
