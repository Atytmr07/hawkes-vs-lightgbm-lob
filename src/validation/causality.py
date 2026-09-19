"""Directional diagnostics for tie-rule sensitivity of Hawkes Gamma."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

from src.data_loader.aligner import EVENT_TYPES


@njit(cache=True, nogil=True)
def _quotes_first_codes(times: np.ndarray, codes: np.ndarray) -> np.ndarray:
    result = codes.copy()
    left = 0
    while left < len(times):
        raw_ms = times[left] // 1_000_000
        right = left + 1
        while right < len(times) and times[right] // 1_000_000 == raw_ms:
            right += 1
        write = left
        for k in range(left, right):
            if codes[k] < 8:
                result[write] = codes[k]
                write += 1
        for k in range(left, right):
            if codes[k] >= 8:
                result[write] = codes[k]
                write += 1
        left = right
    return result


@njit(cache=True, nogil=True)
def _touch_trade_counts(times: np.ndarray, codes: np.ndarray) -> tuple[int, int, int, int]:
    groups = 0
    rows = 0
    trades = 0
    touches = 0
    left = 0
    while left < len(times):
        raw_ms = times[left] // 1_000_000
        right = left + 1
        while right < len(times) and times[right] // 1_000_000 == raw_ms:
            right += 1
        group_trades = 0
        group_touches = 0
        for k in range(left, right):
            code = codes[k]
            if code >= 8:
                group_trades += 1
            elif code == 2 or code == 3 or code == 6 or code == 7:
                group_touches += 1
        if group_trades and group_touches:
            groups += 1
            rows += right - left
            trades += group_trades
            touches += group_touches
        left = right
    return groups, rows, trades, touches


def invert_cross_feed_ties(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert feed order at exact-ms ties without changing the event multiset.

    Input is the default trades-first aligned stream.  The returned timestamps
    retain the same deterministic jitter slots, while codes are stably
    partitioned to quotes-first inside each raw millisecond.  Relative order
    within each feed is unchanged.  This isolates ordering from event typing.
    """
    times = np.asarray(event_times_ns, dtype=np.int64)
    codes = np.asarray(event_codes, dtype=np.uint8)
    if times.ndim != 1 or codes.ndim != 1 or len(times) != len(codes):
        raise ValueError("event_times_ns and event_codes must be equal-length 1-D arrays")
    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        raise ValueError("event_times_ns must be nondecreasing")
    return times.copy(), _quotes_first_codes(times, codes)


def touch_trade_collision_diagnostic(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
) -> dict[str, int | float]:
    """Count exact-ms groups affected by touch/trade attribution ambiguity."""
    times = np.asarray(event_times_ns, dtype=np.int64)
    codes = np.asarray(event_codes, dtype=np.uint8)
    if times.ndim != 1 or codes.ndim != 1 or len(times) != len(codes):
        raise ValueError("event_times_ns and event_codes must be equal-length 1-D arrays")
    groups, rows, trades, touches = _touch_trade_counts(times, codes)
    return {
        "ambiguous_groups": groups,
        "rows_in_ambiguous_groups": rows,
        "trades_in_ambiguous_groups": trades,
        "touches_in_ambiguous_groups": touches,
        "share_of_all_rows": rows / max(len(codes), 1),
    }


def gamma_change_diagnostic(
    default_gamma: np.ndarray,
    inverted_gamma: np.ndarray,
    *,
    top_k: int = 15,
) -> dict[str, object]:
    default = np.asarray(default_gamma, np.float64)
    inverted = np.asarray(inverted_gamma, np.float64)
    if default.shape != (10, 10) or inverted.shape != (10, 10):
        raise ValueError("Gamma matrices must both be 10x10")
    delta = inverted - default
    blocks = {
        "quote_to_quote": (slice(0, 8), slice(0, 8)),
        "trade_to_quote": (slice(0, 8), slice(8, 10)),
        "quote_to_trade": (slice(8, 10), slice(0, 8)),
        "trade_to_trade": (slice(8, 10), slice(8, 10)),
    }
    total_squared = float(np.sum(delta * delta))
    block_rows: dict[str, dict[str, float]] = {}
    for name, (target, source) in blocks.items():
        base_block = default[target, source]
        delta_block = delta[target, source]
        block_rows[name] = {
            "default_frobenius": float(np.linalg.norm(base_block)),
            "delta_frobenius": float(np.linalg.norm(delta_block)),
            "relative_change": float(np.linalg.norm(delta_block) / max(np.linalg.norm(base_block), 1e-12)),
            "share_of_total_squared_change": float(np.sum(delta_block * delta_block) / max(total_squared, 1e-24)),
            "max_absolute_change": float(np.max(np.abs(delta_block))),
        }
    flat_order = np.argsort(np.abs(delta).ravel())[::-1][:top_k]
    top_terms = []
    for flat in flat_order:
        target, source = np.unravel_index(flat, delta.shape)
        top_terms.append({
            "target": EVENT_TYPES[target], "source": EVENT_TYPES[source],
            "direction": (
                "trade_to_quote" if target < 8 <= source else
                "quote_to_trade" if source < 8 <= target else
                "quote_to_quote" if target < 8 and source < 8 else
                "trade_to_trade"
            ),
            "default": float(default[target, source]),
            "book_first": float(inverted[target, source]),
            "delta": float(delta[target, source]),
            "absolute_change": float(abs(delta[target, source])),
        })
    return {
        "overall_relative_frobenius_change": float(np.linalg.norm(delta) / max(np.linalg.norm(default), 1e-12)),
        "blocks": block_rows,
        "top_terms": top_terms,
    }
