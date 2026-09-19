"""Non-integrated proposal for exact-ms touch-crossing attribution.

This module deliberately has no imports from the production pipeline.  It is a
candidate treatment for the tie-sensitive quote->trade Hawkes block documented
in PROJECT_SPEC §12.4, not an active data transformation.

The proposal marks exact-millisecond trade/quote groups that contain both a
trade and a touch-price move.  A later, explicitly approved estimator could
exclude only the cross-feed causal edge inside such groups (or introduce an
``ambiguous`` event state), while retaining the events for marginal intensities.
"""

from __future__ import annotations

import numpy as np


# Event codes in src.data_loader.aligner.  Duplicated intentionally so this
# proposal cannot silently become a production dependency.
TRADE_BUY = 8
TRADE_SELL = 9
# Price-move codes in src.data_loader.aligner.EVENT_TYPES.  The earlier draft
# accidentally mixed quantity-up codes into this set; keeping the explicit
# constants here makes the proposal auditable without importing production.
TOUCH_PRICE_MOVES = frozenset((2, 3, 6, 7))


def exact_ms_touch_crossing_mask(
    transaction_time: np.ndarray,
    event_code: np.ndarray,
) -> np.ndarray:
    """Return rows in ambiguous exact-ms trade/touch-price groups.

    Preconditions: both arrays are one-dimensional, equal length, and sorted
    by transaction_time.  A group is marked only when it has at least one trade
    and at least one touch-price event at the identical raw millisecond.
    Quantity-only quote changes are intentionally not included.
    """
    times = np.asarray(transaction_time, dtype=np.int64)
    codes = np.asarray(event_code, dtype=np.uint8)
    if times.ndim != 1 or codes.ndim != 1 or len(times) != len(codes):
        raise ValueError("transaction_time and event_code must be equal-length 1-D arrays")

    marked = np.zeros(len(times), dtype=bool)
    if len(times) == 0:
        return marked
    starts = np.r_[True, times[1:] != times[:-1]]
    bounds = np.r_[np.flatnonzero(starts), len(times)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        group = codes[left:right]
        has_trade = bool(np.any((group == TRADE_BUY) | (group == TRADE_SELL)))
        has_touch_move = bool(np.any(np.isin(group, tuple(TOUCH_PRICE_MOVES))))
        if has_trade and has_touch_move:
            marked[left:right] = True
    return marked
