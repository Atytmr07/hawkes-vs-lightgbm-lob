"""Warm-started rolling hourly Hawkes spectral-radius tracker for M3."""

from __future__ import annotations

import numpy as np

from src.models.hawkes_mle import HawkesFit, fit_hawkes

# STATUS_38DAY.md §1's two known bookTicker outages, largest-interval bounds
# (UTC). STATUS_STATIONARITY.md §7 traced the one confirmed stationarity-
# projection activation directly to the 2024-03-08 gap: the hour
# 16:00-17:00 is 89.4% trades with zero quote events from minute 5 onward,
# a degenerate, non-representative input for an hourly joint-process refit.
# Excluded here per that section's (previously unimplemented) recommendation.
_KNOWN_GAP_WINDOWS_NS = tuple(
    (
        np.datetime64(start, "ns").astype(np.int64),
        np.datetime64(end, "ns").astype(np.int64),
    )
    for start, end in (
        ("2024-02-28T17:32:13", "2024-02-28T18:15:20"),
        ("2024-03-08T16:03:08", "2024-03-08T17:09:20"),
    )
)


def rolling_hourly_spectral_radius(
    times_ns: np.ndarray,
    codes: np.ndarray,
    *,
    max_events: int = 250_000,
    maxiter: int = 5_000,
    exclude_windows_ns: tuple[tuple[int, int], ...] = _KNOWN_GAP_WINDOWS_NS,
) -> tuple[np.ndarray, np.ndarray, list[HawkesFit]]:
    """Warm-started hourly refit. ``maxiter`` was previously hard-coded to 15,
    on the §7.2 assumption that warm-starting drops convergence to under ten
    iterations. Measured directly (STATUS_SPECTRAL_CONVERGENCE.md): even a
    fully-converged prior-hour warm start can still need several hundred
    L-BFGS-B iterations for some event types, so 15 left every hourly fit
    non-converged (`result.success` never True) over a full test-week scan
    (167 windows). 2,000 (the maxiter already used by every other
    `fit_hawkes` call site in this codebase) got 165/167 hours fully
    converged; 5,000 was the empirically smallest tested budget that reached
    167/167 (1670/1670 per-target fits), at effectively the same measured
    wall-clock cost as 2,000 -- already-converged fits stop at their own
    L-BFGS-B criterion regardless of the cap, so raising it only costs more
    for the handful of slow-converging (target type, hour) pairs. See the
    status doc for the full before/after convergence and rho table.
    """
    hour_ns = 3_600_000_000_000
    boundaries = np.arange(times_ns[0] // hour_ns * hour_ns + hour_ns, times_ns[-1] + 1, hour_ns)
    radii = np.full(len(boundaries), np.nan)
    fits: list[HawkesFit] = []
    warm = None
    for index, boundary in enumerate(boundaries):
        window_start = int(boundary - hour_ns)
        window_end = int(boundary)
        if any(
            window_start < gap_end and gap_start < window_end
            for gap_start, gap_end in exclude_windows_ns
        ):
            continue
        left = np.searchsorted(times_ns, boundary - hour_ns, side="left")
        right = np.searchsorted(times_ns, boundary, side="left")
        if right - left < 100:
            continue
        warm = fit_hawkes(
            times_ns[left:right], codes[left:right],
            max_events=max_events, warm_start=warm, maxiter=maxiter,
        )
        # fit_hawkes unconditionally attaches _selected_state_cache (a
        # states array up to max_events x 50 floats, ~100MB at this
        # function's default max_events=250_000) so fit_seasonal_baseline
        # can reuse it without a second full recursion. Nothing here ever
        # calls fit_seasonal_baseline, and warm_start only reads
        # mu/alpha/alpha_mark, so the cache is pure waste for every hourly
        # fit this function produces -- left alive, a multi-day window's
        # worth of hourly fits accumulates tens of GB for no benefit (see
        # STATUS_M3.md; this caused a real MemoryError this session).
        if hasattr(warm, "_selected_state_cache"):
            del warm._selected_state_cache
        radii[index] = warm.spectral_radius
        fits.append(warm)
    return boundaries, radii, fits
