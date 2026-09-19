"""Exact exponential state recursion for the 10-event, five-scale bank."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # keeps diagnostics importable before requirements are installed
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


BETAS = np.asarray([0.2, 1.0, 5.0, 25.0, 100.0], dtype=np.float64)
N_EVENT_TYPES = 10


@njit(cache=True, nogil=True, fastmath=True)
def state_at_samples(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    sample_times_ns: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """Return post-event EWMA states at causal sample times.

    Events at exactly the sample timestamp are included. State is carried across
    the complete input sequence; callers should concatenate days for continuous
    windows when cross-midnight warm-up matters.
    """
    p_count = len(betas)
    output = np.zeros((len(sample_times_ns), N_EVENT_TYPES * p_count), np.float64)
    state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    last_update = np.empty(N_EVENT_TYPES, np.int64)
    event_index = 0
    initial_time = event_times_ns[0] if len(event_times_ns) else (
        sample_times_ns[0] if len(sample_times_ns) else 0
    )
    last_update[:] = initial_time
    for sample_index in range(len(sample_times_ns)):
        sample_time = sample_times_ns[sample_index]
        while event_index < len(event_times_ns) and event_times_ns[event_index] <= sample_time:
            event_time = event_times_ns[event_index]
            code = event_codes[event_index]
            dt = (event_time - last_update[code]) * 1e-9
            if dt:
                for p in range(p_count):
                    state[code, p] *= np.exp(-betas[p] * dt)
            state[code, :] += 1.0
            last_update[code] = event_time
            event_index += 1
        for event_type in range(N_EVENT_TYPES):
            dt = (sample_time - last_update[event_type]) * 1e-9
            for p in range(p_count):
                output[sample_index, event_type * p_count + p] = (
                    state[event_type, p] * np.exp(-betas[p] * dt)
                )
    return output


@njit(cache=True, nogil=True, fastmath=True)
def weighted_state_at_samples(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    event_weights: np.ndarray,
    sample_times_ns: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """Marked analog of ``state_at_samples``: post-event EWMA states at causal
    sample times, where each event contributes ``event_weights[k]`` instead of
    a flat 1.0. Used for M3's linearized trade-size mark bank (§2.4, §3.2):
    callers pass 0.0 for event types that carry no mark, so only the marked
    event types' columns of the output are non-trivial.
    """
    p_count = len(betas)
    output = np.zeros((len(sample_times_ns), N_EVENT_TYPES * p_count), np.float64)
    state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    last_update = np.empty(N_EVENT_TYPES, np.int64)
    event_index = 0
    initial_time = event_times_ns[0] if len(event_times_ns) else (
        sample_times_ns[0] if len(sample_times_ns) else 0
    )
    last_update[:] = initial_time
    for sample_index in range(len(sample_times_ns)):
        sample_time = sample_times_ns[sample_index]
        while event_index < len(event_times_ns) and event_times_ns[event_index] <= sample_time:
            event_time = event_times_ns[event_index]
            code = event_codes[event_index]
            dt = (event_time - last_update[code]) * 1e-9
            if dt:
                for p in range(p_count):
                    state[code, p] *= np.exp(-betas[p] * dt)
            state[code, :] += event_weights[event_index]
            last_update[code] = event_time
            event_index += 1
        for event_type in range(N_EVENT_TYPES):
            dt = (sample_time - last_update[event_type]) * 1e-9
            for p in range(p_count):
                output[sample_index, event_type * p_count + p] = (
                    state[event_type, p] * np.exp(-betas[p] * dt)
                )
    return output


@njit(cache=True, nogil=True, fastmath=True)
def state_before_selected_events(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    selected_indices: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """EWMA state immediately before selected jumps, without self-excitation."""
    p_count = len(betas)
    output = np.zeros((len(selected_indices), N_EVENT_TYPES * p_count), np.float64)
    state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    selected_pointer = 0
    initial_time = event_times_ns[0] if len(event_times_ns) else 0
    last_update = np.empty(N_EVENT_TYPES, np.int64)
    last_update[:] = initial_time
    for k in range(len(event_times_ns)):
        event_time = event_times_ns[k]
        code = event_codes[k]
        dt = (event_time - last_update[code]) * 1e-9
        if dt:
            for p in range(p_count):
                state[code, p] *= np.exp(-betas[p] * dt)
        last_update[code] = event_time
        if selected_pointer < len(selected_indices) and k == selected_indices[selected_pointer]:
            for j in range(N_EVENT_TYPES):
                selected_dt = (event_time - last_update[j]) * 1e-9
                for p in range(p_count):
                    output[selected_pointer, j * p_count + p] = (
                        state[j, p] * np.exp(-betas[p] * selected_dt)
                    )
            selected_pointer += 1
        state[code, :] += 1.0
    return output


@njit(cache=True, nogil=True, fastmath=True)
def state_before_selected_events_mask_touch_trade(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    selected_indices: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """States with ambiguous same-time touch/trade parent edges removed.

    ``event_times_ns`` must use the observed timestamp resolution (milliseconds
    for Binance's transaction time), not the deterministic jittered timestamp.
    Within a timestamp group, same-feed history is retained.  If the group has
    both a trade and a touch-price move, only the artificial cross-feed history
    between those event classes is removed from the selected event's state.
    Both events remain in the process and affect every later timestamp.
    """
    p_count = len(betas)
    output = np.zeros((len(selected_indices), N_EVENT_TYPES * p_count), np.float64)
    state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    group_state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    selected_pointer = 0
    initial_time = event_times_ns[0] if len(event_times_ns) else 0
    last_update = np.empty(N_EVENT_TYPES, np.int64)
    last_update[:] = initial_time
    group_time = initial_time - 1
    group_ambiguous = False
    group_end = 0

    for k in range(len(event_times_ns)):
        event_time = event_times_ns[k]
        code = event_codes[k]
        if event_time != group_time:
            group_time = event_time
            group_state[:, :] = 0.0
            group_end = k + 1
            while group_end < len(event_times_ns) and event_times_ns[group_end] == event_time:
                group_end += 1
            has_trade = False
            has_touch = False
            for q in range(k, group_end):
                group_code = event_codes[q]
                if group_code >= 8:
                    has_trade = True
                elif group_code == 2 or group_code == 3 or group_code == 6 or group_code == 7:
                    has_touch = True
            group_ambiguous = has_trade and has_touch

        dt = (event_time - last_update[code]) * 1e-9
        if dt:
            for p in range(p_count):
                state[code, p] *= np.exp(-betas[p] * dt)
        last_update[code] = event_time

        if selected_pointer < len(selected_indices) and k == selected_indices[selected_pointer]:
            target_is_trade = code >= 8
            target_is_touch = code == 2 or code == 3 or code == 6 or code == 7
            for source in range(N_EVENT_TYPES):
                selected_dt = (event_time - last_update[source]) * 1e-9
                for p in range(p_count):
                    value = state[source, p] * np.exp(-betas[p] * selected_dt)
                    if group_ambiguous:
                        source_is_trade = source >= 8
                        source_is_touch = (
                            source == 2 or source == 3 or source == 6 or source == 7
                        )
                        if (target_is_trade and source_is_touch) or (
                            target_is_touch and source_is_trade
                        ):
                            value -= group_state[source, p]
                    output[selected_pointer, source * p_count + p] = max(value, 0.0)
            selected_pointer += 1

        state[code, :] += 1.0
        group_state[code, :] += 1.0
    return output


@njit(cache=True, nogil=True, fastmath=True)
def weighted_state_before_selected_events(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    event_weights: np.ndarray,
    selected_indices: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """Marked EWMA state immediately before selected jumps."""
    p_count = len(betas)
    output = np.zeros((len(selected_indices), N_EVENT_TYPES * p_count), np.float64)
    state = np.zeros((N_EVENT_TYPES, p_count), np.float64)
    selected_pointer = 0
    initial_time = event_times_ns[0] if len(event_times_ns) else 0
    last_update = np.empty(N_EVENT_TYPES, np.int64)
    last_update[:] = initial_time
    for k in range(len(event_times_ns)):
        event_time = event_times_ns[k]
        code = event_codes[k]
        dt = (event_time - last_update[code]) * 1e-9
        if dt:
            for p in range(p_count):
                state[code, p] *= np.exp(-betas[p] * dt)
        last_update[code] = event_time
        if selected_pointer < len(selected_indices) and k == selected_indices[selected_pointer]:
            for j in range(N_EVENT_TYPES):
                selected_dt = (event_time - last_update[j]) * 1e-9
                for p in range(p_count):
                    output[selected_pointer, j * p_count + p] = (
                        state[j, p] * np.exp(-betas[p] * selected_dt)
                    )
            selected_pointer += 1
        state[code, :] += event_weights[k]
    return output


def brute_force_state(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    sample_times_ns: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """Reference O(events × samples × scales) convolution used only in tests."""
    out = np.zeros((len(sample_times_ns), N_EVENT_TYPES * len(betas)))
    for row, sample_time in enumerate(sample_times_ns):
        eligible = event_times_ns <= sample_time
        ages = (sample_time - event_times_ns[eligible]) * 1e-9
        codes = event_codes[eligible]
        for code, age in zip(codes, ages, strict=True):
            out[row, code * len(betas):(code + 1) * len(betas)] += np.exp(-betas * age)
    return out


def brute_force_weighted_state(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    event_weights: np.ndarray,
    sample_times_ns: np.ndarray,
    betas: np.ndarray = BETAS,
) -> np.ndarray:
    """Reference convolution for ``weighted_state_at_samples``, used only in tests."""
    out = np.zeros((len(sample_times_ns), N_EVENT_TYPES * len(betas)))
    for row, sample_time in enumerate(sample_times_ns):
        eligible = event_times_ns <= sample_time
        ages = (sample_time - event_times_ns[eligible]) * 1e-9
        codes = event_codes[eligible]
        weights = event_weights[eligible]
        for code, age, weight in zip(codes, ages, weights, strict=True):
            out[row, code * len(betas):(code + 1) * len(betas)] += weight * np.exp(-betas * age)
    return out


def feature_names() -> list[str]:
    return [f"z_event{j}_beta{beta:g}" for j in range(10) for beta in BETAS]


def mark_feature_names(event_types: tuple[int, ...] = (8, 9)) -> list[str]:
    """Names for the M3 trade-size mark bank columns (§2.4, §3.2).

    Only trade_buy (8) and trade_sell (9) carry a nonzero mark by design
    (see ``STATUS_M3.md``); the other 8 event types' columns of
    ``weighted_state_at_samples`` are identically zero under that weighting
    and are not exposed as features.
    """
    return [f"z_mark_event{j}_beta{beta:g}" for j in event_types for beta in BETAS]
