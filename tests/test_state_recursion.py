import numpy as np

from src.features.ewma_bank import (
    BETAS, brute_force_state, brute_force_weighted_state, state_at_samples,
    state_before_selected_events, weighted_state_at_samples,
    weighted_state_before_selected_events,
)


def test_numba_state_matches_brute_force_convolution():
    times = np.array([0, 2_000_000, 2_000_100, 15_000_000, 50_000_000], dtype=np.int64)
    codes = np.array([0, 1, 0, 9, 3], dtype=np.uint8)
    samples = np.array([0, 2_000_100, 10_000_000, 50_000_000, 75_000_000], dtype=np.int64)
    actual = state_at_samples(times, codes, samples, BETAS)
    expected = brute_force_state(times, codes, samples, BETAS)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_weighted_state_at_samples_matches_brute_force_convolution():
    times = np.array([0, 2_000_000, 2_000_100, 15_000_000, 50_000_000], dtype=np.int64)
    codes = np.array([0, 1, 0, 9, 3], dtype=np.uint8)
    weights = np.array([1.0, 0.0, 2.5, 3.2, 0.0])
    samples = np.array([0, 2_000_100, 10_000_000, 50_000_000, 75_000_000], dtype=np.int64)
    actual = weighted_state_at_samples(times, codes, weights, samples, BETAS)
    expected = brute_force_weighted_state(times, codes, weights, samples, BETAS)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_weighted_state_at_samples_zero_weight_gives_zero_column():
    # An event type whose every occurrence carries weight 0.0 (the M3 design
    # for non-trade event types) must contribute exactly zero, at every scale,
    # for the entire horizon -- not merely "small".
    times = np.array([0, 1_000_000, 5_000_000], dtype=np.int64)
    codes = np.array([2, 2, 2], dtype=np.uint8)
    weights = np.zeros(3)
    samples = np.array([0, 10_000_000], dtype=np.int64)
    actual = weighted_state_at_samples(times, codes, weights, samples, BETAS)
    np.testing.assert_array_equal(actual, 0.0)


def test_pre_jump_state_excludes_selected_event_itself():
    times = np.array([0, 1_000_000_000], dtype=np.int64)
    codes = np.array([2, 2], dtype=np.uint8)
    states = state_before_selected_events(times, codes, np.array([0, 1]), BETAS)
    np.testing.assert_allclose(states[0], 0.0)
    np.testing.assert_allclose(states[1, 2 * len(BETAS):(2 + 1) * len(BETAS)], np.exp(-BETAS))


def test_marked_state_is_additive_not_bilinear():
    times = np.array([0, 1_000_000_000], dtype=np.int64)
    codes = np.array([1, 1], dtype=np.uint8)
    marks = np.array([2.5, 3.0])
    states = weighted_state_before_selected_events(times, codes, marks, np.array([1]), BETAS)
    np.testing.assert_allclose(states[0, len(BETAS):2 * len(BETAS)], 2.5 * np.exp(-BETAS))
