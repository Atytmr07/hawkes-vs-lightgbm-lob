import numpy as np

from src.features.marks import rolling_median_before, trade_size_marks


def _brute_force_rolling_median_before(values: np.ndarray, times_ms: np.ndarray, window_ms: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        left = np.searchsorted(times_ms, times_ms[i] - window_ms, side="left")
        window = values[left:i]  # strictly before i: causal, excludes current
        if len(window):
            out[i] = np.median(window)
    return out


def test_rolling_median_before_matches_brute_force_random():
    rng = np.random.default_rng(11)
    n = 400
    times_ms = np.cumsum(rng.integers(0, 5_000, size=n)).astype(np.int64)
    values = rng.exponential(scale=3.0, size=n)
    actual = rolling_median_before(values, times_ms, window_ms=20_000)
    expected = _brute_force_rolling_median_before(values, times_ms, window_ms=20_000)
    nan_actual = np.isnan(actual)
    np.testing.assert_array_equal(nan_actual, np.isnan(expected))
    np.testing.assert_allclose(actual[~nan_actual], expected[~nan_actual], rtol=1e-12)


def test_rolling_median_before_matches_brute_force_with_ties():
    # Repeated values exercise the Fenwick bucket-collision path.
    rng = np.random.default_rng(5)
    n = 300
    times_ms = np.cumsum(rng.integers(0, 1_000, size=n)).astype(np.int64)
    values = rng.integers(1, 6, size=n).astype(np.float64)  # only 5 distinct values
    actual = rolling_median_before(values, times_ms, window_ms=3_000)
    expected = _brute_force_rolling_median_before(values, times_ms, window_ms=3_000)
    nan_actual = np.isnan(actual)
    np.testing.assert_array_equal(nan_actual, np.isnan(expected))
    np.testing.assert_allclose(actual[~nan_actual], expected[~nan_actual], rtol=1e-12)


def test_trade_size_marks_excludes_self_and_defaults_to_neutral():
    # First trade has no history -> neutral mark 1.0, regardless of its own size.
    times_ms = np.array([0, 1_000, 2_000, 3_000], dtype=np.int64)
    volumes = np.array([100.0, 2.0, 2.0, 2.0])
    marks = trade_size_marks(times_ms, volumes, window_ms=10_000)
    assert marks[0] == 1.0
    # Trades 2..4 see a trailing median of 2.0 or 100/2/2 depending on position;
    # in particular mark[1] uses only trade 0's volume (100.0) as the "median"
    # of a single-element window, so mark[1] = 2.0 / 100.0.
    np.testing.assert_allclose(marks[1], 2.0 / 100.0)


def test_trade_size_marks_is_causal_not_leaking_future():
    # A huge trade placed AFTER event k must not affect m_k.
    times_ms = np.array([0, 1_000, 2_000], dtype=np.int64)
    volumes_baseline = np.array([1.0, 1.0, 1.0])
    volumes_future_spike = np.array([1.0, 1.0, 1_000_000.0])
    marks_baseline = trade_size_marks(times_ms, volumes_baseline, window_ms=10_000)
    marks_spike = trade_size_marks(times_ms, volumes_future_spike, window_ms=10_000)
    np.testing.assert_allclose(marks_baseline[:2], marks_spike[:2])
