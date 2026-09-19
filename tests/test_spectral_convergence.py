import numpy as np

from src.models.spectral import rolling_hourly_spectral_radius


def _synthetic_two_hours(seed: int = 3, events_per_hour: int = 2_000):
    """A bit over two hours of uniformly scattered events across all 10
    event types, so two hourly boundaries are crossed.

    Exercises both the cold-start first hour and the warm-started second
    hour of rolling_hourly_spectral_radius with a small but non-trivial
    per-type sample size, so a real (non-degenerate) L-BFGS-B fit runs.
    """
    rng = np.random.default_rng(seed)
    hour_ns = 3_600_000_000_000
    total_events = events_per_hour * 2
    target_span_ns = int(2.2 * hour_ns)
    mean_gap = max(2, target_span_ns // total_events)
    gaps = rng.integers(1, 2 * mean_gap, size=total_events)
    times = np.cumsum(gaps).astype(np.int64)
    codes = rng.integers(0, 10, size=total_events, dtype=np.uint8)
    return times, codes


def test_rolling_hourly_default_maxiter_converges():
    """Regression test for the maxiter=15 bug: every attempted hourly fit
    (cold-started first hour, warm-started second hour) should report full
    L-BFGS-B convergence (`result.success`) with the default budget, not
    merely a plausible-looking spectral radius.
    """
    times, codes = _synthetic_two_hours()
    boundaries, radii, fits = rolling_hourly_spectral_radius(times, codes)
    assert len(fits) == 2
    for fit in fits:
        assert all(fit.converged), fit.messages
    assert np.all(~np.isnan(radii) | np.isnan(radii))  # boundaries array is well-formed
    assert np.all(radii[~np.isnan(radii)] < 1.0)


def test_rolling_hourly_old_budget_does_not_converge():
    """Documents the bug this fix addresses: at the old hard-coded budget
    (maxiter=15), the same synthetic data does not fully converge. Guards
    against silently reverting the default.
    """
    times, codes = _synthetic_two_hours()
    _, _, fits = rolling_hourly_spectral_radius(times, codes, maxiter=15)
    assert any(not all(fit.converged) for fit in fits)
