import numpy as np

from src.models.hawkes_mle import fit_hawkes


def test_hawkes_fit_is_subcritical():
    rng = np.random.default_rng(7)
    gaps = rng.integers(1_000_000, 100_000_000, size=500)
    times = np.cumsum(gaps).astype(np.int64)
    codes = rng.integers(0, 10, size=500, dtype=np.uint8)
    fit = fit_hawkes(times, codes, max_events=500, maxiter=5)
    assert fit.spectral_radius < 1.0
    assert fit.stationarity_scale <= 1.0
