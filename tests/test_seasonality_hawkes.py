import numpy as np

from src.models.hawkes_mle import fit_hawkes, fit_seasonal_baseline


def test_seasonal_baseline_is_positive_and_profiles_hawkes_objective():
    rng = np.random.default_rng(11)
    gaps = rng.integers(1_000_000, 200_000_000, size=600)
    times = np.cumsum(gaps).astype(np.int64) + 1_706_659_200_000_000_000
    codes = rng.integers(0, 10, size=len(times), dtype=np.uint8)
    fit = fit_hawkes(times, codes, max_events=600, maxiter=50)
    comparison = fit_seasonal_baseline(
        fit, times, codes, max_events=600, quadrature_seconds=300, maxiter=200
    )
    rates = fit.seasonal_baseline.rates(times[:20] // 1_000_000)
    assert np.all(rates > 0)
    assert comparison["seasonal_profile_loglik"] >= comparison["constant_profile_loglik"] - 1e-5
    assert len(comparison["converged"]) == 10
