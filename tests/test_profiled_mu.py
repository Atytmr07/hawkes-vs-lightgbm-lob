import numpy as np

from src.models.hawkes_mle import BETAS, fit_hawkes, profile_mu_given_alpha


def _synthetic_events(seed: int, n: int = 800):
    rng = np.random.default_rng(seed)
    gaps = rng.integers(1_000_000, 100_000_000, size=n)
    times = np.cumsum(gaps).astype(np.int64)
    codes = rng.integers(0, 10, size=n, dtype=np.uint8)
    return times, codes


def test_profiled_mu_matches_joint_fit_at_its_own_alpha():
    """At a joint optimum, mu is already at its own conditional optimum
    given the jointly-fit alpha -- profiling mu alone at that same alpha
    should reproduce the joint fit's mu closely, including any boundary
    (mu ~ 1e-9) target types."""
    times, codes = _synthetic_events(seed=11)
    joint = fit_hawkes(times, codes, max_events=800, maxiter=200, tie_treatment="mask_touch_trade")
    profiled = profile_mu_given_alpha(times, codes, joint.alpha, joint.betas, max_events=800, tie_treatment="mask_touch_trade")
    assert np.allclose(profiled, joint.mu, rtol=5e-2, atol=5e-2)


def test_profiled_mu_handles_boundary_case_without_raising():
    """A large, foreign alpha can make a target type's excitation alone
    already explain its observed rate at mu -> 0 (no interior root of the
    stationary condition) -- this must clamp to the lower bound, not
    raise (see the brentq ValueError this guards against)."""
    times, codes = _synthetic_events(seed=13)
    rich_alpha = np.full((10, 10, len(BETAS)), 50.0)
    mu = profile_mu_given_alpha(times, codes, rich_alpha, BETAS, max_events=800, tie_treatment="mask_touch_trade")
    assert np.all(mu >= 1e-9)
    assert np.all(np.isfinite(mu))


def test_profiled_mu_is_nonnegative_and_finite_on_a_real_alpha_shape():
    times, codes = _synthetic_events(seed=17)
    joint = fit_hawkes(times, codes, max_events=800, maxiter=50)
    profiled = profile_mu_given_alpha(times, codes, joint.alpha, joint.betas, max_events=800)
    assert profiled.shape == (10,)
    assert np.all(profiled >= 1e-9)
    assert np.all(np.isfinite(profiled))
