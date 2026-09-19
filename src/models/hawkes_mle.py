"""Regularized multi-exponential Hawkes MLE with stratified event subsampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.ewma_bank import (
    BETAS, N_EVENT_TYPES, state_before_selected_events,
    state_before_selected_events_mask_touch_trade,
    weighted_state_before_selected_events,
)
from src.features.seasonality import SeasonalBaseline, seasonality_design


COMPENSATOR_CHUNK_EVENTS = 2_000_000


def _is_nondecreasing(values: np.ndarray, chunk_size: int = 5_000_000) -> bool:
    """Check ordering without allocating a full-length boolean temporary."""
    if len(values) < 2:
        return True
    for start in range(1, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        if np.any(values[start:stop] < values[start - 1:stop - 1]):
            return False
    return True


@dataclass
class HawkesFit:
    mu: np.ndarray
    alpha: np.ndarray
    alpha_mark: np.ndarray | None
    betas: np.ndarray
    spectral_radius: float
    stationarity_scale: float
    converged: list[bool]
    messages: list[str]
    selected_events: int
    seasonal_baseline: SeasonalBaseline | None = None
    tie_treatment: str = "ordered"
    penalized_negative_loglik: float | None = None

    @property
    def gamma(self) -> np.ndarray:
        effective = self.alpha if self.alpha_mark is None else self.alpha + self.alpha_mark
        return np.sum(effective / self.betas[None, None, :], axis=2)

    def intensities(
        self,
        count_states: np.ndarray,
        mark_states: np.ndarray | None = None,
        sample_time_ms: np.ndarray | None = None,
    ) -> np.ndarray:
        flat = self.alpha.reshape(N_EVENT_TYPES, -1)
        if self.seasonal_baseline is None:
            baseline = self.mu[None, :]
        else:
            if sample_time_ms is None:
                raise ValueError("sample_time_ms is required for a seasonal Hawkes fit")
            baseline = self.seasonal_baseline.rates(sample_time_ms)
        result = baseline + count_states @ flat.T
        if self.alpha_mark is not None:
            if mark_states is None:
                raise ValueError("mark_states are required for a marked Hawkes fit")
            result = result + mark_states @ self.alpha_mark.reshape(N_EVENT_TYPES, -1).T
        return np.maximum(result, 1e-12)


def _stratified_indices(codes: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    allocation = max(1, maximum // N_EVENT_TYPES)
    for event_type in range(N_EVENT_TYPES):
        candidates = np.flatnonzero(codes == event_type)
        if len(candidates) > allocation:
            candidates = np.sort(rng.choice(candidates, allocation, replace=False))
        parts.append(candidates)
    return np.sort(np.concatenate(parts)).astype(np.int64)


def fit_hawkes(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    *,
    max_events: int = 10_000_000,
    l2: float = 1e-6,
    max_spectral_radius: float = 0.995,
    seed: int = 20240330,
    warm_start: HawkesFit | None = None,
    maxiter: int = 100,
    event_marks: np.ndarray | None = None,
    tie_treatment: str = "ordered",
) -> HawkesFit:
    """Fit nonnegative rows by L-BFGS-B, then enforce the subcritical set.

    The event log term is inverse-probability weighted after type-stratified
    subsampling. The compensator always uses every supplied event. The final
    radial projection is the exact stationarity safeguard used by downstream
    code and is reported so boundary saturation cannot be hidden.
    """
    from scipy.optimize import minimize

    if tie_treatment not in ("ordered", "mask_touch_trade"):
        raise ValueError("tie_treatment must be 'ordered' or 'mask_touch_trade'")
    if tie_treatment == "mask_touch_trade" and event_marks is not None:
        raise NotImplementedError("masked exact-ms attribution is not implemented for marked fits")
    if len(event_times_ns) < 2:
        raise ValueError("at least two events are required")
    raw_times = np.asarray(event_times_ns, np.int64)
    raw_codes = np.asarray(event_codes, np.uint8)
    if _is_nondecreasing(raw_times):
        times = raw_times
        codes = raw_codes
        marks = None if event_marks is None else np.asarray(event_marks, np.float64)
    else:
        order = np.argsort(raw_times, kind="stable")
        times = raw_times[order]
        codes = raw_codes[order]
        marks = None if event_marks is None else np.asarray(event_marks, np.float64)[order]
    if marks is not None and (len(marks) != len(times) or np.any(marks < 0)):
        raise ValueError("event_marks must be nonnegative and match events")
    selected = _stratified_indices(codes, min(max_events, len(codes)), seed)
    if tie_treatment == "mask_touch_trade":
        # Binance T has millisecond resolution.  Jitter is an ordering device,
        # not measured elapsed time, so it must not create decay inside a tie.
        state_times = (times // 1_000_000) * 1_000_000
        states = state_before_selected_events_mask_touch_trade(
            state_times, codes, selected, BETAS
        )
    else:
        state_times = times
        states = state_before_selected_events(times, codes, selected, BETAS)
    mark_states = None if marks is None else weighted_state_before_selected_events(
        times, codes, marks, selected, BETAS
    )
    duration = max((state_times[-1] - state_times[0]) * 1e-9, 1e-6)
    compensator = np.zeros((N_EVENT_TYPES, len(BETAS)), np.float64)
    mark_compensator = None if marks is None else np.zeros_like(compensator)
    # The full 31-day window contains hundreds of millions of events.  Compute
    # exactly the same compensator sum in bounded chunks instead of materializing
    # one full-length float64 age/exp array per beta.
    for start in range(0, len(times), COMPENSATOR_CHUNK_EVENTS):
        stop = min(start + COMPENSATOR_CHUNK_EVENTS, len(times))
        chunk_codes = codes[start:stop]
        ages = (state_times[-1] - state_times[start:stop]).astype(np.float64) * 1e-9
        for p, beta in enumerate(BETAS):
            weights = (1.0 - np.exp(-beta * ages)) / beta
            compensator[:, p] += np.bincount(
                chunk_codes, weights=weights, minlength=N_EVENT_TYPES
            )
            if marks is not None:
                mark_compensator[:, p] += np.bincount(
                    chunk_codes,
                    weights=weights * marks[start:stop],
                    minlength=N_EVENT_TYPES,
                )
    comp_flat = compensator.reshape(-1)
    if mark_compensator is not None:
        comp_flat = np.concatenate((comp_flat, mark_compensator.reshape(-1)))
    counts = np.bincount(codes, minlength=N_EVENT_TYPES).astype(np.float64)
    mu = np.empty(N_EVENT_TYPES, np.float64)
    alpha = np.zeros((N_EVENT_TYPES, N_EVENT_TYPES, len(BETAS)), np.float64)
    alpha_mark = None if marks is None else np.zeros_like(alpha)
    converged: list[bool] = []
    messages: list[str] = []
    objective_values: list[float] = []
    selected_codes = codes[selected]
    for target_type in range(N_EVENT_TYPES):
        x = states[selected_codes == target_type]
        if mark_states is not None:
            x = np.column_stack((x, mark_states[selected_codes == target_type]))
        sample_count = len(x)
        if sample_count == 0:
            mu[target_type] = max(counts[target_type] / duration, 1e-9)
            converged.append(False)
            messages.append("no sampled target events")
            objective_values.append(float("nan"))
            continue
        weight = counts[target_type] / sample_count
        parameter_scale = np.maximum(np.mean(x, axis=0), 1.0)
        scaled_x = x / parameter_scale
        scaled_compensator = comp_flat / parameter_scale
        initial = np.full(1 + x.shape[1], 1e-7)
        initial[0] = max(counts[target_type] / duration * 0.5, 1e-6)
        if warm_start is not None:
            initial[0] = warm_start.mu[target_type]
            warm_parameters = warm_start.alpha[target_type].reshape(-1)
            if marks is not None:
                if warm_start.alpha_mark is None:
                    warm_parameters = np.concatenate((warm_parameters, np.zeros_like(warm_parameters)))
                else:
                    warm_parameters = np.concatenate((warm_parameters, warm_start.alpha_mark[target_type].reshape(-1)))
            initial[1:] = warm_parameters * parameter_scale

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            intensity = np.maximum(theta[0] + scaled_x @ theta[1:], 1e-12)
            actual_alpha = theta[1:] / parameter_scale
            value = -weight * np.log(intensity).sum() + theta[0] * duration + scaled_compensator @ theta[1:]
            value += 0.5 * l2 * np.dot(actual_alpha, actual_alpha)
            residual = -weight / intensity
            gradient = np.empty_like(theta)
            gradient[0] = residual.sum() + duration
            gradient[1:] = scaled_x.T @ residual + scaled_compensator + l2 * actual_alpha / parameter_scale
            return float(value), gradient

        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            bounds=[(1e-9, None)] * len(initial),
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 100},
        )
        mu[target_type] = result.x[0]
        fitted = result.x[1:] / parameter_scale
        split = N_EVENT_TYPES * len(BETAS)
        alpha[target_type] = fitted[:split].reshape(N_EVENT_TYPES, len(BETAS))
        if alpha_mark is not None:
            alpha_mark[target_type] = fitted[split:].reshape(N_EVENT_TYPES, len(BETAS))
        converged.append(bool(result.success))
        messages.append(str(result.message))
        objective_values.append(float(result.fun))

    effective = alpha if alpha_mark is None else alpha + alpha_mark
    gamma = np.sum(effective / BETAS[None, None, :], axis=2)
    radius = float(max(abs(np.linalg.eigvals(gamma))))
    scale = 1.0
    if radius >= max_spectral_radius:
        scale = max_spectral_radius / radius
        alpha *= scale
        if alpha_mark is not None:
            alpha_mark *= scale
        effective = alpha if alpha_mark is None else alpha + alpha_mark
        radius = float(max(abs(np.linalg.eigvals(np.sum(effective / BETAS[None, None, :], axis=2)))))
    fitted = HawkesFit(
        mu, alpha, alpha_mark, BETAS.copy(), radius, scale, converged, messages,
        len(selected), None, tie_treatment, float(np.nansum(objective_values)),
    )
    # Reused immediately by fit_seasonal_baseline on the same arrays.  This
    # avoids a second full event-stream recursion while remaining an ephemeral
    # in-memory optimization (it is not serialized into model results).
    fitted._selected_state_cache = {
        "times_id": id(event_times_ns),
        "codes_id": id(event_codes),
        "event_count": len(times),
        "seed": seed,
        "maximum": min(max_events, len(codes)),
        "selected": selected,
        "states": states,
    }
    return fitted


def profile_mu_given_alpha(
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    alpha: np.ndarray,
    betas: np.ndarray,
    *,
    max_events: int = 100_000,
    seed: int = 20240330,
    tie_treatment: str = "ordered",
) -> np.ndarray:
    """Profile mu_i by exact 1-D MLE for each target type, holding a fixed
    (typically transferred) alpha constant -- the estimator the transfer
    protocol (PROJECT_SPEC.md Section 2.3, "re-estimate only mu(t) on the
    target asset") actually specifies, as opposed to the simplified
    empirical-rate stand-in (counts_i / duration) used in this project's
    first transfer-ladder pass (STATUS_TRANSFER_LADDER.md).

    Derivation: for target type i, holding alpha_i fixed, the same
    per-event weighted log-likelihood fit_hawkes optimizes jointly reduces
    to a function of mu_i alone (the compensator . alpha_i term and the L2
    penalty on alpha do not depend on mu_i, so they drop out of the
    stationary condition):

        d/dmu_i [ -weight * sum_k log(mu_i + exc_k) + mu_i * T ] = 0
        <=>  sum_k 1 / (mu_i + exc_k) = T / weight

    where exc_k = x_k . alpha_i is the fixed excitation at each selected
    type-i event's pre-event state x_k, T is the window duration, and
    weight = counts_i / n_selected_i is the same inverse-probability
    weight fit_hawkes uses to unbias the stratified subsample. The left
    side (call it h(mu)) is strictly decreasing in mu_i for mu_i > 0 (each
    term shrinks as mu_i grows), so h has at most one positive root -- but
    unlike the joint fit (where alpha is also free to move to keep this
    interior), a *fixed*, foreign (transferred) alpha's excitation can, at
    some target types, already explain as much or more of the observed
    rate than the equation asks for even at mu_i -> 0 (h(0+) <= 0): the
    boundary mu_i = 1e-9 is then the correct constrained MLE optimum, not
    a bug to bracket around. Both cases are handled explicitly below
    rather than assuming an interior root always exists.
    """
    from scipy.optimize import brentq

    if tie_treatment not in ("ordered", "mask_touch_trade"):
        raise ValueError("tie_treatment must be 'ordered' or 'mask_touch_trade'")
    raw_times = np.asarray(event_times_ns, np.int64)
    raw_codes = np.asarray(event_codes, np.uint8)
    if _is_nondecreasing(raw_times):
        times, codes = raw_times, raw_codes
    else:
        order = np.argsort(raw_times, kind="stable")
        times, codes = raw_times[order], raw_codes[order]
    selected = _stratified_indices(codes, min(max_events, len(codes)), seed)
    if tie_treatment == "mask_touch_trade":
        state_times = (times // 1_000_000) * 1_000_000
        states = state_before_selected_events_mask_touch_trade(state_times, codes, selected, betas)
    else:
        state_times = times
        states = state_before_selected_events(times, codes, selected, betas)
    duration = max((state_times[-1] - state_times[0]) * 1e-9, 1e-6)
    counts = np.bincount(codes, minlength=N_EVENT_TYPES).astype(np.float64)
    selected_codes = codes[selected]
    alpha_flat = alpha.reshape(N_EVENT_TYPES, -1)

    mu = np.empty(N_EVENT_TYPES, np.float64)
    for target_type in range(N_EVENT_TYPES):
        x = states[selected_codes == target_type]
        n_k = len(x)
        if n_k == 0 or counts[target_type] == 0:
            mu[target_type] = max(counts[target_type] / duration, 1e-9)
            continue
        exc = np.maximum(x @ alpha_flat[target_type], 0.0)
        weight = counts[target_type] / n_k
        rhs = duration / weight

        def stationary_condition(candidate_mu: float, exc=exc, rhs=rhs) -> float:
            return float(np.sum(1.0 / (candidate_mu + exc)) - rhs)

        lo = 1e-9
        if stationary_condition(lo) <= 0:
            # Fixed alpha's excitation alone already explains this target
            # type's rate at the boundary -- the constrained optimum is
            # the boundary itself, not an interior root (see docstring).
            mu[target_type] = lo
            continue
        hi = max(n_k * weight / duration, lo * 2)
        while stationary_condition(hi) > 0:
            hi *= 2
        mu[target_type] = brentq(stationary_condition, lo, hi, xtol=1e-12, rtol=1e-12)
    return mu


def _positive_harmonic_baseline(
    parameters: np.ndarray,
    harmonic_design: np.ndarray,
    funding_design: np.ndarray,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact §3.3 form and its derivatives under a positivity map."""
    eta = parameters[0]
    coefficients = parameters[1:17]
    funding = parameters[17:23]
    pairs = coefficients.reshape(8, 2)
    amplitudes = np.sqrt(np.sum(pairs * pairs, axis=1) + epsilon)
    exp_eta = np.exp(eta)
    a0 = epsilon + exp_eta + amplitudes.sum()
    base = a0 + harmonic_design @ coefficients
    jump_linear = funding_design @ funding
    jump = np.exp(np.clip(jump_linear, -20.0, 20.0))
    rate = np.maximum(base * jump, epsilon)
    derivative = np.empty((len(rate), 23), np.float64)
    derivative[:, 0] = exp_eta * jump
    amplitude_gradient = (pairs / amplitudes[:, None]).reshape(-1)
    derivative[:, 1:17] = jump[:, None] * (harmonic_design + amplitude_gradient[None, :])
    derivative[:, 17:23] = rate[:, None] * funding_design
    return rate, derivative


def fit_seasonal_baseline(
    fit: HawkesFit,
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    *,
    max_events: int = 100_000,
    seed: int = 20240330,
    quadrature_seconds: int = 60,
    maxiter: int = 1_000,
) -> dict[str, object]:
    """Profile §3.3 μ_i(t) inside the Hawkes likelihood, holding kernels fixed.

    The event term uses the same type-stratified inverse-probability weighting
    as the kernel MLE. The baseline compensator is integrated on a uniform UTC
    grid. Since excitation compensators are unchanged, the reported likelihood
    difference is the exact profiled baseline contribution under that grid.
    """
    from scipy.optimize import minimize

    raw_times = np.asarray(event_times_ns, np.int64)
    raw_codes = np.asarray(event_codes, np.uint8)
    if _is_nondecreasing(raw_times):
        times = raw_times
        codes = raw_codes
    else:
        order = np.argsort(raw_times, kind="stable")
        times = raw_times[order]
        codes = raw_codes[order]
    maximum = min(max_events, len(codes))
    cache = getattr(fit, "_selected_state_cache", None)
    if (
        cache is not None
        and cache["times_id"] == id(event_times_ns)
        and cache["codes_id"] == id(event_codes)
        and cache["event_count"] == len(times)
        and cache["seed"] == seed
        and cache["maximum"] == maximum
    ):
        selected = cache["selected"]
        states = cache["states"]
    else:
        selected = _stratified_indices(codes, maximum, seed)
        if fit.tie_treatment == "mask_touch_trade":
            coarse_times = (times // 1_000_000) * 1_000_000
            states = state_before_selected_events_mask_touch_trade(
                coarse_times, codes, selected, fit.betas
            )
        else:
            states = state_before_selected_events(times, codes, selected, fit.betas)
    selected_codes = codes[selected]
    selected_ms = times[selected] // 1_000_000
    event_design, _ = seasonality_design(selected_ms)
    event_harmonic, event_funding = event_design[:, :16], event_design[:, 16:]
    duration = max((times[-1] - times[0]) * 1e-9, 1e-6)
    grid_count = max(2, int(np.ceil(duration / quadrature_seconds)))
    grid_ns = np.linspace(times[0], times[-1], grid_count, dtype=np.int64)
    grid_design, _ = seasonality_design(grid_ns // 1_000_000)
    grid_harmonic, grid_funding = grid_design[:, :16], grid_design[:, 16:]
    grid_weight = duration / grid_count
    counts = np.bincount(codes, minlength=N_EVENT_TYPES).astype(np.float64)
    eta = np.empty(N_EVENT_TYPES)
    harmonic = np.empty((N_EVENT_TYPES, 16))
    funding = np.empty((N_EVENT_TYPES, 6))
    converged: list[bool] = []
    messages: list[str] = []
    constant_loglik = 0.0
    seasonal_loglik = 0.0
    for target in range(N_EVENT_TYPES):
        target_mask = selected_codes == target
        excitation = states[target_mask] @ fit.alpha[target].reshape(-1)
        h_event = event_harmonic[target_mask]
        f_event = event_funding[target_mask]
        sample_count = int(target_mask.sum())
        weight = counts[target] / max(sample_count, 1)
        constant_rate = np.maximum(fit.mu[target] + excitation, 1e-12)
        constant_loglik += float(weight * np.log(constant_rate).sum() - fit.mu[target] * duration)
        initial = np.zeros(23, np.float64)
        initial[0] = np.log(max(fit.mu[target], 1e-6))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            event_rate, event_derivative = _positive_harmonic_baseline(
                parameters, h_event, f_event
            )
            grid_rate, grid_derivative = _positive_harmonic_baseline(
                parameters, grid_harmonic, grid_funding
            )
            total_rate = np.maximum(event_rate + excitation, 1e-12)
            value = -weight * np.log(total_rate).sum() + grid_weight * grid_rate.sum()
            gradient = -weight * (event_derivative / total_rate[:, None]).sum(axis=0)
            gradient += grid_weight * grid_derivative.sum(axis=0)
            return float(value), gradient

        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            bounds=[(-20, 20)] + [(None, None)] * 16 + [(-5, 5)] * 6,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 100},
        )
        eta[target] = result.x[0]
        harmonic[target] = result.x[1:17]
        funding[target] = result.x[17:23]
        converged.append(bool(result.success))
        messages.append(str(result.message))
        seasonal_loglik -= float(result.fun)
    baseline = SeasonalBaseline(eta, harmonic, funding)
    fit.seasonal_baseline = baseline
    return {
        "fit": fit,
        "constant_profile_loglik": constant_loglik,
        "seasonal_profile_loglik": seasonal_loglik,
        "loglik_improvement": seasonal_loglik - constant_loglik,
        "improvement_per_event": (seasonal_loglik - constant_loglik) / len(codes),
        "converged": converged,
        "messages": messages,
        "quadrature_seconds": quadrature_seconds,
        "selected_events": len(selected),
    }
