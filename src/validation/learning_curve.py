"""Fixed-test pilot learning curve for M0, M1 and Hawkes-compressed M2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.ewma_bank import BETAS, feature_names as ewma_names, mark_feature_names, weighted_state_at_samples
from src.features.marks import trade_size_marks
from src.features.ofi import M0_NAMES
from src.models.hawkes_mle import fit_hawkes, fit_seasonal_baseline
from src.models.spectral import rolling_hourly_spectral_radius
from src.models.tree_models import fit_tree
from src.utils.metrics import average_precision, brier_score
from src.validation.causality import gamma_change_diagnostic


@dataclass
class CurveResult:
    budget: str
    train_rows: int
    test_rows: int
    prevalence: float
    m0_pr_auc: float
    m1_pr_auc: float
    m2_pr_auc: float
    m0_brier: float
    m1_brier: float
    m2_brier: float
    hawkes_spectral_radius: float
    hawkes_stationarity_scale: float
    hawkes_converged_rows: int
    hawkes_selected_events: int
    hawkes_baseline: str = "constant"
    seasonal_loglik_improvement: float | None = None
    m3_pr_auc: float | None = None
    m3_brier: float | None = None


def _days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_features(feature_dir: Path, start: date, end: date) -> pa.Table:
    paths = [feature_dir / f"BTCUSDT-features-{day}.parquet" for day in _days(start, end)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing feature partitions: {missing}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def _matrix(table: pa.Table, names: list[str]) -> np.ndarray:
    return np.column_stack([
        table.column(name).combine_chunks().to_numpy(zero_copy_only=False) for name in names
    ]).astype(np.float32)


def _target(table: pa.Table) -> np.ndarray:
    return table.column("target_a_50_500").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)


def _times(table: pa.Table) -> np.ndarray:
    return table.column("sample_time_ms").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)


def load_events(aligned_dir: Path, start_ms: int, end_ms: int) -> tuple[np.ndarray, np.ndarray]:
    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    times_parts = []
    codes_parts = []
    for day in _days(start_day, end_day):
        path = aligned_dir / f"BTCUSDT-events-{day}.parquet"
        table = pq.read_table(path, columns=["event_time_ns", "event_code"], memory_map=True)
        times = table.column("event_time_ns").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        codes = table.column("event_code").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
        keep = (times >= start_ms * 1_000_000) & (times < end_ms * 1_000_000)
        times_parts.append(times[keep])
        codes_parts.append(codes[keep])
    return np.concatenate(times_parts), np.concatenate(codes_parts)


def load_trade_events_with_volume(
    aligned_dir: Path, start_ms: int, end_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trade-only (event_code >= 8) times/codes/volume over a window.

    Filters to trades *per day*, before accumulating, so peak memory stays
    small (trades are a small fraction of all events, e.g. ~19% over the
    38-day window: 196.9M of 1,053.4M -- STATUS_38DAY.md §3) regardless of
    how many calendar days the window spans. Additive: ``load_events`` is
    unchanged; this is only used for M3's trade-size marks (§2.4, §3.2),
    which are defined solely on trade_buy/trade_sell.
    """
    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    times_parts, codes_parts, volume_parts = [], [], []
    for day in _days(start_day, end_day):
        path = aligned_dir / f"BTCUSDT-events-{day}.parquet"
        table = pq.read_table(path, columns=["event_time_ns", "event_code", "volume"], memory_map=True)
        times = table.column("event_time_ns").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        codes = table.column("event_code").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
        keep = (times >= start_ms * 1_000_000) & (times < end_ms * 1_000_000) & (codes >= 8)
        volume = table.column("volume").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        times_parts.append(times[keep])
        codes_parts.append(codes[keep])
        volume_parts.append(volume[keep])
    return np.concatenate(times_parts), np.concatenate(codes_parts), np.concatenate(volume_parts)


def _forward_fill_regime(boundary_ms: np.ndarray, radii: np.ndarray, sample_time_ms: np.ndarray) -> np.ndarray:
    """Causal step function: the most recently *completed* hourly ρ(Γ_t) as
    of each sample time. A boundary's fit uses only events strictly before
    it, so its value is safe to use starting exactly at that boundary
    (§2.4). NaN before the first completed hourly window exists (LightGBM
    handles NaN as "missing" natively -- see tree_models.fit_tree).
    """
    valid = ~np.isnan(radii)
    boundary_ms = boundary_ms[valid]
    radii = radii[valid]
    index = np.searchsorted(boundary_ms, sample_time_ms, side="right") - 1
    result = np.full(len(sample_time_ms), np.nan, dtype=np.float64)
    have = index >= 0
    result[have] = radii[index[have]]
    return result


def m3_extra_names() -> list[str]:
    return ["regime_rho_hourly"] + mark_feature_names()


def compute_m3_extra_features(
    aligned_dir: Path,
    event_times_ns: np.ndarray,
    event_codes: np.ndarray,
    train_boundary_ms: int,
    upper_bound_ms: int,
    train_sample_ms: np.ndarray,
    test_sample_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """M3's two additional blocks (§2.4): rolling hourly ρ(Γ_t) regime and
    linearized trade-size marks (§3.2).

    Memory design (this matters at N=31d: the training window alone can
    span ~850M+ raw events): the regime tracker for the *training* portion
    reuses ``event_times_ns``/``event_codes`` -- the exact same arrays M2's
    own Hawkes calibration already loaded for this budget just before this
    call -- instead of loading a second, redundant multi-hundred-million
    row copy. Only the (much smaller) fixed test week is loaded separately.
    Marks are computed from a *trade-only* substream (``load_trade_events_with_volume``),
    which is a small fraction of all events regardless of window length, so
    no full-length (one entry per raw event) weight array is ever
    allocated -- earlier draft of this function did exactly that and hit
    numpy._core._exceptions.ArrayMemoryError trying to concatenate a
    ~1.05B-row int64 array on a 16GB machine; see STATUS_M3.md.

    Design decisions (see STATUS_M3.md for full rationale):
    - The mark is defined only for trade_buy/trade_sell (codes 8, 9); every
      other event type gets weight 0.0, so only those two event types'
      columns of the marked EWMA bank are exposed as features.
    - The regime series is computed on the continuous event stream
      (matching how the Hawkes MLE itself already treats the window), not
      the per-day-reset convention the persisted M1 EWMA features use -- a
      deliberate, disclosed inconsistency, not an oversight.
    - The test week's hourly tracker is *not* warm-started from the
      training period's last Hawkes state (a consequence of loading it
      separately for memory reasons) -- it cold-starts at
      ``train_boundary_ms``. §2.3/STATUS_SPECTRAL_CONVERGENCE.md already
      showed cold-started hourly fits converge fine at ``maxiter=5000``;
      this only costs a few extra iterations on the test week's first
      hour, not a correctness or coverage gap.
    """
    # rolling_hourly_spectral_radius returns every hourly HawkesFit, and
    # fit_hawkes unconditionally attaches a _selected_state_cache to each
    # one (a states array up to max_events x 50 floats, ~100MB at the
    # default max_events=250_000) -- for N=14d's ~336 training hours that
    # is ~33GB of pure waste if the fit list is kept alive, since only
    # `.converged` is needed here. Extract that immediately and drop the
    # rest; this fixed a real MemoryError this session (see STATUS_M3.md).
    train_boundaries, train_radii, train_fits = rolling_hourly_spectral_radius(event_times_ns, event_codes)
    train_converged = sum(1 for fit in train_fits if all(fit.converged))
    n_train_fits = len(train_fits)
    del train_fits

    test_times_ns, test_codes = load_events(aligned_dir, train_boundary_ms, upper_bound_ms)
    test_boundaries, test_radii, test_fits = rolling_hourly_spectral_radius(test_times_ns, test_codes)
    test_converged = sum(1 for fit in test_fits if all(fit.converged))
    n_test_fits = len(test_fits)
    del test_times_ns, test_codes, test_fits

    boundaries = np.concatenate((train_boundaries, test_boundaries))
    radii = np.concatenate((train_radii, test_radii))
    boundary_ms = boundaries // 1_000_000
    train_regime = _forward_fill_regime(boundary_ms, radii, train_sample_ms)
    test_regime = _forward_fill_regime(boundary_ms, radii, test_sample_ms)

    trade_times_ns, trade_codes, trade_volume = load_trade_events_with_volume(
        aligned_dir, int(event_times_ns[0]) // 1_000_000, upper_bound_ms
    )
    trade_times_ms = trade_times_ns // 1_000_000
    marks = trade_size_marks(trade_times_ms, trade_volume)

    p_count = len(BETAS)
    mark_start = 8 * p_count
    train_marks = weighted_state_at_samples(
        trade_times_ns, trade_codes, marks, train_sample_ms.astype(np.int64) * 1_000_000, BETAS
    )[:, mark_start:mark_start + 2 * p_count]
    test_marks = weighted_state_at_samples(
        trade_times_ns, trade_codes, marks, test_sample_ms.astype(np.int64) * 1_000_000, BETAS
    )[:, mark_start:mark_start + 2 * p_count]

    train_extra = np.column_stack((train_regime, train_marks)).astype(np.float32)
    test_extra = np.column_stack((test_regime, test_marks)).astype(np.float32)
    valid_radii = radii[~np.isnan(radii)]
    diagnostics = {
        "hourly_windows_fitted": int(len(valid_radii)),
        "hourly_windows_fitted_train": n_train_fits,
        "hourly_windows_fitted_test": n_test_fits,
        "hourly_windows_fully_converged": train_converged + test_converged,
        "hourly_rho_min": float(valid_radii.min()) if len(valid_radii) else None,
        "hourly_rho_max": float(valid_radii.max()) if len(valid_radii) else None,
        "n_trade_events": int(len(trade_codes)),
        "n_events_train": int(len(event_codes)),
    }
    return train_extra, test_extra, diagnostics


def run_pilot_learning_curve(
    *,
    feature_dir: Path = Path("data/features"),
    aligned_dir: Path = Path("data/parquet/aligned"),
    start: date = date(2024, 1, 31),
    end: date = date(2024, 2, 9),
    test_days: int = 3,
    max_mle_events: int | dict[str, int] = 100_000,
    budget_labels: tuple[str, ...] = ("2h", "1d", "3d"),
    test_start: date | None = None,
    use_seasonality: bool = True,
    checkpoint_path: Path | None = None,
    initial_results: list[CurveResult] | None = None,
    include_m3: bool = False,
) -> tuple[list[CurveResult], dict[str, dict[str, np.ndarray]]]:
    table = load_features(feature_dir, start, end)
    times = _times(table)
    target = _target(table)
    test_start_day = test_start or end - timedelta(days=test_days - 1)
    if not start <= test_start_day <= end:
        raise ValueError("test_start must fall inside the feature date range")
    test_start_ms = int(np.datetime64(test_start_day.isoformat(), "ms").astype(np.int64))
    test_mask = times >= test_start_ms
    pretest_times = times[times < test_start_ms]
    if len(pretest_times) == 0:
        raise ValueError("no observations precede the fixed pilot test set")
    # Some official daily archives end hours before UTC midnight. Keep the
    # split causal, but anchor N to the last actually observed training sample.
    train_boundary = int(pretest_times.max()) + 1_000
    test_table = table.filter(pa.array(test_mask))
    y_test = target[test_mask]
    m0_names = list(M0_NAMES)
    m1_names = m0_names + ewma_names()
    test_m0 = _matrix(test_table, m0_names)
    test_m1 = _matrix(test_table, m1_names)
    test_z = _matrix(test_table, ewma_names())
    all_budgets = {
        "2h": 2 * 3_600_000,
        "1d": 24 * 3_600_000,
        "3d": 3 * 24 * 3_600_000,
        "7d": 7 * 24 * 3_600_000,
        "14d": 14 * 24 * 3_600_000,
        "23d": 23 * 24 * 3_600_000,
        "31d": 31 * 24 * 3_600_000,
    }
    budgets = {label: all_budgets[label] for label in budget_labels}
    results: list[CurveResult] = list(initial_results or [])
    predictions: dict[str, dict[str, np.ndarray]] = {}
    # +1 ms (not +1_000, unlike train_boundary above): this upper bound is the
    # overall table's last sample, which can sit right at the edge of the
    # final aligned day-partition; a full second could cross into a day with
    # no file on disk. +1 keeps the exclusive upper bound inside the same day
    # while still covering the very last sample's own millisecond.
    test_upper_bound_ms = int(times.max()) + 1
    warm_hawkes = None
    for label, width_ms in budgets.items():
        # STATUS_CALIBRATION_BUDGET.md: a single scalar here means the same
        # MLE calibration sample is used at every N, an unscaled fraction of
        # available events that collapses from ~8% at N=2h to ~0.01% at
        # N=31d. A dict lets each N carry its own budget (e.g. a schedule
        # that grows with N) while still warm-starting across N as before.
        label_budget = max_mle_events[label] if isinstance(max_mle_events, dict) else max_mle_events
        lower = train_boundary - width_ms
        mask = (times >= lower) & (times < train_boundary)
        train_table = table.filter(pa.array(mask))
        y_train = target[mask]
        train_m0 = _matrix(train_table, m0_names)
        train_m1 = _matrix(train_table, m1_names)
        train_z = _matrix(train_table, ewma_names())
        m0 = fit_tree(train_m0, y_train, m0_names)
        m1 = fit_tree(train_m1, y_train, m1_names)
        event_lower = int(_times(train_table).min())
        event_times, event_codes = load_events(aligned_dir, event_lower, train_boundary)
        hawkes = fit_hawkes(
            event_times, event_codes, max_events=label_budget, maxiter=2_000,
            warm_start=warm_hawkes,
        )
        seasonal_improvement = None
        if use_seasonality:
            seasonal = fit_seasonal_baseline(
                hawkes, event_times, event_codes,
                max_events=label_budget, maxiter=1_000,
            )
            seasonal_improvement = float(seasonal["loglik_improvement"])
        train_intensity = hawkes.intensities(
            train_z, sample_time_ms=_times(train_table) if use_seasonality else None,
        ).astype(np.float32)
        test_intensity = hawkes.intensities(
            test_z, sample_time_ms=_times(test_table) if use_seasonality else None,
        ).astype(np.float32)
        m2_names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
        m2 = fit_tree(np.column_stack((train_m0, train_intensity)), y_train, m2_names)
        pred0 = m0.predict(test_m0)
        pred1 = m1.predict(test_m1)
        pred2 = m2.predict(np.column_stack((test_m0, test_intensity)))

        m3_pr_auc = None
        m3_brier = None
        pred3 = None
        if include_m3:
            train_extra, test_extra, m3_diag = compute_m3_extra_features(
                aligned_dir, event_times, event_codes, train_boundary, test_upper_bound_ms,
                _times(train_table), _times(test_table),
            )
            m3_names = m1_names + m3_extra_names()
            m3 = fit_tree(np.column_stack((train_m1, train_extra)), y_train, m3_names)
            pred3 = m3.predict(np.column_stack((test_m1, test_extra)))
            m3_pr_auc = average_precision(y_test, pred3)
            m3_brier = brier_score(y_test, pred3)

        predictions[label] = {"time_ms": _times(test_table), "y": y_test, "m0": pred0, "m1": pred1, "m2": pred2}
        if pred3 is not None:
            predictions[label]["m3"] = pred3
            predictions[label]["m3_diagnostics"] = m3_diag
        results.append(CurveResult(
            budget=label, train_rows=len(y_train), test_rows=len(y_test), prevalence=float(y_test.mean()),
            m0_pr_auc=average_precision(y_test, pred0), m1_pr_auc=average_precision(y_test, pred1),
            m2_pr_auc=average_precision(y_test, pred2), m0_brier=brier_score(y_test, pred0),
            m1_brier=brier_score(y_test, pred1), m2_brier=brier_score(y_test, pred2),
            hawkes_spectral_radius=hawkes.spectral_radius,
            hawkes_stationarity_scale=hawkes.stationarity_scale,
            hawkes_converged_rows=sum(hawkes.converged), hawkes_selected_events=hawkes.selected_events,
            hawkes_baseline="seasonal_profile" if use_seasonality else "constant",
            seasonal_loglik_improvement=seasonal_improvement,
            m3_pr_auc=m3_pr_auc, m3_brier=m3_brier,
        ))
        warm_hawkes = hawkes
        if checkpoint_path is not None:
            write_results(results, checkpoint_path)
        m3_log = f", M3={results[-1].m3_pr_auc:.8f}" if results[-1].m3_pr_auc is not None else ""
        print(
            f"learning curve {label}: train={len(y_train):,}, test={len(y_test):,}, "
            f"M0={results[-1].m0_pr_auc:.8f}, M1={results[-1].m1_pr_auc:.8f}, "
            f"M2={results[-1].m2_pr_auc:.8f}{m3_log}",
            flush=True,
        )
    return results, predictions


def write_results(results: list[CurveResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n", encoding="utf-8")


def run_rolling_predictions(
    *,
    feature_dir: Path = Path("data/features"),
    aligned_dir: Path = Path("data/parquet/aligned"),
    start: date = date(2024, 1, 31),
    end: date = date(2024, 2, 9),
    budget_ms: int = 2 * 3_600_000,
    training_days: int = 1,
    max_mle_events: int = 20_000,
    use_seasonality: bool = True,
    checkpoint_dir: Path | None = None,
    include_m3: bool = False,
    m3_variant: str = "both",
) -> dict[str, np.ndarray]:
    """Genuine daily rolling forecasts using a trailing causal training window.

    ``include_m3`` (naive path -- see STATUS_CALIBRATION_BUDGET.md's compute
    sizing): recomputes ``rolling_hourly_spectral_radius`` over the full
    trailing training window from scratch on every forecast day, with no
    cross-day warm-start carry-over (each day's regime tracker cold-starts
    independently, same as ``compute_m3_extra_features`` already does for
    the static learning curve's training/test split). This is the dominant
    cost of a rolling M3 run; an incremental version that reuses the
    previous day's hourly state was explicitly not built here.

    ``m3_variant`` (only meaningful with ``include_m3=True``) selects which
    of M3's two extra blocks feed the "m3" column: ``"both"`` (default,
    matches the original M3 definition), ``"regime_only"`` (just the
    hourly rho(Gamma_t) step feature), or ``"marks_only"`` (just the
    trade-size marks). Added for the STATUS_M3.md Section 11 ablation,
    which found the regime block -- not marks -- drives essentially all of
    the calibration improvement in the "both" case on the static learning
    curve; this lets that finding be checked for formal rolling-GW
    significance without recomputing the same expensive regime tracker in
    a separate, hand-written script.
    """
    if m3_variant not in ("both", "regime_only", "marks_only"):
        raise ValueError("m3_variant must be 'both', 'regime_only', or 'marks_only'")
    if training_days < 1:
        raise ValueError("training_days must be at least one")
    all_times: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_m1: list[np.ndarray] = []
    all_m2: list[np.ndarray] = []
    all_m3: list[np.ndarray] = []
    forecast_day = start + timedelta(days=training_days)
    warm_hawkes = None
    m0_names = list(M0_NAMES)
    m1_names = m0_names + ewma_names()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    while forecast_day <= end:
        checkpoint = None if checkpoint_dir is None else (
            checkpoint_dir / f"rolling-{forecast_day.isoformat()}.npz"
        )
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint) as saved:
                all_times.append(saved["time_ms"])
                all_targets.append(saved["y"])
                all_m1.append(saved["m1"])
                all_m2.append(saved["m2"])
                if include_m3:
                    if "m3" not in saved:
                        raise ValueError(
                            f"{checkpoint} has no 'm3' array (checkpointed "
                            "without include_m3=True) -- use a separate "
                            "checkpoint_dir to recompute this window with M3"
                        )
                    all_m3.append(saved["m3"])
            print(f"rolling cached {forecast_day}: {len(all_targets[-1]):,} rows", flush=True)
            forecast_day += timedelta(days=1)
            continue
        forecast = load_features(feature_dir, forecast_day, forecast_day)
        forecast_times = _times(forecast)
        training_day = forecast_day - timedelta(days=1)
        training_start_day = forecast_day - timedelta(days=training_days)
        candidate = load_features(feature_dir, training_start_day, training_day)
        candidate_times = _times(candidate)
        boundary = int(candidate_times.max()) + 1_000
        lower = boundary - budget_ms
        keep = (candidate_times >= lower) & (candidate_times < boundary)
        training = candidate.filter(pa.array(keep))
        y_train = _target(training)
        y_forecast = _target(forecast)
        train_m0 = _matrix(training, m0_names)
        train_m1 = _matrix(training, m1_names)
        train_z = _matrix(training, ewma_names())
        forecast_m0 = _matrix(forecast, m0_names)
        forecast_m1 = _matrix(forecast, m1_names)
        forecast_z = _matrix(forecast, ewma_names())
        m1 = fit_tree(train_m1, y_train, m1_names, num_boost_round=120)
        event_times, event_codes = load_events(aligned_dir, lower, boundary)
        hawkes = fit_hawkes(
            event_times, event_codes, max_events=max_mle_events, maxiter=2_000,
            warm_start=warm_hawkes,
        )
        if use_seasonality:
            fit_seasonal_baseline(
                hawkes, event_times, event_codes,
                max_events=max_mle_events, maxiter=1_000,
            )
        warm_hawkes = hawkes
        train_intensity = hawkes.intensities(
            train_z, sample_time_ms=_times(training) if use_seasonality else None,
        ).astype(np.float32)
        forecast_intensity = hawkes.intensities(
            forecast_z, sample_time_ms=forecast_times if use_seasonality else None,
        ).astype(np.float32)
        m2_names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
        m2 = fit_tree(np.column_stack((train_m0, train_intensity)), y_train, m2_names, num_boost_round=120)
        pred1 = m1.predict(forecast_m1)
        pred2 = m2.predict(np.column_stack((forecast_m0, forecast_intensity)))
        pred3 = None
        if include_m3:
            forecast_upper_bound_ms = int(forecast_times.max()) + 1
            train_extra, test_extra, m3_diag = compute_m3_extra_features(
                aligned_dir, event_times, event_codes, boundary, forecast_upper_bound_ms,
                _times(training), forecast_times,
            )
            if m3_variant == "regime_only":
                train_extra, test_extra = train_extra[:, :1], test_extra[:, :1]
                extra_names = ["regime_rho_hourly"]
            elif m3_variant == "marks_only":
                train_extra, test_extra = train_extra[:, 1:], test_extra[:, 1:]
                extra_names = mark_feature_names()
            else:
                extra_names = m3_extra_names()
            m3_names = m1_names + extra_names
            m3 = fit_tree(np.column_stack((train_m1, train_extra)), y_train, m3_names, num_boost_round=120)
            pred3 = m3.predict(np.column_stack((forecast_m1, test_extra)))
        if checkpoint is not None:
            save_kwargs = dict(time_ms=forecast_times, y=y_forecast, m1=pred1, m2=pred2)
            if pred3 is not None:
                save_kwargs["m3"] = pred3
            np.savez_compressed(checkpoint, **save_kwargs)
        all_times.append(forecast_times)
        all_targets.append(y_forecast)
        all_m1.append(pred1)
        all_m2.append(pred2)
        m3_log = ""
        if pred3 is not None:
            all_m3.append(pred3)
            m3_log = (
                f", m3_pr_auc={average_precision(y_forecast, pred3):.6f}"
                f", hourly_rho_max={m3_diag['hourly_rho_max']:.4f}"
                f", hourly_converged={m3_diag['hourly_windows_fully_converged']}/{m3_diag['hourly_windows_fitted']}"
            )
        print(
            f"rolling forecast {forecast_day}: train={len(y_train):,}, test={len(y_forecast):,}"
            f", m1_pr_auc={average_precision(y_forecast, pred1):.6f}"
            f", m2_pr_auc={average_precision(y_forecast, pred2):.6f}{m3_log}",
            flush=True,
        )
        forecast_day += timedelta(days=1)
    result = {
        "time_ms": np.concatenate(all_times), "y": np.concatenate(all_targets),
        "m1": np.concatenate(all_m1), "m2": np.concatenate(all_m2),
    }
    if include_m3:
        result["m3"] = np.concatenate(all_m3)
    return result


def run_causality_sensitivity(
    *,
    default_result: CurveResult,
    book_first_feature_dir: Path,
    default_aligned_dir: Path,
    book_first_aligned_dir: Path,
    max_mle_events: int = 100_000,
    feature_dir: Path = Path("data/features"),
    start: date = date(2024, 1, 31),
    end: date = date(2024, 2, 9),
    test_start: date | None = None,
    budget_label: str = "3d",
    use_seasonality: bool = True,
) -> dict[str, object]:
    inverted_results, _ = run_pilot_learning_curve(
        feature_dir=book_first_feature_dir,
        aligned_dir=book_first_aligned_dir,
        start=start, end=end, test_start=test_start,
        max_mle_events=max_mle_events, budget_labels=(budget_label,),
        use_seasonality=use_seasonality,
    )
    inverted = inverted_results[0]
    table = load_features(feature_dir, start, end)
    times = _times(table)
    split_day = test_start or end - timedelta(days=2)
    test_start_ms = int(np.datetime64(split_day.isoformat(), "ms").astype(np.int64))
    boundary = int(times[times < test_start_ms].max()) + 1_000
    budget_days = {"2h": 2 / 24, "1d": 1, "3d": 3, "7d": 7, "14d": 14, "23d": 23, "31d": 31}
    if budget_label not in budget_days:
        raise ValueError(f"unknown budget label: {budget_label}")
    lower = boundary - int(budget_days[budget_label] * 86_400_000)
    observed_lower = int(times[(times >= lower) & (times < boundary)].min())
    default_times, default_codes = load_events(default_aligned_dir, observed_lower, boundary)
    inverted_times, inverted_codes = load_events(book_first_aligned_dir, observed_lower, boundary)
    default_fit = fit_hawkes(default_times, default_codes, max_events=max_mle_events, maxiter=2_000)
    inverted_fit = fit_hawkes(inverted_times, inverted_codes, max_events=max_mle_events, maxiter=2_000)
    difference = inverted_fit.gamma - default_fit.gamma
    relative = float(np.linalg.norm(difference) / max(np.linalg.norm(default_fit.gamma), 1e-12))
    return {
        "budget": budget_label,
        "default_pr_auc": {"m0": default_result.m0_pr_auc, "m1": default_result.m1_pr_auc, "m2": default_result.m2_pr_auc},
        "book_first_pr_auc": {"m0": inverted.m0_pr_auc, "m1": inverted.m1_pr_auc, "m2": inverted.m2_pr_auc},
        "m2_pr_auc_delta_book_minus_default": inverted.m2_pr_auc - default_result.m2_pr_auc,
        "gamma_relative_frobenius_change": relative,
        "gamma_max_absolute_change": float(np.max(np.abs(difference))),
        "default_spectral_radius": default_fit.spectral_radius,
        "book_first_spectral_radius": inverted_fit.spectral_radius,
        "default_converged_rows": int(sum(default_fit.converged)),
        "book_first_converged_rows": int(sum(inverted_fit.converged)),
        "gamma_diagnostic": gamma_change_diagnostic(default_fit.gamma, inverted_fit.gamma),
    }
