"""Full 31-day ROLLING protocol for the secondary/exploratory grid (18
specifications), with day-block Romano-Wolf FWER control over 31 real
out-of-sample days -- not the cheaper fixed-test-week 7-day-block version
in STATUS_SECONDARY_GRID.md/scripts/13.

Same cost optimization as scripts/13: fit_hawkes/fit_seasonal_baseline
(unsupervised, target-agnostic) is run exactly ONCE per forecast day, not
once per (day, spec) -- only the supervised LightGBM trees (M1 and M2,
which do depend on the target) are refit per spec, 18 times a day instead
of once.

Mirrors src.validation.learning_curve.run_rolling_predictions's exact
windowing (trailing training_days window, daily walk-forward), generalized
from the single hardcoded target_a_50_500 column to any of the 18
secondary-grid target columns in data/features_38day_secondary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa

from src.features.ewma_bank import feature_names as ewma_names
from src.features.ofi import M0_NAMES
from src.models.hawkes_mle import fit_hawkes, fit_seasonal_baseline
from src.models.tree_models import fit_tree
from src.utils.metrics import log_loss_vector
from src.validation.learning_curve import _matrix, _times, load_events, load_features
from src.validation.multiple_testing import romano_wolf_stepdown

SPECS = [
    (target, x, h)
    for target in ("a", "b")
    for x in (30, 50, 70)
    for h in (100, 500, 2000)
]


def _target_col(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day_secondary"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--training-days", type=int, default=7)
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    parser.add_argument("--no-seasonality", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/results/rolling_secondary_grid_romano_wolf.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/results/rolling_secondary_grid_checkpoints"))
    args = parser.parse_args()

    use_seasonality = not args.no_seasonality
    budget_ms = args.training_days * 86_400_000
    m0_names = list(M0_NAMES)
    m1_names = m0_names + ewma_names()
    col_names = [f"target_{t}_x{x}_h{h}" for t, x, h in SPECS]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_day_ids: list[np.ndarray] = []
    all_diffs: dict[str, list[np.ndarray]] = {c: [] for c in col_names}

    forecast_day = args.start_date + timedelta(days=args.training_days)
    warm_hawkes = None
    t_start = time.perf_counter()
    while forecast_day <= args.end_date:
        checkpoint = args.checkpoint_dir / f"rolling-secondary-{forecast_day.isoformat()}.npz"
        forecast = load_features(args.feature_dir, forecast_day, forecast_day)
        forecast_times = _times(forecast)

        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                day_ids = saved["day_ids"]
                diffs = {c: saved[c] for c in col_names}
            print(f"cached {forecast_day}: {len(day_ids):,} rows", flush=True)
            all_day_ids.append(day_ids)
            for c in col_names:
                all_diffs[c].append(diffs[c])
            forecast_day += timedelta(days=1)
            continue

        training_day = forecast_day - timedelta(days=1)
        training_start_day = forecast_day - timedelta(days=args.training_days)
        candidate = load_features(args.feature_dir, training_start_day, training_day)
        candidate_times = _times(candidate)
        boundary = int(candidate_times.max()) + 1_000
        lower = boundary - budget_ms
        keep = (candidate_times >= lower) & (candidate_times < boundary)
        training = candidate.filter(pa.array(keep))
        train_m0 = _matrix(training, m0_names)
        train_m1 = _matrix(training, m1_names)
        train_z = _matrix(training, ewma_names())
        forecast_m0 = _matrix(forecast, m0_names)
        forecast_m1 = _matrix(forecast, m1_names)
        forecast_z = _matrix(forecast, ewma_names())

        # Hawkes calibration: ONCE per day, shared across all 18 specs (unsupervised).
        t0 = time.perf_counter()
        event_times, event_codes = load_events(args.aligned_dir, lower, boundary)
        hawkes = fit_hawkes(
            event_times, event_codes, max_events=args.max_mle_events, maxiter=2_000,
            warm_start=warm_hawkes,
        )
        if use_seasonality:
            fit_seasonal_baseline(
                hawkes, event_times, event_codes,
                max_events=args.max_mle_events, maxiter=1_000,
            )
        warm_hawkes = hawkes
        train_intensity = hawkes.intensities(
            train_z, sample_time_ms=_times(training) if use_seasonality else None,
        ).astype(np.float32)
        forecast_intensity = hawkes.intensities(
            forecast_z, sample_time_ms=forecast_times if use_seasonality else None,
        ).astype(np.float32)
        m2_names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
        hawkes_time = time.perf_counter() - t0

        day_ids = np.full(len(forecast_times), forecast_day.isoformat())
        diffs_today: dict[str, np.ndarray] = {}
        t1 = time.perf_counter()
        for col in col_names:
            y_train = _target_col(training, col)
            y_forecast = _target_col(forecast, col)
            m1 = fit_tree(train_m1, y_train, m1_names, num_boost_round=120)
            m2 = fit_tree(np.column_stack((train_m0, train_intensity)), y_train, m2_names, num_boost_round=120)
            pred1 = m1.predict(forecast_m1)
            pred2 = m2.predict(np.column_stack((forecast_m0, forecast_intensity)))
            diffs_today[col] = log_loss_vector(y_forecast, pred1) - log_loss_vector(y_forecast, pred2)
        trees_time = time.perf_counter() - t1

        np.savez_compressed(checkpoint, day_ids=day_ids, **diffs_today)
        all_day_ids.append(day_ids)
        for c in col_names:
            all_diffs[c].append(diffs_today[c])

        print(f"day {forecast_day}: rows={len(forecast_times):,}  hawkes={hawkes_time:.1f}s  "
              f"trees(18 specs)={trees_time:.1f}s  elapsed={time.perf_counter()-t_start:.0f}s", flush=True)
        forecast_day += timedelta(days=1)

    day_ids = np.concatenate(all_day_ids)
    loss_matrix = np.column_stack([np.concatenate(all_diffs[c]) for c in col_names])
    adjusted_p = romano_wolf_stepdown(loss_matrix, day_ids)

    per_spec = {}
    for col, p in zip(col_names, adjusted_p):
        col_diff = loss_matrix[:, col_names.index(col)]
        per_spec[col] = {
            "mean_loss_differential_m1_minus_m2": float(col_diff.mean()),
            "romano_wolf_adjusted_p": float(p),
        }

    summary = {
        "design": {
            "protocol": "rolling, 31-day OOS, day-block Romano-Wolf",
            "training_days": args.training_days,
            "max_mle_events": args.max_mle_events,
            "seasonal_baseline": use_seasonality,
            "oos_days": int(len(np.unique(day_ids))),
            "oos_rows": int(len(day_ids)),
        },
        "specs": per_spec,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
