"""Per-N LightGBM hyperparameter tuning for M1 and M2 (the last item on
NEXT_STEPS.md's open list), testing the standing hypothesis that the
coarse 3-bucket pilot_parameters heuristic understates M1's advantage
over M2 (M1 has 50 raw EWMA features vs. M2's 10 Hawkes-intensity
features, so M1 plausibly benefits more from additional tree complexity
at large N, where the current heuristic saturates at the same config for
N=3d through N=31d despite a 10x range in training rows).

Reuses STATUS_CALIBRATION_BUDGET.md's Section 5 N-scaled Hawkes
calibration schedule (already validated as the "fairest" M2 baseline) --
this script only changes the LightGBM side, isolating the tuning
question from the calibration-budget question already closed out.

For each N: causal (time-ordered) validation-split hyperparameter search
for M1 and, separately, for M2 (tune_tree_hyperparameters /
fit_tuned_tree, src/models/tree_models.py), each refit on the full
training window, evaluated on the fixed test week.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa

from src.features.ewma_bank import feature_names as ewma_names
from src.features.ofi import M0_NAMES
from src.models.hawkes_mle import fit_hawkes, fit_seasonal_baseline
from src.models.tree_models import fit_tree, fit_tuned_tree
from src.utils.metrics import average_precision, brier_score
from src.validation.learning_curve import _matrix, _target, _times, load_events, load_features

# Same schedule as scripts/12_run_scaled_budget_sweep.py -- already
# validated in STATUS_CALIBRATION_BUDGET.md Section 5 as a fair, N-scaled
# calibration budget. Reused here unchanged so this script isolates the
# tuning question rather than re-opening the calibration-budget one.
BUDGET_SCHEDULE = {
    "2h": 100_000, "1d": 300_000, "3d": 1_000_000, "7d": 2_000_000,
    "14d": 3_000_000, "23d": 4_000_000, "31d": 5_000_000,
}
ALL_BUDGETS_MS = {
    "2h": 2 * 3_600_000, "1d": 24 * 3_600_000, "3d": 3 * 24 * 3_600_000,
    "7d": 7 * 24 * 3_600_000, "14d": 14 * 24 * 3_600_000,
    "23d": 23 * 24 * 3_600_000, "31d": 31 * 24 * 3_600_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--n-candidates", type=int, default=12)
    parser.add_argument("--budgets", nargs="+", default=list(BUDGET_SCHEDULE.keys()))
    parser.add_argument("--output", type=Path, default=Path("data/results/per_n_tuning.json"))
    args = parser.parse_args()

    table = load_features(args.feature_dir, args.start_date, args.end_date)
    times = _times(table)
    target = _target(table)
    test_start_ms = int(np.datetime64(args.test_start.isoformat(), "ms").astype(np.int64))
    test_mask = times >= test_start_ms
    pretest_times = times[times < test_start_ms]
    train_boundary = int(pretest_times.max()) + 1_000
    test_table = table.filter(pa.array(test_mask))
    y_test = target[test_mask]
    m0_names = list(M0_NAMES)
    m1_names = m0_names + ewma_names()
    test_m0 = _matrix(test_table, m0_names)
    test_m1 = _matrix(test_table, m1_names)
    test_z = _matrix(test_table, ewma_names())

    results = {}
    warm_hawkes = None
    for label in args.budgets:
        t0 = time.perf_counter()
        width_ms = ALL_BUDGETS_MS[label]
        budget = BUDGET_SCHEDULE[label]
        lower = train_boundary - width_ms
        mask = (times >= lower) & (times < train_boundary)
        train_table = table.filter(pa.array(mask))
        y_train = target[mask]
        train_time_ms = _times(train_table)
        train_m0 = _matrix(train_table, m0_names)
        train_m1 = _matrix(train_table, m1_names)
        train_z = _matrix(train_table, ewma_names())

        event_lower = int(train_time_ms.min())
        event_times, event_codes = load_events(args.aligned_dir, event_lower, train_boundary)
        hawkes = fit_hawkes(event_times, event_codes, max_events=budget, maxiter=2_000, warm_start=warm_hawkes)
        seasonal = fit_seasonal_baseline(hawkes, event_times, event_codes, max_events=budget, maxiter=1_000)
        warm_hawkes = hawkes
        train_intensity = hawkes.intensities(train_z, sample_time_ms=train_time_ms).astype(np.float32)
        test_intensity = hawkes.intensities(test_z, sample_time_ms=_times(test_table)).astype(np.float32)
        m2_names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
        train_m2 = np.column_stack((train_m0, train_intensity))
        test_m2 = np.column_stack((test_m0, test_intensity))

        # --- untuned baselines (pilot_parameters heuristic), for comparison ---
        m0_model = fit_tree(train_m0, y_train, m0_names)
        m1_untuned = fit_tree(train_m1, y_train, m1_names)
        m2_untuned = fit_tree(train_m2, y_train, m2_names)
        pred0 = m0_model.predict(test_m0)
        pred1_untuned = m1_untuned.predict(test_m1)
        pred2_untuned = m2_untuned.predict(test_m2)

        # --- tuned M1 ---
        m1_tuned, m1_overrides, m1_val, m1_val_baseline = fit_tuned_tree(
            train_m1, y_train, m1_names, train_time_ms, n_candidates=args.n_candidates,
        )
        pred1_tuned = m1_tuned.predict(test_m1)

        # --- tuned M2 ---
        m2_tuned, m2_overrides, m2_val, m2_val_baseline = fit_tuned_tree(
            train_m2, y_train, m2_names, train_time_ms, n_candidates=args.n_candidates,
        )
        pred2_tuned = m2_tuned.predict(test_m2)

        row = {
            "train_rows": int(len(y_train)), "hawkes_budget": budget,
            "m0_pr_auc": average_precision(y_test, pred0), "m0_brier": brier_score(y_test, pred0),
            "m1_untuned_pr_auc": average_precision(y_test, pred1_untuned),
            "m1_untuned_brier": brier_score(y_test, pred1_untuned),
            "m1_tuned_pr_auc": average_precision(y_test, pred1_tuned),
            "m1_tuned_brier": brier_score(y_test, pred1_tuned),
            "m1_tuning_overrides": m1_overrides,
            "m1_validation_pr_auc": m1_val, "m1_validation_baseline_pr_auc": m1_val_baseline,
            "m2_untuned_pr_auc": average_precision(y_test, pred2_untuned),
            "m2_untuned_brier": brier_score(y_test, pred2_untuned),
            "m2_tuned_pr_auc": average_precision(y_test, pred2_tuned),
            "m2_tuned_brier": brier_score(y_test, pred2_tuned),
            "m2_tuning_overrides": m2_overrides,
            "m2_validation_pr_auc": m2_val, "m2_validation_baseline_pr_auc": m2_val_baseline,
        }
        results[label] = row
        elapsed = time.perf_counter() - t0
        print(
            f"[{label}] rows={row['train_rows']:,} budget={budget:,} elapsed={elapsed:.0f}s | "
            f"M1 untuned={row['m1_untuned_pr_auc']:.6f} tuned={row['m1_tuned_pr_auc']:.6f} "
            f"(overrides={m1_overrides}) | "
            f"M2 untuned={row['m2_untuned_pr_auc']:.6f} tuned={row['m2_tuned_pr_auc']:.6f} "
            f"(overrides={m2_overrides})",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
