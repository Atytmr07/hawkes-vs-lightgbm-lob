"""Secondary/exploratory grid (PROJECT_SPEC.md §2.1, §5.2): 2 targets x
3 X x 3 h = 18 specifications, with day-block Romano-Wolf FWER control
(src/validation/multiple_testing.py::romano_wolf_stepdown).

This grid had never been run before this script -- Romano-Wolf had
nothing to correct. Requires data/features_38day_secondary (built by
scripts/02_build_features.py --secondary-grid, which adds the 18
target_a_x*_h*/target_b_x*_h* columns on top of the same M0/M1 feature
columns already in data/features_38day).

Cost design: Hawkes calibration (fit_hawkes) is unsupervised -- it does
not depend on the target at all -- so it is fit exactly ONCE on the
training window and its lambda(t) intensities are reused for all 18
specs' M2 trees. Only the final supervised LightGBM step varies per spec.
This is what keeps 18 specifications cheap (~minutes, not ~18x a full
per-spec Hawkes refit).

Known codebase quirk, reported not hidden: target_a's implementation
(src/features/targets.py::target_a) does not take a depletion-fraction
argument at all, so target_a_x30_h*, target_a_x50_h*, target_a_x70_h*
are byte-identical columns for a given h (the "X" grid only genuinely
varies target_b). This script still tests all 18 named columns (matching
PROJECT_SPEC.md's literal grid), but the results table notes the
duplication explicitly rather than silently treating 18 as 18 independent
tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.ewma_bank import feature_names as ewma_names
from src.features.ofi import M0_NAMES
from src.models.hawkes_mle import fit_hawkes
from src.models.tree_models import fit_tree
from src.utils.metrics import log_loss_vector
from src.validation.learning_curve import _matrix, _times, load_events
from src.validation.multiple_testing import romano_wolf_stepdown

SPECS = [
    (target, x, h)
    for target in ("a", "b")
    for x in (30, 50, 70)
    for h in (100, 500, 2000)
]


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_secondary_features(feature_dir: Path, start: date, end: date) -> pa.Table:
    paths = [feature_dir / f"BTCUSDT-features-{d}.parquet" for d in _days(start, end)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing secondary-grid feature partitions: {missing}")
    return pa.concat_tables([pq.read_table(p) for p in paths])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day_secondary"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--training-days", type=int, default=7)
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("data/results/secondary_grid_romano_wolf.json"))
    args = parser.parse_args()

    table = load_secondary_features(args.feature_dir, args.start_date, args.end_date)
    times = _times(table)
    test_start_ms = int(np.datetime64(args.test_start.isoformat(), "ms").astype(np.int64))
    test_mask = times >= test_start_ms
    pretest_times = times[times < test_start_ms]
    train_boundary = int(pretest_times.max()) + 1_000
    lower = train_boundary - args.training_days * 24 * 3_600_000
    train_mask = (times >= lower) & (times < train_boundary)

    test_table = table.filter(pa.array(test_mask))
    train_table = table.filter(pa.array(train_mask))
    day_ids = np.asarray(test_table.column("day").combine_chunks().to_pylist())

    m0_names = list(M0_NAMES)
    m1_names = m0_names + ewma_names()
    train_m0, test_m0 = _matrix(train_table, m0_names), _matrix(test_table, m0_names)
    train_m1, test_m1 = _matrix(train_table, m1_names), _matrix(test_table, m1_names)

    print(f"train_rows={len(train_table):,}  test_rows={len(test_table):,}  "
          f"test_days={len(np.unique(day_ids))}", flush=True)

    # Hawkes: fit ONCE, reused for all 18 specs' M2 trees (unsupervised, target-agnostic).
    event_lower = int(_times(train_table).min())
    event_times, event_codes = load_events(args.aligned_dir, event_lower, train_boundary)
    hawkes = fit_hawkes(event_times, event_codes, max_events=args.max_mle_events, maxiter=2_000)
    train_z = _matrix(train_table, ewma_names())
    test_z = _matrix(test_table, ewma_names())
    train_intensity = hawkes.intensities(train_z).astype(np.float32)
    test_intensity = hawkes.intensities(test_z).astype(np.float32)
    m2_names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
    print(f"hawkes fit: rho={hawkes.spectral_radius:.6f} converged={sum(hawkes.converged)}/10", flush=True)

    loss_diff_columns = []
    spec_labels = []
    per_spec = {}
    for target, x, h in SPECS:
        col = f"target_{target}_x{x}_h{h}"
        y_train = table_column_uint8(train_table, col)
        y_test = table_column_uint8(test_table, col)
        m1 = fit_tree(train_m1, y_train, m1_names)
        m2 = fit_tree(np.column_stack((train_m0, train_intensity)), y_train, m2_names)
        pred1 = m1.predict(test_m1)
        pred2 = m2.predict(np.column_stack((test_m0, test_intensity)))
        diff = log_loss_vector(y_test, pred1) - log_loss_vector(y_test, pred2)
        loss_diff_columns.append(diff)
        label = col
        spec_labels.append(label)
        per_spec[label] = {
            "prevalence": float(y_test.mean()),
            "mean_loss_differential_m1_minus_m2": float(diff.mean()),
        }
        print(f"[{label}] prevalence={y_test.mean():.6f}  mean_diff(M1-M2)={diff.mean():+.6f}", flush=True)

    loss_matrix = np.column_stack(loss_diff_columns)
    adjusted_p = romano_wolf_stepdown(loss_matrix, day_ids)
    for label, p in zip(spec_labels, adjusted_p):
        per_spec[label]["romano_wolf_adjusted_p"] = float(p)

    summary = {
        "design": {
            "training_days": args.training_days, "max_mle_events": args.max_mle_events,
            "test_week": f"{args.test_start} onward, {len(np.unique(day_ids))} days",
            "note": "target_a ignores its X argument in this codebase (src/features/targets.py); "
                    "target_a_x30/x50/x70 are identical columns per h -- reported, not hidden.",
        },
        "specs": per_spec,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


def table_column_uint8(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)


if __name__ == "__main__":
    main()
