"""N-scaled calibration budget re-sweep of the fixed-test learning curve
(STATUS_CALIBRATION_BUDGET.md §4's recommendation). The official §14.1 /
pilot_38day_learning_curve.json run used a fixed 100,000-event Hawkes MLE
calibration budget at every N (2h..31d) -- a share of available events
that collapses from ~8% at N=2h to ~0.011% at N=31d. This script re-runs
the same fixed-test-week protocol with a budget that grows with N, sized
via one standalone timing probe at the most expensive point (N=31d,
5,000,000 events: 1185.3s / ~20 minutes) before committing to the full
sweep.

Does not modify data/results/pilot_38day_learning_curve.json or
PROJECT_SPEC.md's §14 table -- writes to a separate file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.learning_curve import run_pilot_learning_curve

# Grows with N so the *share* of available events used stays roughly flat
# instead of collapsing ~750x across the grid (STATUS_CALIBRATION_BUDGET.md
# §2). 2h is left at the original 100k (already ~8% of available events,
# not starved); each larger N gets a budget scaled to keep pace with how
# fast available events grow, capped well below the raw event count.
BUDGET_SCHEDULE = {
    "2h": 100_000,
    "1d": 300_000,
    "3d": 1_000_000,
    "7d": 2_000_000,
    "14d": 3_000_000,
    "23d": 4_000_000,
    "31d": 5_000_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--output", type=Path, default=Path("data/results/scaled_budget_learning_curve.json"))
    args = parser.parse_args()

    results, _ = run_pilot_learning_curve(
        feature_dir=args.feature_dir,
        aligned_dir=args.aligned_dir,
        start=args.start_date,
        end=args.end_date,
        test_start=args.test_start,
        max_mle_events=BUDGET_SCHEDULE,
        budget_labels=tuple(BUDGET_SCHEDULE.keys()),
        use_seasonality=True,
    )

    summary = {
        "design": {
            "window": f"{args.start_date} to {args.end_date}, test week from {args.test_start}",
            "budget_schedule": BUDGET_SCHEDULE,
            "official_fixed_budget_for_comparison": 100_000,
        },
        "results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
