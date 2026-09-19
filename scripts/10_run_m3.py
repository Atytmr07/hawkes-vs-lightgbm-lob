"""Evaluate M3 (M1 + rolling hourly rho(Gamma_t) regime + trade-size marks)
against M0/M1/M2 on the fixed test week, for two budgets (N=3d, N=31d).

Does not modify data/results/pilot_38day_learning_curve.json or any other
existing result. All four models (M0/M1/M2/M3) are computed together in
this single run so the M2-vs-M3 comparison is internally consistent even
though the M2 Hawkes fit is not bit-reproducible run-to-run (see
STATUS_M3.md for the observed magnitude of that pre-existing variation).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.learning_curve import run_pilot_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    parser.add_argument("--budgets", nargs="+", default=["3d", "31d"])
    parser.add_argument("--output", type=Path, default=Path("data/results/m3_comparison.json"))
    args = parser.parse_args()

    results, predictions = run_pilot_learning_curve(
        feature_dir=args.feature_dir,
        aligned_dir=args.aligned_dir,
        start=args.start_date,
        end=args.end_date,
        test_start=args.test_start,
        max_mle_events=args.max_mle_events,
        budget_labels=tuple(args.budgets),
        use_seasonality=True,
        include_m3=True,
    )

    summary = {
        "design": {
            "window": f"{args.start_date} to {args.end_date}, test week from {args.test_start}",
            "budgets": args.budgets,
            "max_mle_events": args.max_mle_events,
        },
        "results": [
            {
                "budget": r.budget,
                "train_rows": r.train_rows,
                "test_rows": r.test_rows,
                "m0_pr_auc": r.m0_pr_auc, "m1_pr_auc": r.m1_pr_auc,
                "m2_pr_auc": r.m2_pr_auc, "m3_pr_auc": r.m3_pr_auc,
                "m0_brier": r.m0_brier, "m1_brier": r.m1_brier,
                "m2_brier": r.m2_brier, "m3_brier": r.m3_brier,
                "delta_pr_auc_m3_minus_m1": (r.m3_pr_auc - r.m1_pr_auc) if r.m3_pr_auc is not None else None,
                "hawkes_spectral_radius": r.hawkes_spectral_radius,
                "hawkes_converged_rows": r.hawkes_converged_rows,
                "m3_diagnostics": predictions[r.budget].get("m3_diagnostics"),
            }
            for r in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
