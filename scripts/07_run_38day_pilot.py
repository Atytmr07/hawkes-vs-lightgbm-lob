"""Run the §13 fixed-test learning curve, rolling GW, and tie-break sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import log_loss_vector
from src.validation.giacomini_white import minute_block_gw
from src.validation.learning_curve import (
    CurveResult,
    run_causality_sensitivity,
    run_pilot_learning_curve,
    run_rolling_predictions,
    write_results,
)


DEFAULT_BUDGETS = ("2h", "1d", "3d", "7d", "14d", "23d", "31d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--book-first-feature-dir", type=Path, default=Path("data/features_38day_book_first"))
    parser.add_argument("--book-first-aligned-dir", type=Path, default=Path("data/parquet/aligned_38day_book_first"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    parser.add_argument("--rolling-training-days", type=int, default=7)
    parser.add_argument("--rolling-max-mle-events", type=int, default=100_000)
    parser.add_argument("--no-seasonality", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    use_seasonality = not args.no_seasonality
    checkpoint = args.output_dir / "pilot_38day_learning_curve.json"
    existing: list[CurveResult] = []
    if args.resume and checkpoint.exists():
        existing = [
            CurveResult(**row)
            for row in json.loads(checkpoint.read_text(encoding="utf-8"))
        ]
    completed = {result.budget for result in existing}
    remaining = tuple(label for label in DEFAULT_BUDGETS if label not in completed)
    if remaining:
        results, _ = run_pilot_learning_curve(
            feature_dir=args.feature_dir,
            aligned_dir=args.aligned_dir,
            start=args.start_date,
            end=args.end_date,
            test_start=args.test_start,
            max_mle_events=args.max_mle_events,
            budget_labels=remaining,
            use_seasonality=use_seasonality,
            checkpoint_path=checkpoint,
            initial_results=existing,
        )
    else:
        results = existing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(results, args.output_dir / "pilot_38day_learning_curve.json")

    sensitivity_path = args.output_dir / "pilot_38day_causality_sensitivity.json"
    if args.resume and sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    else:
        three_day = next(result for result in results if result.budget == "3d")
        sensitivity = run_causality_sensitivity(
            default_result=three_day,
            feature_dir=args.feature_dir,
            book_first_feature_dir=args.book_first_feature_dir,
            default_aligned_dir=args.aligned_dir,
            book_first_aligned_dir=args.book_first_aligned_dir,
            start=args.start_date,
            end=args.end_date,
            test_start=args.test_start,
            budget_label="3d",
            max_mle_events=args.max_mle_events,
            use_seasonality=use_seasonality,
        )
        sensitivity_path.write_text(
            json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8"
        )

    rolling = run_rolling_predictions(
        feature_dir=args.feature_dir,
        aligned_dir=args.aligned_dir,
        start=args.start_date,
        end=args.end_date,
        budget_ms=args.rolling_training_days * 86_400_000,
        training_days=args.rolling_training_days,
        max_mle_events=args.rolling_max_mle_events,
        use_seasonality=use_seasonality,
        checkpoint_dir=args.output_dir / "rolling_38day_checkpoints",
    )
    gw = minute_block_gw(
        rolling["time_ms"],
        log_loss_vector(rolling["y"], rolling["m1"]),
        log_loss_vector(rolling["y"], rolling["m2"]),
    )
    gw.update({
        "label": "38-day clean-window rolling GW",
        "rolling_training_days": args.rolling_training_days,
        "rolling_max_mle_events": args.rolling_max_mle_events,
        "seasonal_baseline": use_seasonality,
        "oos_rows": int(len(rolling["y"])),
    })
    (args.output_dir / "pilot_38day_gw.json").write_text(
        json.dumps(gw, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "window": {
            "start": args.start_date.isoformat(),
            "test_start": args.test_start.isoformat(),
            "end": args.end_date.isoformat(),
        },
        "seasonal_baseline": use_seasonality,
        "learning_curve": [result.__dict__ for result in results],
        "causality_sensitivity": sensitivity,
        "gw": gw,
    }
    (args.output_dir / "pilot_38day_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
