"""Run the explicitly bounded ten-day pilot learning curve and GW comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import log_loss_vector
from src.validation.giacomini_white import minute_block_gw
from src.validation.learning_curve import (
    CurveResult, run_causality_sensitivity, run_pilot_learning_curve,
    run_rolling_predictions, write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    parser.add_argument("--causality-only", action="store_true")
    parser.add_argument("--book-first-feature-dir", type=Path, default=Path("data/features_book_first"))
    parser.add_argument("--book-first-aligned-dir", type=Path, default=Path("data/parquet/aligned_book_first"))
    args = parser.parse_args()
    if args.causality_only:
        stored = json.loads((args.output_dir / "pilot_learning_curve.json").read_text(encoding="utf-8"))
        default = CurveResult(**next(row for row in stored if row["budget"] == "3d"))
        sensitivity = run_causality_sensitivity(
            default_result=default,
            book_first_feature_dir=args.book_first_feature_dir,
            default_aligned_dir=args.aligned_dir,
            book_first_aligned_dir=args.book_first_aligned_dir,
            max_mle_events=args.max_mle_events,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pilot_causality_sensitivity.json").write_text(
            json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(sensitivity, indent=2))
        return
    results, predictions = run_pilot_learning_curve(
        feature_dir=args.feature_dir, aligned_dir=args.aligned_dir, max_mle_events=args.max_mle_events
    )
    write_results(results, args.output_dir / "pilot_learning_curve.json")
    chosen = run_rolling_predictions(
        feature_dir=args.feature_dir, aligned_dir=args.aligned_dir,
        max_mle_events=min(args.max_mle_events, 20_000),
    )
    gw = minute_block_gw(
        chosen["time_ms"], log_loss_vector(chosen["y"], chosen["m1"]),
        log_loss_vector(chosen["y"], chosen["m2"]),
    )
    (args.output_dir / "pilot_gw.json").write_text(json.dumps(gw, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"learning_curve": [result.__dict__ for result in results], "gw": gw}, indent=2))


if __name__ == "__main__":
    main()
