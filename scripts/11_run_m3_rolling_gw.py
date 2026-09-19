"""Formal rolling Giacomini-White test for M3 vs M1 (STATUS_M3.md §5's
Primary Endpoint interpretation), on the same trailing-7-day rolling
protocol and 2024-02-29..2024-03-30 OOS window as the official M1-vs-M2
result in STATUS_38DAY.md §6/§14.2.

Naive implementation (STATUS_CALIBRATION_BUDGET.md's compute sizing):
run_rolling_predictions(include_m3=True) recomputes the hourly regime
tracker from scratch every forecast day. Does not modify
data/results/pilot_38day_gw.json or any other existing result. Reports
both GW(M1,M2) -- reproduced fresh, as a sanity check against the official
value -- and GW(M1,M3), the new confirmatory test.

--m3-variant selects which of M3's two extra blocks feed the "m3" column
(STATUS_M3.md §11's ablation): "both" (default, the original --output
path, unchanged), "regime_only", or "marks_only". A non-default variant
writes to its own output file and checkpoint directory so it never
collides with the official "both" result.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import log_loss_vector
from src.validation.giacomini_white import minute_block_gw
from src.validation.learning_curve import run_rolling_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--rolling-training-days", type=int, default=7)
    parser.add_argument("--rolling-max-mle-events", type=int, default=100_000)
    parser.add_argument("--no-seasonality", action="store_true")
    parser.add_argument("--m3-variant", choices=["both", "regime_only", "marks_only"], default="both")
    args = parser.parse_args()

    use_seasonality = not args.no_seasonality
    suffix = "" if args.m3_variant == "both" else f"_{args.m3_variant}"
    checkpoint_dir = args.output_dir / f"rolling_38day_m3_checkpoints{suffix}"

    rolling = run_rolling_predictions(
        feature_dir=args.feature_dir,
        aligned_dir=args.aligned_dir,
        start=args.start_date,
        end=args.end_date,
        budget_ms=args.rolling_training_days * 86_400_000,
        training_days=args.rolling_training_days,
        max_mle_events=args.rolling_max_mle_events,
        use_seasonality=use_seasonality,
        checkpoint_dir=checkpoint_dir,
        include_m3=True,
        m3_variant=args.m3_variant,
    )

    gw_m1_m2 = minute_block_gw(
        rolling["time_ms"],
        log_loss_vector(rolling["y"], rolling["m1"]),
        log_loss_vector(rolling["y"], rolling["m2"]),
    )
    gw_m1_m2.update({"label": "38-day M3-session rolling GW, M1 vs M2 (sanity check)"})

    gw_m1_m3 = minute_block_gw(
        rolling["time_ms"],
        log_loss_vector(rolling["y"], rolling["m1"]),
        log_loss_vector(rolling["y"], rolling["m3"]),
    )
    gw_m1_m3.update({
        "label": f"38-day rolling GW, M1 vs M3[{args.m3_variant}] "
                 "(STATUS_M3.md §5 Primary Endpoint interpretation / §11 ablation follow-up)",
    })

    summary = {
        "window": {"start": args.start_date.isoformat(), "end": args.end_date.isoformat()},
        "rolling_training_days": args.rolling_training_days,
        "rolling_max_mle_events": args.rolling_max_mle_events,
        "seasonal_baseline": use_seasonality,
        "m3_variant": args.m3_variant,
        "oos_rows": int(len(rolling["y"])),
        "gw_m1_vs_m2": gw_m1_m2,
        "gw_m1_vs_m3": gw_m1_m3,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"m3_rolling_gw{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
