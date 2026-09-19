"""Compare constant and §3.3 seasonal Hawkes baselines on one real day."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.aligner import EVENT_TYPES
from src.models.hawkes_mle import fit_hawkes, fit_seasonal_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 1, 31))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned"))
    parser.add_argument("--output", type=Path, default=Path("data/results/seasonality_comparison_2024-01-31.json"))
    parser.add_argument("--max-events", type=int, default=100_000)
    args = parser.parse_args()
    path = args.aligned_dir / f"BTCUSDT-events-{args.date}.parquet"
    table = pq.read_table(path, columns=["event_time_ns", "event_code"], memory_map=True)
    times = table.column("event_time_ns").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    codes = table.column("event_code").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    constant = fit_hawkes(times, codes, max_events=args.max_events, maxiter=2_000)
    comparison = fit_seasonal_baseline(
        constant, times, codes, max_events=args.max_events,
        quadrature_seconds=60, maxiter=2_000,
    )
    start = datetime(args.date.year, args.date.month, args.date.day, tzinfo=timezone.utc)
    grid_ms = np.arange(
        int(start.timestamp() * 1000),
        int((start + timedelta(days=1)).timestamp() * 1000),
        60_000,
        dtype=np.int64,
    )
    seasonal_rates = constant.seasonal_baseline.rates(grid_ms)
    per_type = {}
    for event_type, index in zip(EVENT_TYPES, range(10), strict=True):
        per_type[event_type] = {
            "constant_mu": float(constant.mu[index]),
            "seasonal_min": float(seasonal_rates[:, index].min()),
            "seasonal_median": float(np.median(seasonal_rates[:, index])),
            "seasonal_max": float(seasonal_rates[:, index].max()),
            "harmonic_l2": float(np.linalg.norm(constant.seasonal_baseline.harmonic[index])),
            "funding_l2": float(np.linalg.norm(constant.seasonal_baseline.funding[index])),
        }
    output = {
        "date": args.date.isoformat(), "event_rows": len(times),
        "selected_events": comparison["selected_events"],
        "constant_kernel_converged_rows": int(sum(constant.converged)),
        "seasonal_baseline_converged_rows": int(sum(comparison["converged"])),
        "constant_profile_loglik": comparison["constant_profile_loglik"],
        "seasonal_profile_loglik": comparison["seasonal_profile_loglik"],
        "loglik_improvement": comparison["loglik_improvement"],
        "improvement_per_event": comparison["improvement_per_event"],
        "spectral_radius": constant.spectral_radius,
        "stationarity_scale": constant.stationarity_scale,
        "quadrature_seconds": comparison["quadrature_seconds"],
        "per_event_type": per_type,
        "seasonal_messages": comparison["messages"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
