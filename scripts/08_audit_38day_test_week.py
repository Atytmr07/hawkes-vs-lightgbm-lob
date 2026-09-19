"""Audit the §13 fixed test week before allowing the 38-day model run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def trade_summary(path: Path) -> dict[str, float | int]:
    file = pq.ParquetFile(path, memory_map=True)
    rows = 0
    quantity = notional = 0.0
    squared_returns = 0.0
    previous_price: float | None = None
    for batch in file.iter_batches(columns=["price", "qty"], batch_size=1_000_000):
        price = batch.column("price").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        qty = batch.column("qty").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        rows += len(price)
        quantity += float(qty.sum())
        notional += float(np.dot(price, qty))
        if previous_price is not None and len(price):
            squared_returns += float(np.log(price[0] / previous_price) ** 2)
        if len(price) > 1:
            squared_returns += float(np.square(np.diff(np.log(price))).sum())
        if len(price):
            previous_price = float(price[-1])
    return {
        "trade_rows": rows,
        "base_quantity": quantity,
        "quote_notional": notional,
        "realized_volatility": float(np.sqrt(squared_returns)),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        return np.full(len(values), np.nan)
    return (values - median) / (1.4826 * mad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 2, 22))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2024, 3, 24))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument("--output", type=Path, default=Path("data/results/38day_test_week_anomaly.json"))
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()
    if not args.start_date < args.test_start <= args.end_date:
        raise ValueError("require start_date < test_start <= end_date")

    records: list[dict[str, object]] = []
    for day in days(args.start_date, args.end_date):
        trades = args.parquet_dir / "trades" / f"BTCUSDT-trades-{day}.parquet"
        book = args.parquet_dir / "bookTicker" / f"BTCUSDT-bookTicker-{day}.parquet"
        stats = args.aligned_dir / f"BTCUSDT-events-{day}.stats.json"
        if not trades.exists() or not book.exists() or not stats.exists():
            raise FileNotFoundError(f"missing Parquet/alignment inputs for {day}")
        record: dict[str, object] = {"date": day.isoformat(), **trade_summary(trades)}
        record["book_rows"] = pq.read_metadata(book).num_rows
        event_stats = json.loads(stats.read_text(encoding="utf-8"))
        record["aligned_events"] = int(event_stats["trades"]) + int(event_stats["quote_events"])
        records.append(record)

    train = [r for r in records if r["date"] < args.test_start.isoformat()]
    test = [r for r in records if r["date"] >= args.test_start.isoformat()]
    metric_names = ("trade_rows", "quote_notional", "realized_volatility", "book_rows", "aligned_events")
    for metric in metric_names:
        baseline = np.asarray([float(r[metric]) for r in train])
        z = robust_z(np.asarray([float(r[metric]) for r in test]))
        # Test values are scored against the *training* distribution, not their
        # own seven-day distribution.
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        denominator = 1.4826 * mad
        for record in test:
            value = float(record[metric])
            score = np.nan if denominator == 0 else (value - median) / denominator
            record.setdefault("robust_z", {})[metric] = float(score)

    flagged = [
        {"date": r["date"], "metric": metric, "robust_z": score}
        for r in test
        for metric, score in r["robust_z"].items()
        if np.isfinite(score) and abs(score) > args.threshold
    ]
    output = {
        "window": {"start": args.start_date.isoformat(), "test_start": args.test_start.isoformat(), "end": args.end_date.isoformat()},
        "threshold": args.threshold,
        "method": "per-day robust z-score relative to pre-test daily median/MAD; flags abs(z)>threshold",
        "metrics": list(metric_names),
        "test_week_anomalous": bool(flagged),
        "flags": flagged,
        "daily_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if flagged:
        raise SystemExit("test-week anomaly flag(s) found; do not run experiments without human decision")


if __name__ == "__main__":
    main()
