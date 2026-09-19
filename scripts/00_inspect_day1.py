"""Inspect real Binance daily archives without changing the research spec."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def _csv_path(feed: str, day: str, raw_dir: Path) -> Path:
    csv_path = raw_dir / f"BTCUSDT-{feed}-{day}.csv"
    if csv_path.exists():
        return csv_path
    archive_path = csv_path.with_suffix(".zip")
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(csv_path.name, raw_dir)
    return csv_path


def inspect(day: str, raw_dir: Path) -> dict[str, object]:
    trade_path = _csv_path("trades", day, raw_dir)
    book_path = _csv_path("bookTicker", day, raw_dir)
    trade_columns = pd.read_csv(trade_path, nrows=0).columns.tolist()
    book_columns = pd.read_csv(book_path, nrows=0).columns.tolist()
    trades = pd.read_csv(trade_path, usecols=["time"])
    book = pd.read_csv(book_path, usecols=["transaction_time", "update_id"])
    times = book["transaction_time"].to_numpy(np.int64)
    ids = book["update_id"].to_numpy(np.int64)
    raw_time_back = int((np.diff(times) < 0).sum())
    raw_composite_back = int(((np.diff(times) < 0) | ((np.diff(times) == 0) & (np.diff(ids) < 0))).sum())
    order = np.lexsort((ids, times))
    sorted_times = times[order]
    diffs = np.diff(sorted_times)
    sub_10ms = float((diffs < 10).mean() * 100.0)
    quote_unique = np.unique(sorted_times)
    overlap = np.intersect1d(np.unique(trades["time"].to_numpy(np.int64)), quote_unique).size
    collision_rate = float(overlap / len(quote_unique) * 100.0)
    normalized = float((collision_rate / 100.0) / (sub_10ms / 100.0)) if sub_10ms else float("nan")
    return {
        "day": day,
        "trade_columns": trade_columns,
        "bookTicker_columns": book_columns,
        "trade_rows": len(trades),
        "bookTicker_rows": len(book),
        "raw_time_decreases": raw_time_back,
        "raw_time_decrease_pct": raw_time_back / max(1, len(book) - 1) * 100.0,
        "raw_composite_violations": raw_composite_back,
        "sub_10ms_pct": sub_10ms,
        "exact_ms_collisions": int(overlap),
        "collision_pct": collision_rate,
        "normalized_collision_x": normalized,
    }


def raw_ordering(day: str, raw_dir: Path, parquet_dir: Path) -> dict[str, object]:
    parquet_path = parquet_dir / "bookTicker" / f"BTCUSDT-bookTicker-{day}.parquet"
    csv_path = raw_dir / f"BTCUSDT-bookTicker-{day}.csv"
    if parquet_path.exists():
        batches = pq.ParquetFile(parquet_path).iter_batches(
            batch_size=2_000_000, columns=["transaction_time", "update_id"]
        )
    elif csv_path.exists():
        batches = pacsv.open_csv(
            csv_path,
            read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
            convert_options=pacsv.ConvertOptions(include_columns=["transaction_time", "update_id"]),
        )
    else:
        raise FileNotFoundError(f"no raw-order source for {day}")
    rows = time_back = same_ms_id_back = 0
    previous_time = previous_id = None
    for batch in batches:
        times = batch.column("transaction_time").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ids = batch.column("update_id").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if previous_time is not None and len(times):
            time_back += int(times[0] < previous_time)
            same_ms_id_back += int(times[0] == previous_time and ids[0] < previous_id)
        if len(times) > 1:
            differences = np.diff(times)
            time_back += int((differences < 0).sum())
            same_ms_id_back += int(((differences == 0) & (np.diff(ids) < 0)).sum())
        if len(times):
            previous_time, previous_id = int(times[-1]), int(ids[-1])
        rows += len(times)
    return {
        "day": day, "rows": rows, "raw_time_decreases": time_back,
        "raw_time_decrease_pct": time_back / max(1, rows - 1) * 100.0,
        "same_ms_update_id_decreases": same_ms_id_back,
        "composite_violations": time_back + same_ms_id_back,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-03-30")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--ordering-only", action="store_true")
    parser.add_argument("--dates", nargs="+")
    args = parser.parse_args()
    if args.ordering_only:
        days = args.dates or ["2024-01-31", "2024-02-28", "2024-03-30"]
        print(json.dumps([raw_ordering(day, args.raw_dir, args.parquet_dir) for day in days], indent=2))
    else:
        print(json.dumps(inspect(args.date, args.raw_dir), indent=2))


if __name__ == "__main__":
    main()
