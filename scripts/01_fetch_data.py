"""Download, convert and align a requested date range."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.aligner import align_day
from src.data_loader.downloader import run


def days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 31))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--tie-rule", choices=("trades_first", "book_first"), default="trades_first")
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned"))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    args = parser.parse_args()
    run(start=args.start_date, end=args.end_date, symbol=args.symbol,
        parquet_dir=args.parquet_dir, download_workers=args.download_workers)
    for day in days(args.start_date, args.end_date):
        path, stats = align_day(day=day, tie_rule=args.tie_rule, aligned_dir=args.aligned_dir,
                                 parquet_dir=args.parquet_dir, symbol=args.symbol)
        print(f"{day}: {path.stat().st_size:,} bytes, {sum(stats.event_counts.values()):,} events")


if __name__ == "__main__":
    main()
