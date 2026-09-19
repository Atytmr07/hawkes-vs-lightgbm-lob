"""Build causal 1-second pilot matrices from all intervening event-level data."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.features.ewma_bank import feature_names as ewma_names
from src.features.ewma_bank import state_at_samples
from src.features.ofi import M0_NAMES, l1_features_at_samples
from src.features.seasonality import seasonality_design
from src.features.targets import target_a, target_b

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


def _numpy(table: pa.Table, name: str, dtype) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def build_day(
    day: date,
    *,
    parquet_dir: Path,
    aligned_dir: Path,
    feature_dir: Path,
    sample_ms: int = 1000,
    warmup_ms: int = 3_600_000,
    secondary_grid: bool = False,
    symbol: str = "BTCUSDT",
) -> Path:
    book_path = parquet_dir / "bookTicker" / f"{symbol}-bookTicker-{day}.parquet"
    aligned_path = aligned_dir / f"{symbol}-events-{day}.parquet"
    book = pq.read_table(book_path, columns=[
        "update_id", "transaction_time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"
    ], memory_map=True)
    raw_time = _numpy(book, "transaction_time", np.int64)
    raw_update_id = _numpy(book, "update_id", np.int64)
    raw_diff = np.diff(raw_time)
    if np.any(raw_diff < 0) or np.any((raw_diff == 0) & (np.diff(raw_update_id) < 0)):
        indices = pc.sort_indices(
            book,
            sort_keys=[("transaction_time", "ascending"), ("update_id", "ascending")],
        )
        book = pc.take(book, indices)
    events = pq.read_table(aligned_path, columns=["event_time_ns", "event_code"], memory_map=True)
    qt = _numpy(book, "transaction_time", np.int64)
    bp = _numpy(book, "best_bid_price", np.float64)
    bq = _numpy(book, "best_bid_qty", np.float64)
    ap = _numpy(book, "best_ask_price", np.float64)
    aq = _numpy(book, "best_ask_qty", np.float64)
    start = ((int(qt[0]) + sample_ms - 1) // sample_ms) * sample_ms
    end = ((int(qt[-1]) - 550) // sample_ms) * sample_ms
    sample_times = np.arange(start, end + 1, sample_ms, dtype=np.int64)
    m0, valid = l1_features_at_samples(qt, bp, bq, ap, aq, sample_times, samples_per_hour=3_600_000 // sample_ms)
    samples_per_hour = 3_600_000 // sample_ms
    end_indices = np.searchsorted(qt, sample_times + 550, side="right") - 1
    end_indices = np.maximum(end_indices, 0)
    valid &= sample_times + 550 - qt[end_indices] <= 5_000
    valid &= sample_times >= int(qt[0]) + warmup_ms
    event_times = _numpy(events, "event_time_ns", np.int64)
    event_codes = _numpy(events, "event_code", np.uint8)
    z = state_at_samples(event_times, event_codes, sample_times * 1_000_000)
    seasonal, seasonal_names = seasonality_design(sample_times)
    labels = target_a(qt, bp, ap, sample_times, 50, 500)
    columns: dict[str, pa.Array] = {
        "sample_time_ms": pa.array(sample_times[valid]),
        "day": pa.array(np.full(valid.sum(), day.isoformat())),
        "target_a_50_500": pa.array(labels[valid]),
    }
    for index, name in enumerate(M0_NAMES):
        columns[name] = pa.array(m0[valid, index].astype(np.float32))
    for index, name in enumerate(ewma_names()):
        columns[name] = pa.array(z[valid, index].astype(np.float32))
    for index, name in enumerate(seasonal_names):
        columns[name] = pa.array(seasonal[valid, index].astype(np.float32))
    if secondary_grid:
        for fraction in (0.3, 0.5, 0.7):
            for horizon in (100, 500, 2000):
                suffix = f"x{int(fraction * 100)}_h{horizon}"
                columns[f"target_a_{suffix}"] = pa.array(target_a(qt, bp, ap, sample_times, 50, horizon)[valid])
                columns[f"target_b_{suffix}"] = pa.array(target_b(qt, bp, bq, ap, aq, sample_times, fraction, 50, horizon)[valid])
    table = pa.table(columns)
    feature_dir.mkdir(parents=True, exist_ok=True)
    target = feature_dir / f"{symbol}-features-{day}.parquet"
    temporary = target.with_suffix(".parquet.part")
    pq.write_table(table, temporary, compression="zstd", compression_level=3, row_group_size=100_000)
    temporary.replace(target)
    print(f"features {day}: {len(table):,} samples, prevalence={labels[valid].mean():.6f}, {target.stat().st_size:,} bytes")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 31))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--sample-ms", type=int, default=1000)
    parser.add_argument("--secondary-grid", action="store_true")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()
    current = args.start_date
    while current <= args.end_date:
        build_day(current, parquet_dir=args.parquet_dir, aligned_dir=args.aligned_dir, feature_dir=args.feature_dir, sample_ms=args.sample_ms, secondary_grid=args.secondary_grid, symbol=args.symbol)
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
