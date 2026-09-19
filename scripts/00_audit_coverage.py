"""Download and audit daily bookTicker completeness independently of ordering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.downloader import FeedDay, download_zip


SECONDS_PER_DAY = 86_400


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _longest_false_run(active: np.ndarray) -> tuple[int, int, int]:
    padded = np.concatenate((np.array([True]), active, np.array([True])))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    best_length = best_start = 0
    for start, end in zip(changes[::2], changes[1::2], strict=True):
        length = int(end - start)
        if length > best_length:
            best_length, best_start = length, int(start)
    return best_length, best_start, best_start + best_length


def audit_zip(path: Path, day: date, *, block_size: int = 64 * 1024 * 1024) -> dict[str, object]:
    member_name = f"BTCUSDT-bookTicker-{day}.csv"
    day_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    active = np.zeros(SECONDS_PER_DAY, dtype=bool)
    rows = invalid_day_rows = raw_time_decreases = same_ms_id_decreases = 0
    previous_time = previous_id = None
    minimum_time = None
    maximum_time = None
    with zipfile.ZipFile(path) as archive, archive.open(member_name) as source:
        reader = pacsv.open_csv(
            pa.PythonFile(source),
            read_options=pacsv.ReadOptions(block_size=block_size),
            convert_options=pacsv.ConvertOptions(include_columns=["transaction_time", "update_id"]),
        )
        for batch in reader:
            times = batch.column("transaction_time").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            ids = batch.column("update_id").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if len(times) == 0:
                continue
            if previous_time is not None:
                raw_time_decreases += int(times[0] < previous_time)
                same_ms_id_decreases += int(times[0] == previous_time and ids[0] < previous_id)
            differences = np.diff(times)
            raw_time_decreases += int((differences < 0).sum())
            same_ms_id_decreases += int(((differences == 0) & (np.diff(ids) < 0)).sum())
            second = (times - day_start_ms) // 1000
            valid = (second >= 0) & (second < SECONDS_PER_DAY)
            active[second[valid].astype(np.int64)] = True
            invalid_day_rows += int((~valid).sum())
            rows += len(times)
            batch_min, batch_max = int(times.min()), int(times.max())
            minimum_time = batch_min if minimum_time is None else min(minimum_time, batch_min)
            maximum_time = batch_max if maximum_time is None else max(maximum_time, batch_max)
            previous_time, previous_id = int(times[-1]), int(ids[-1])
    longest, gap_start, gap_end = _longest_false_run(active)
    active_bins = int(active.sum())
    active_pct = active_bins / SECONDS_PER_DAY * 100.0
    clean = active_pct >= 99.0 and longest <= 5 and invalid_day_rows == 0
    return {
        "date": day.isoformat(), "rows": rows, "active_1s_bins": active_bins,
        "active_1s_pct": active_pct, "inactive_1s_bins": SECONDS_PER_DAY - active_bins,
        "largest_inactive_gap_s": longest,
        "largest_gap_start_utc": str(timedelta(seconds=gap_start)),
        "largest_gap_end_utc": str(timedelta(seconds=gap_end)),
        "raw_time_decreases": raw_time_decreases,
        "same_ms_update_id_decreases": same_ms_id_decreases,
        "invalid_day_rows": invalid_day_rows,
        "minimum_timestamp_ms": minimum_time, "maximum_timestamp_ms": maximum_time,
        "status": "CLEAN" if clean else "GAPPED",
    }


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "coverage_audit_2024-01-31_2024-03-30"
    (output_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Date | Rows | Active 1s | Largest gap | Raw T decreases | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['date']} | {row['rows']:,} | {row['active_1s_pct']:.4f}% | "
            f"{row['largest_inactive_gap_s']:,}s | {row['raw_time_decreases']:,} | {row['status']} |"
        )
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 31))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 30))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    items = [FeedDay("bookTicker", day) for day in iter_days(args.start_date, args.end_date)]
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {executor.submit(download_zip, item, args.raw_dir): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            path = future.result()
            print(f"READY {item.day} {path.stat().st_size:,}", flush=True)
    rows = []
    for item in items:
        result = audit_zip(args.raw_dir / f"{item.stem}.zip", item.day)
        rows.append(result)
        print(
            f"AUDIT {item.day} rows={result['rows']:,} active={result['active_1s_pct']:.4f}% "
            f"gap={result['largest_inactive_gap_s']}s status={result['status']}", flush=True,
        )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
