"""Download and convert the fixed BTCUSDT Phase 1 data window.

The raw ZIP archives are retained.  Extracted CSV files are removed only after
an atomically-written Parquet file has been validated successfully.
"""

from __future__ import annotations

import argparse
import os
import socket
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


BASE_URL = "https://data.binance.vision/data/futures/um/daily"
SYMBOL = "BTCUSDT"
START_DATE = date(2024, 1, 31)
END_DATE = date(2024, 3, 30)
FEEDS = ("trades", "bookTicker")

FEED_SCHEMAS = {
    "trades": pa.schema(
        [
            ("id", pa.int64()),
            ("price", pa.float64()),
            ("qty", pa.float64()),
            ("quote_qty", pa.float64()),
            ("time", pa.int64()),
            ("is_buyer_maker", pa.bool_()),
        ]
    ),
    "bookTicker": pa.schema(
        [
            ("update_id", pa.int64()),
            ("best_bid_price", pa.float64()),
            ("best_bid_qty", pa.float64()),
            ("best_ask_price", pa.float64()),
            ("best_ask_qty", pa.float64()),
            ("transaction_time", pa.int64()),
            ("event_time", pa.int64()),
        ]
    ),
}


@dataclass(frozen=True)
class FeedDay:
    feed: str
    day: date
    symbol: str = SYMBOL

    @property
    def stem(self) -> str:
        return f"{self.symbol}-{self.feed}-{self.day.isoformat()}"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.feed}/{self.symbol}/{self.stem}.zip"


@dataclass(frozen=True)
class ConversionResult:
    item: FeedDay
    rows: int
    zip_bytes: int
    parquet_bytes: int
    parquet_path: Path


def iter_days(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def expected_csv_name(item: FeedDay) -> str:
    return f"{item.stem}.csv"


def zip_path_for(item: FeedDay, raw_dir: Path) -> Path:
    return raw_dir / f"{item.stem}.zip"


def csv_path_for(item: FeedDay, raw_dir: Path) -> Path:
    return raw_dir / expected_csv_name(item)


def parquet_path_for(item: FeedDay, parquet_dir: Path) -> Path:
    return parquet_dir / item.feed / f"{item.stem}.parquet"


def validate_zip(path: Path, item: FeedDay) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty ZIP: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"invalid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    expected = expected_csv_name(item)
    if expected not in members:
        raise ValueError(f"{path} does not contain expected member {expected!r}")


def download_zip(item: FeedDay, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = zip_path_for(item, raw_dir)
    if target.exists() and target.stat().st_size > 0:
        validate_zip(target, item)
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    _download_with_resume(item.url, temporary)
    try:
        validate_zip(temporary, item)
    except ValueError:
        # A completed but invalid partial cannot be resumed safely.  Retry once
        # from byte zero while leaving the validated target path untouched.
        temporary.unlink(missing_ok=True)
        _download_with_resume(item.url, temporary)
        validate_zip(temporary, item)
    os.replace(temporary, target)
    return target


def _download_with_resume(
    url: str,
    temporary: Path,
    *,
    timeout_seconds: int = 15,
    max_consecutive_failures: int = 20,
    range_bytes: int = 8 * 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
) -> None:
    """Download ``url`` in resumable ranges with bounded no-progress retries."""

    consecutive_failures = 0
    last_reported = temporary.stat().st_size if temporary.exists() else 0
    while True:
        existing = temporary.stat().st_size if temporary.exists() else 0
        range_end = existing + range_bytes - 1
        headers = {
            "User-Agent": "hawkess-ml-phase1/1.0",
            "Range": f"bytes={existing}-{range_end}",
        }
        request = urllib.request.Request(url, headers=headers)
        if existing == last_reported:
            action = f"resume@{existing:,}" if existing else "download"
            print(f"{action} {url}", flush=True)
        before = existing
        total_size: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = response.getcode()
                if status == 206:
                    content_range = response.headers.get("Content-Range")
                    if not content_range or "/" not in content_range:
                        raise OSError(f"missing Content-Range for resumed request: {url}")
                    total_text = content_range.rsplit("/", 1)[1]
                    if total_text == "*":
                        raise OSError(f"unknown total size in Content-Range: {content_range}")
                    total_size = int(total_text)
                    mode = "ab" if existing else "wb"
                elif status == 200:
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length is not None else None
                    mode = "wb"
                else:
                    raise OSError(f"unexpected HTTP status {status} for {url}")
                with temporary.open(mode) as destination:
                    while True:
                        chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        destination.write(chunk)
                actual_size = temporary.stat().st_size
                if total_size is not None and actual_size > total_size:
                    raise OSError(
                        f"oversized download for {url}: {actual_size:,}/{total_size:,} bytes"
                    )
            if total_size is not None and actual_size == total_size:
                return
            if actual_size <= before:
                raise OSError(f"range request made no progress for {url}")
            consecutive_failures = 0
            if actual_size - last_reported >= 64 * 1024 * 1024:
                if total_size is None:
                    print(f"progress {actual_size:,} bytes: {url}", flush=True)
                else:
                    print(
                        f"progress {actual_size:,}/{total_size:,} bytes: {url}",
                        flush=True,
                    )
                last_reported = actual_size
            continue
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and existing:
                # The server reports that the requested start is at/after EOF;
                # ZIP validation in the caller decides whether the file is done.
                return
            last_error = exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            last_error = exc

        after = temporary.stat().st_size if temporary.exists() else 0
        if after > before:
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            raise RuntimeError(
                f"download made no progress in {max_consecutive_failures} consecutive attempts: {url}"
            ) from last_error
        delay = min(2 ** (consecutive_failures - 1), 30)
        print(
            f"retry in {delay}s after {type(last_error).__name__}: {last_error}",
            flush=True,
        )
        time.sleep(delay)


def extract_csv(item: FeedDay, raw_dir: Path) -> Path:
    source = zip_path_for(item, raw_dir)
    validate_zip(source, item)
    target = csv_path_for(item, raw_dir)
    if target.exists() and target.stat().st_size > 0:
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(source) as archive:
            with archive.open(expected_csv_name(item)) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def csv_to_parquet(
    item: FeedDay,
    csv_path: Path,
    parquet_dir: Path,
    *,
    block_size: int = 64 * 1024 * 1024,
) -> tuple[Path, int]:
    target = parquet_path_for(item, parquet_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)

    schema = FEED_SCHEMAS[item.feed]
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            column_types={field.name: field.type for field in schema},
            strings_can_be_null=False,
        ),
    )

    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in reader:
            if batch.schema != schema:
                batch = batch.cast(schema)
            writer.write_batch(batch)
            rows += batch.num_rows
        writer.close()
        writer = None

        metadata = pq.read_metadata(temporary)
        if metadata.num_rows != rows:
            raise RuntimeError(
                f"Parquet row validation failed for {item.stem}: "
                f"wrote {rows}, footer reports {metadata.num_rows}"
            )
        os.replace(temporary, target)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return target, rows


def convert_feed_day(
    item: FeedDay,
    raw_dir: Path,
    parquet_dir: Path,
) -> ConversionResult:
    source_zip = zip_path_for(item, raw_dir)
    target = parquet_path_for(item, parquet_dir)

    if target.exists() and target.stat().st_size > 0:
        metadata = pq.read_metadata(target)
        csv_path_for(item, raw_dir).unlink(missing_ok=True)
        return ConversionResult(
            item=item,
            rows=metadata.num_rows,
            zip_bytes=source_zip.stat().st_size,
            parquet_bytes=target.stat().st_size,
            parquet_path=target,
        )

    csv_path = extract_csv(item, raw_dir)
    target, rows = csv_to_parquet(item, csv_path, parquet_dir)
    csv_path.unlink()
    return ConversionResult(
        item=item,
        rows=rows,
        zip_bytes=source_zip.stat().st_size,
        parquet_bytes=target.stat().st_size,
        parquet_path=target,
    )


def build_items(start: date, end: date, feeds: Sequence[str], symbol: str = SYMBOL) -> list[FeedDay]:
    unknown = set(feeds) - set(FEEDS)
    if unknown:
        raise ValueError(f"unknown feeds: {sorted(unknown)}")
    return [FeedDay(feed, day, symbol) for day in iter_days(start, end) for feed in feeds]


def run(
    *,
    start: date = START_DATE,
    end: date = END_DATE,
    feeds: Sequence[str] = FEEDS,
    symbol: str = SYMBOL,
    raw_dir: Path = Path("data/raw"),
    parquet_dir: Path = Path("data/parquet"),
    download_workers: int = 4,
) -> list[ConversionResult]:
    items = build_items(start, end, feeds, symbol)
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {executor.submit(download_zip, item, raw_dir): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            path = future.result()
            print(f"downloaded {item.stem}: {path.stat().st_size:,} bytes", flush=True)

    results = []
    for item in items:
        result = convert_feed_day(item, raw_dir, parquet_dir)
        results.append(result)
        print(
            f"parquet {item.stem}: {result.rows:,} rows, "
            f"{result.parquet_bytes:,} bytes",
            flush=True,
        )
    return results


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, default=START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=END_DATE)
    parser.add_argument("--feeds", nargs="+", choices=FEEDS, default=list(FEEDS))
    parser.add_argument("--symbol", type=str, default=SYMBOL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--download-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        start=args.start_date,
        end=args.end_date,
        feeds=args.feeds,
        symbol=args.symbol,
        raw_dir=args.raw_dir,
        parquet_dir=args.parquet_dir,
        download_workers=args.download_workers,
    )


if __name__ == "__main__":
    main()
