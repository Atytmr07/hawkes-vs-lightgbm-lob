"""Download a large immutable archive with resumable HTTP range requests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download_range(url: str, path: Path, start: int, end: int) -> tuple[Path, int]:
    expected = end - start + 1
    failures = 0
    while True:
        existing = path.stat().st_size if path.exists() else 0
        if existing > expected:
            raise RuntimeError(f"Oversized part {path}: {existing} > {expected}")
        if existing == expected:
            return path, expected

        request_start = start + existing
        with path.open("ab") as output:
            completed = subprocess.run(
                [
                    "curl.exe", "-L", "--fail", "--silent", "--show-error",
                    "--speed-limit", "1024", "--speed-time", "45",
                    "--max-time", "300", "--range", f"{request_start}-{end}",
                    "--output", "-", url,
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            failures += 1
            if failures > 20:
                raise RuntimeError(
                    f"Repeated curl failure for {path.name}: "
                    f"{completed.stderr.decode(errors='replace').strip()}"
                )
            time.sleep(min(2 * failures, 15))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix(output.suffix + ".prefix")
    parts_dir = output.with_suffix(output.suffix + ".parts")
    assembling = output.with_suffix(output.suffix + ".assembling")

    if output.exists() and output.stat().st_size == args.expected_size:
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure in {bad}")
        print(f"READY {output} {output.stat().st_size:,}", flush=True)
        return

    if output.exists() and not prefix.exists():
        output.replace(prefix)
    if not prefix.exists():
        prefix.touch()
    prefix_size = prefix.stat().st_size
    if prefix_size >= args.expected_size:
        raise RuntimeError(f"Invalid prefix size: {prefix_size:,}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    remaining = args.expected_size - prefix_size
    chunk_size = (remaining + args.connections - 1) // args.connections
    ranges: list[tuple[int, Path, int, int]] = []
    for index in range(args.connections):
        start = prefix_size + index * chunk_size
        if start >= args.expected_size:
            break
        end = min(args.expected_size - 1, start + chunk_size - 1)
        ranges.append((index, parts_dir / f"part-{index:02d}.bin", start, end))

    with ThreadPoolExecutor(max_workers=args.connections) as executor:
        futures = {
            executor.submit(download_range, args.url, path, start, end): index
            for index, path, start, end in ranges
        }
        for future in as_completed(futures):
            path, size = future.result()
            print(f"PART {futures[future]:02d} {path.name} {size:,}", flush=True)

    with assembling.open("wb") as destination:
        for source_path in [prefix, *(path for _, path, _, _ in ranges)]:
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    if assembling.stat().st_size != args.expected_size:
        raise RuntimeError(
            f"Assembled size mismatch: {assembling.stat().st_size:,} != {args.expected_size:,}"
        )
    with zipfile.ZipFile(assembling) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC failure in {bad}")
    os.replace(assembling, output)

    prefix.unlink()
    for _, path, _, _ in ranges:
        path.unlink()
    parts_dir.rmdir()
    print(f"READY {output} {output.stat().st_size:,}", flush=True)


if __name__ == "__main__":
    main()
