"""Prove whether a monthly CSV is the daily CSVs concatenated without repeat headers."""

from __future__ import annotations

import argparse
import zlib
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("month", help="YYYY-MM")
    parser.add_argument("monthly_size", type=int, help="Remote monthly member uncompressed size")
    parser.add_argument("monthly_crc", help="Remote monthly member CRC32 in hexadecimal")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    archives = sorted(args.raw_dir.glob(f"BTCUSDT-bookTicker-{args.month}-??.zip"))
    if not archives:
        raise FileNotFoundError(f"No daily archives found for {args.month}")

    crc = 0
    total = 0
    canonical_header: bytes | None = None
    for index, path in enumerate(archives):
        with zipfile.ZipFile(path) as archive:
            info = archive.infolist()[0]
            with archive.open(info) as source:
                header = source.readline()
                if canonical_header is None:
                    canonical_header = header
                    crc = zlib.crc32(header, crc)
                    total += len(header)
                elif header != canonical_header:
                    raise RuntimeError(f"Header mismatch in {path}")
                while chunk := source.read(4 * 1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
                    total += len(chunk)
        print(
            f"DAY {index + 1:02d}/{len(archives):02d} {path.stem[-10:]} "
            f"combined_bytes={total:,}",
            flush=True,
        )

    expected_crc = int(args.monthly_crc, 16)
    print(f"combined_size={total}")
    print(f"monthly_size={args.monthly_size}")
    print(f"combined_crc={crc & 0xFFFFFFFF:08x}")
    print(f"monthly_crc={expected_crc:08x}")
    print(f"byte_identity_proven={total == args.monthly_size and crc == expected_crc}")


if __name__ == "__main__":
    main()
