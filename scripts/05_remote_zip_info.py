"""Inspect a remote ZIP central directory using small HTTP range reads."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            end = self.size - 1
        else:
            end = min(self.size - 1, self.position + size - 1)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        self.position += len(data)
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    reader = HTTPRangeReader(args.url)
    with zipfile.ZipFile(reader) as archive:
        print(f"archive_bytes={reader.size}")
        for info in archive.infolist():
            print(
                f"member={info.filename} compressed={info.compress_size} "
                f"uncompressed={info.file_size} method={info.compress_type} crc={info.CRC:08x}"
            )


if __name__ == "__main__":
    main()
