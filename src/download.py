"""Resumable bounded HTTP ranges for large artifacts on unreliable connections."""

import hashlib
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common import sha256_file


def fetch_ranges(
    url: str,
    destination: Path,
    size: int,
    expected_sha256: str | None = None,
    workers: int = 16,
    chunk_bytes: int = 8 * 1024**2,
) -> Path:
    """Resume independent checked byte ranges; publish only a complete verified file.

    Signed locations and authorization are never logged. Responses must match the
    exact requested range and total length; a server ignoring Range is rejected.
    """
    if size <= 0 or not 1 <= workers <= 16 or chunk_bytes <= 0:
        raise ValueError("Invalid artifact size, worker count or chunk size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        if expected_sha256 is None or sha256_file(destination) == expected_sha256:
            return destination
    identity = hashlib.sha256(f"{destination.name}:{size}:{expected_sha256}".encode()).hexdigest()[
        :16
    ]
    parts = destination.parent / f".parts-{identity}"
    parts.mkdir(exist_ok=True)
    ranges = [
        (start, min(size - 1, start + chunk_bytes - 1)) for start in range(0, size, chunk_bytes)
    ]

    def fetch(pair):
        start, end = pair
        part = parts / f"{start:014d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                        "User-Agent": "football-movement-course-project/0.1",
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    actual = response.headers.get("Content-Range")
                    if response.status != 206 or actual != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server returned a mismatched HTTP byte range")
                    pending = part.with_suffix(".pending")
                    with pending.open("wb") as handle:
                        remaining = end - start + 1
                        while remaining:
                            block = response.read(min(256 * 1024, remaining))
                            if not block:
                                raise OSError("Incomplete artifact range")
                            handle.write(block)
                            remaining -= len(block)
                    pending.replace(part)
                return
            except (OSError, urllib.error.URLError) as exc:
                if attempt == 4:
                    raise RuntimeError(f"Artifact range {start}-{end} failed: {type(exc).__name__}")
                time.sleep(min(2**attempt, 8))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, pair) for pair in ranges]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(ranges):
                print(f"{destination.name}: {count}/{len(ranges)} ranges ready", flush=True)
    assembled = destination.with_suffix(destination.suffix + ".assembling")
    with assembled.open("wb") as output:
        for start, _ in ranges:
            with (parts / f"{start:014d}.part").open("rb") as source:
                for block in iter(lambda: source.read(1024**2), b""):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if assembled.stat().st_size != size:
        raise RuntimeError("Assembled artifact length does not match metadata")
    if expected_sha256 and sha256_file(assembled) != expected_sha256:
        raise RuntimeError("Artifact checksum failed; output has not been published")
    assembled.replace(destination)
    # Exact checked part files are removed after publication; no recursive deletion.
    for start, _ in ranges:
        (parts / f"{start:014d}.part").unlink()
    parts.rmdir()
    return destination
