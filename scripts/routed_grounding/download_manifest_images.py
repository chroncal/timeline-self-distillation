#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download exactly the official COCO images referenced by a pilot manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

# COCO's public S3 endpoint supports HTTP.  Use it because some institutional
# HTTPS proxies present a certificate for the S3 backend rather than this
# hostname; every downloaded file is still provenance-linked in the receipt.
COCO_TRAIN2014_ROOT = "http://images.cocodataset.org/train2014"


def _receipt(record: dict, destination: Path, url: str, status: str) -> dict:
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "image_id": record["image_id"],
        "path": str(destination),
        "url": url,
        "status": status,
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _download(record: dict, *, retries: int) -> dict:
    destination = Path(record["image_path"])
    expected_name = str(record["file_name"])
    if destination.name != expected_name:
        raise ValueError(f"manifest path/name mismatch: {destination} != {expected_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{COCO_TRAIN2014_ROOT}/{expected_name}"
    if destination.is_file() and destination.stat().st_size > 0:
        return _receipt(record, destination, url, "existing")

    partial = destination.with_suffix(destination.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "routed-grounding-repair/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if partial.stat().st_size <= 0:
                raise OSError("downloaded an empty file")
            partial.replace(destination)
            return _receipt(record, destination, url, "downloaded")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0 or args.retries < 0:
        parser.error("workers must be positive and retries non-negative")

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_image = {str(record["image_id"]): record for record in records}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(lambda record: _download(record, retries=args.retries), by_image.values()))

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_root": COCO_TRAIN2014_ROOT,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "count": len(receipts),
        "images": sorted(receipts, key=lambda item: str(item["image_id"])),
    }
    args.receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
