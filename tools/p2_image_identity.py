"""Ordered, content-based identity of the original image files used by P2."""
from __future__ import annotations

import hashlib
from pathlib import Path


def image_bytes_sha256(image_dir, ids):
    """Hash UTF-8 ID, one NUL byte, and the 32 raw bytes of each file's SHA-256."""
    directory = Path(image_dir)
    combined = hashlib.sha256()
    for image_id in ids:
        file_digest = hashlib.sha256()
        with (directory / image_id).open("rb") as image:
            for block in iter(lambda: image.read(1024 * 1024), b""):
                file_digest.update(block)
        combined.update(image_id.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_digest.digest())
    return combined.hexdigest()
