"""Minimal file locks with retry/backoff.

The upstream code uses proper-lockfile with retries. This module mirrors that
behavior using atomic lockfile creation.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(path: str | Path, retries: int = 30, min_timeout_ms: int = 5, max_timeout_ms: int = 100) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = min_timeout_ms
    acquired = False
    for _ in range(retries + 1):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(delay_ms / 1000)
            delay_ms = min(max_timeout_ms, max(delay_ms, min_timeout_ms) * 2)
    if not acquired:
        raise TimeoutError(f"Could not acquire lock {lock_path}")
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
