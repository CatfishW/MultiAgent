from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import json
import time


def _default_serializer(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


class LRUCache:
    def __init__(self, max_size: int = 512, ttl_s: float | None = None) -> None:
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            created_at, payload = value
            if self.ttl_s is not None and (time.time() - created_at) > self.ttl_s:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return payload

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


class JsonDiskCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> Path:
        path = self._path_for(key)
        with self._lock:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_default_serializer), encoding="utf-8")
        return path

    def get_or_compute(self, key: str, factory: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value
