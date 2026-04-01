from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import BenchmarkExample, TaskRegime


@dataclass(slots=True)
class DatasetSpec:
    name: str
    family: str
    regime: TaskRegime
    default_loader: str
    default_source: str | None = None
    subset: str | None = None
    split: str = "test"
    modality: str = "text"
    notes: str = ""
    requires_local_assets: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    def load(self, source: str | None = None, split: str | None = None, limit: int | None = None) -> list[BenchmarkExample]:
        raise NotImplementedError
