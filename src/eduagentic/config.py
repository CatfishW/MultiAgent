from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os

import yaml

from .core.contracts import BudgetPolicy


@dataclass(slots=True)
class EndpointConfig:
    name: str
    base_url: str
    capability: str = "text"
    api_key_env: str | None = None
    default_model: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 2
    enabled: bool = True

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass(slots=True)
class RetrieverConfig:
    chunk_size: int = 220
    chunk_overlap: int = 40
    lexical_weight: float = 0.58
    char_weight: float = 0.17
    latent_weight: float = 0.25
    top_k: int = 8
    final_k: int = 4
    max_features: int = 40000
    latent_dim: int = 128
    mmr_lambda: float = 0.68
    cache_dir: str = ".cache/eduagentic/retrieval"


@dataclass(slots=True)
class RouterConfig:
    evidence_threshold: float = 0.52
    coordination_threshold: float = 0.5
    prefer_hybrid_for_mixed: bool = True
    model_path: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    default_architecture: str = "hybrid_fast"
    enable_swarm_runtime: bool = False
    enable_critic: bool = True
    enable_rubric_agent: bool = True
    enable_diagnoser: bool = True
    enable_planner_llm: bool = False
    use_fast_rule_planner: bool = True
    parallel_specialists: bool = True
    cache_dir: str = ".cache/eduagentic/models"


@dataclass(slots=True)
class DatasetConfig:
    root_dir: str = "data"
    use_huggingface_when_available: bool = True
    local_only: bool = False
    registry_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    endpoints: dict[str, EndpointConfig] = field(default_factory=dict)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)


DEFAULT_CONFIG = AppConfig(
    endpoints={
        "llm": EndpointConfig(
            name="llm",
            base_url="https://game.agaii.org/llm/v1",
            capability="text",
            api_key_env="AGAII_LLM_API_KEY",
        ),
        "mllm": EndpointConfig(
            name="mllm",
            base_url="https://game.agaii.org/mllm/v1",
            capability="multimodal",
            api_key_env="AGAII_MLLM_API_KEY",
        ),
    }
)


def _merge_dataclass(base: Any, updates: dict[str, Any]) -> Any:
    values = asdict(base)
    for key, value in updates.items():
        current = values.get(key)
        if hasattr(current, "__dict__") and isinstance(value, dict):
            values[key] = _merge_dataclass(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(value)
            values[key] = merged
        else:
            values[key] = value
    return type(base)(**values)


def load_app_config(config: str | Path | dict[str, Any] | None = None) -> AppConfig:
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, AppConfig):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".json"}:
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text) or {}
    elif isinstance(config, dict):
        payload = config
    else:
        raise TypeError(f"Unsupported config input: {type(config)!r}")

    merged = _merge_dataclass(DEFAULT_CONFIG, payload)
    endpoints: dict[str, EndpointConfig] = {}
    for name, value in merged.endpoints.items():
        if isinstance(value, EndpointConfig):
            endpoints[name] = value
        else:
            endpoint_payload = dict(value)
            endpoint_payload.setdefault("name", name)
            endpoints[name] = EndpointConfig(**endpoint_payload)
    merged.endpoints = endpoints
    if not isinstance(merged.retriever, RetrieverConfig):
        merged.retriever = RetrieverConfig(**merged.retriever)
    if not isinstance(merged.router, RouterConfig):
        merged.router = RouterConfig(**merged.router)
    if not isinstance(merged.pipeline, PipelineConfig):
        merged.pipeline = PipelineConfig(**merged.pipeline)
    if not isinstance(merged.datasets, DatasetConfig):
        merged.datasets = DatasetConfig(**merged.datasets)
    if not isinstance(merged.budget, BudgetPolicy):
        merged.budget = BudgetPolicy(**merged.budget)
    return merged
