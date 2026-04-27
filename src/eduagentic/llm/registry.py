from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import asyncio
import re

from ..config import AppConfig, EndpointConfig
from ..utils.cache import JsonDiskCache
from .openai_compat import OpenAICompatClient


_MODEL_SIZE_RE = re.compile(r"(?P<size>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[bm])", re.IGNORECASE)


@dataclass(slots=True)
class ModelDescriptor:
    endpoint: str
    model_id: str
    capability: str
    raw: dict[str, Any]

    @property
    def rank_key(self) -> tuple[int, float, str]:
        lowered = self.model_id.lower()
        explicit_small = any(tag in lowered for tag in ["mini", "small", "tiny"])
        compact_family = any(tag in lowered for tag in ["1b", "2b", "3b", "4b", "7b", "8b"])
        if explicit_small:
            class_rank = 0
        elif compact_family:
            class_rank = 1
        else:
            class_rank = 2
        size_match = _MODEL_SIZE_RE.search(lowered)
        if size_match:
            size = float(size_match.group("size"))
            if size_match.group("unit").lower() == "m":
                size = size / 1000.0
        else:
            size = 0.5 if explicit_small else 999.0
        return (class_rank, size, self.model_id)


class ModelRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cache = JsonDiskCache(Path(config.pipeline.cache_dir) / "model_registry")
        self._models: dict[str, list[ModelDescriptor]] = {}

    def client_for(self, endpoint: EndpointConfig) -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout_s=endpoint.timeout_s,
            request_retries=max(1, int(endpoint.max_retries)),
            cache_dir=str(Path(self.config.pipeline.cache_dir) / endpoint.name),
        )

    async def refresh(self, force: bool = False) -> dict[str, list[ModelDescriptor]]:
        results: dict[str, list[ModelDescriptor]] = {}
        tasks = []
        for name, endpoint in self.config.endpoints.items():
            if not endpoint.enabled:
                continue
            cache_key = f"{name}_models"
            cached = None if force else self.cache.get(cache_key)
            if cached is not None:
                results[name] = [ModelDescriptor(**item) for item in cached]
                continue
            tasks.append((name, endpoint, self.client_for(endpoint).list_models()))
        if tasks:
            fetched = await asyncio.gather(*[coroutine for _, _, coroutine in tasks], return_exceptions=True)
            for (name, endpoint, _), value in zip(tasks, fetched):
                if isinstance(value, Exception):
                    default_model = endpoint.default_model
                    fallback = []
                    if default_model:
                        fallback.append(ModelDescriptor(endpoint=name, model_id=default_model, capability=endpoint.capability, raw={"id": default_model, "fallback": True}))
                    results[name] = fallback
                    continue
                models = [
                    ModelDescriptor(endpoint=name, model_id=item.get("id", "unknown"), capability=endpoint.capability, raw=dict(item))
                    for item in value
                ]
                results[name] = models
                self.cache.set(f"{name}_models", [asdict(model) for model in models])
        self._models = results
        return results

    async def get_models(self, endpoint_name: str) -> list[ModelDescriptor]:
        if endpoint_name not in self._models:
            await self.refresh()
        return self._models.get(endpoint_name, [])

    @staticmethod
    def _endpoint_supports(endpoint_config: EndpointConfig, capability: str) -> bool:
        if capability == "multimodal":
            return endpoint_config.capability == "multimodal" or endpoint_config.supports_vision
        if capability == "text":
            return endpoint_config.capability in {"text", "multimodal"}
        return endpoint_config.capability == capability

    async def _pick_configured_model(
        self,
        *,
        capability: str,
        endpoint: str,
        endpoint_config: EndpointConfig,
        field_name: str,
        strict: bool,
    ) -> ModelDescriptor | None:
        model_id = getattr(endpoint_config, field_name)
        if not model_id or not self._endpoint_supports(endpoint_config, capability):
            return None
        models = await self.get_models(endpoint)
        for model in models:
            if model.model_id == model_id:
                return model
        if models:
            if strict:
                available = ", ".join(sorted(model.model_id for model in models))
                raise LookupError(
                    f"Configured {field_name} {model_id!r} for endpoint {endpoint!r} "
                    f"was not advertised by {endpoint_config.base_url}/models. "
                    f"Available ids: {available}"
                )
            return None
        return ModelDescriptor(
            endpoint=endpoint,
            model_id=model_id,
            capability=endpoint_config.capability,
            raw={"id": model_id, field_name: True, "unverified": True},
        )

    async def pick_model(
        self,
        *,
        capability: str,
        endpoint_name: str | None = None,
        prefer_fast: bool = True,
    ) -> ModelDescriptor:
        candidates: list[ModelDescriptor] = []
        endpoints = [endpoint_name] if endpoint_name else list(self.config.endpoints.keys())
        # Honor endpoint.pinned_model first: if any candidate endpoint declares a
        # pinned model, return exactly that model id. This bypasses the
        # prefer_fast rank heuristic so dual-backbone experiments can force a
        # specific backbone (e.g. Qwen3.5-4B or Qwen3.5-27B-FP8) for every
        # agent in the pipeline.
        for endpoint in endpoints:
            endpoint_config = self.config.endpoints.get(endpoint)
            if endpoint_config is None:
                continue
            descriptor = await self._pick_configured_model(
                capability=capability,
                endpoint=endpoint,
                endpoint_config=endpoint_config,
                field_name="pinned_model",
                strict=True,
            )
            if descriptor is not None:
                return descriptor
        for endpoint in endpoints:
            endpoint_config = self.config.endpoints.get(endpoint)
            if endpoint_config is None:
                continue
            descriptor = await self._pick_configured_model(
                capability=capability,
                endpoint=endpoint,
                endpoint_config=endpoint_config,
                field_name="default_model",
                strict=False,
            )
            if descriptor is not None:
                return descriptor
        for endpoint in endpoints:
            endpoint_config = self.config.endpoints.get(endpoint)
            models = await self.get_models(endpoint)
            for model in models:
                if capability == "multimodal":
                    if model.capability != "multimodal" and not (endpoint_config and endpoint_config.supports_vision):
                        continue
                if capability == "text" and model.capability not in {"text", "multimodal"}:
                    continue
                candidates.append(model)
        if not candidates:
            raise LookupError(f"No models available for capability={capability!r}")
        if prefer_fast:
            candidates.sort(key=lambda item: item.rank_key)
        else:
            candidates.sort(key=lambda item: item.model_id)
        return candidates[0]
