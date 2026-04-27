from __future__ import annotations

import pytest

from eduagentic.config import AppConfig, EndpointConfig
from eduagentic.core.contracts import ModelResponse
from eduagentic.llm.openai_compat import OpenAICompatClient
from eduagentic.llm.registry import ModelDescriptor, ModelRegistry


class DummyClient:
    def __init__(self, models):
        self._models = models

    async def list_models(self):
        return self._models


def test_model_descriptor_prefers_small_models():
    items = [
        ModelDescriptor(endpoint="llm", model_id="mega-70b", capability="text", raw={}),
        ModelDescriptor(endpoint="llm", model_id="fast-7b", capability="text", raw={}),
        ModelDescriptor(endpoint="llm", model_id="tiny-mini", capability="text", raw={}),
    ]
    ranked = sorted(items, key=lambda item: item.rank_key)
    assert ranked[0].model_id == "tiny-mini"


@pytest.mark.asyncio
async def test_registry_refresh_uses_client(monkeypatch, tmp_path):
    config = AppConfig(
        endpoints={
            "llm": EndpointConfig(name="llm", base_url="https://example.invalid/v1", capability="text", default_model="fallback-model")
        }
    )
    registry = ModelRegistry(config)
    monkeypatch.setattr(registry, "client_for", lambda endpoint: DummyClient([{"id": "small-model"}, {"id": "bigger-13b"}]))
    models = await registry.refresh(force=True)
    assert "llm" in models
    assert [item.model_id for item in models["llm"]][:2] == ["small-model", "bigger-13b"]


def test_openai_compat_extract_text():
    client = OpenAICompatClient(base_url="https://example.invalid/v1")
    payload = {"choices": [{"message": {"content": "hello world"}}]}
    assert client._extract_text(payload) == "hello world"


@pytest.mark.asyncio
async def test_pick_model_allows_multimodal_from_vision_capable_text_endpoint():
    config = AppConfig(
        endpoints={
            "llm": EndpointConfig(
                name="llm",
                base_url="https://example.invalid/v1",
                capability="text",
                supports_vision=True,
            )
        }
    )
    registry = ModelRegistry(config)
    registry._models = {
        "llm": [ModelDescriptor(endpoint="llm", model_id="qwen-4b", capability="text", raw={})]
    }

    descriptor = await registry.pick_model(capability="multimodal")

    assert descriptor.model_id == "qwen-4b"


@pytest.mark.asyncio
async def test_pick_model_prefers_endpoint_default_model_when_advertised():
    config = AppConfig(
        endpoints={
            "llm": EndpointConfig(
                name="llm",
                base_url="https://example.invalid/v1",
                capability="text",
                default_model="paper-27b",
            )
        }
    )
    registry = ModelRegistry(config)
    registry._models = {
        "llm": [
            ModelDescriptor(endpoint="llm", model_id="qwen-4b", capability="text", raw={}),
            ModelDescriptor(endpoint="llm", model_id="paper-27b", capability="text", raw={}),
        ]
    }

    descriptor = await registry.pick_model(capability="text", endpoint_name="llm")

    assert descriptor.model_id == "paper-27b"


@pytest.mark.asyncio
async def test_pick_model_raises_for_unknown_pinned_model_when_models_are_known():
    config = AppConfig(
        endpoints={
            "llm": EndpointConfig(
                name="llm",
                base_url="https://example.invalid/v1",
                capability="text",
                pinned_model="missing-27b",
            )
        }
    )
    registry = ModelRegistry(config)
    registry._models = {
        "llm": [ModelDescriptor(endpoint="llm", model_id="qwen-4b", capability="text", raw={})]
    }

    with pytest.raises(LookupError, match="was not advertised"):
        await registry.pick_model(capability="text", endpoint_name="llm")
