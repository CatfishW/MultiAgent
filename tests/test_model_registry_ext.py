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
