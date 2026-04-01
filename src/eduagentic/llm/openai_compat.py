from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import base64
import time

import httpx

from ..core.contracts import ModelMessage, ModelResponse
from ..utils.cache import JsonDiskCache, LRUCache
from ..utils.text import stable_hash


def _read_image_as_data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png"
    suffix = p.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class OpenAICompatClient:
    """Minimal OpenAI-compatible client with model-response caching.

    It targets local or self-hosted endpoints that expose `/models` and
    `/chat/completions` under a shared base URL.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        cache_dir: str | None = None,
        chat_path: str = "chat/completions",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.chat_path = chat_path.lstrip("/")
        self.memory_cache = LRUCache(max_size=1024, ttl_s=60 * 60)
        self.disk_cache = JsonDiskCache(cache_dir) if cache_dir else None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return stable_hash(self.base_url, self.chat_path, payload)

    def _normalize_messages(self, messages: list[ModelMessage | dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ModelMessage):
                normalized.append(asdict(message))
            else:
                normalized.append(dict(message))
        return normalized

    def _merge_images_into_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not images:
            return messages
        merged = [dict(message) for message in messages]
        user_index = None
        for idx in range(len(merged) - 1, -1, -1):
            if merged[idx].get("role") == "user":
                user_index = idx
                break
        if user_index is None:
            merged.append({"role": "user", "content": []})
            user_index = len(merged) - 1

        content = merged[user_index].get("content")
        if isinstance(content, str):
            content_list: list[dict[str, Any]] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_list = list(content)
        else:
            content_list = []
        for image in images:
            image_url = image if image.startswith("http") or image.startswith("data:") else _read_image_as_data_url(image)
            content_list.append({"type": "image_url", "image_url": {"url": image_url}})
        merged[user_index]["content"] = content_list
        return merged

    async def list_models(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.base_url}/models", headers=self._headers())
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [dict(item) for item in payload["data"]]
        if isinstance(payload, list):
            return [dict(item) if isinstance(item, dict) else {"id": str(item)} for item in payload]
        raise ValueError(f"Unexpected /models payload: {payload!r}")

    async def chat(
        self,
        *,
        model: str,
        messages: list[ModelMessage | dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 900,
        images: list[str] | None = None,
        use_cache: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> ModelResponse:
        normalized_messages = self._merge_images_into_messages(self._normalize_messages(messages), images)
        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(extra)

        cache_key = self._cache_key(payload)
        if use_cache:
            memory_hit = self.memory_cache.get(cache_key)
            if memory_hit is not None:
                return ModelResponse(**memory_hit)
            if self.disk_cache:
                disk_hit = self.disk_cache.get(cache_key)
                if disk_hit is not None:
                    self.memory_cache.set(cache_key, disk_hit)
                    return ModelResponse(**disk_hit)

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/{self.chat_path}",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = self._extract_text(raw)
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        result = ModelResponse(text=text, model=model, usage=usage, raw=raw, latency_ms=latency_ms)
        serializable = asdict(result)
        self.memory_cache.set(cache_key, serializable)
        if self.disk_cache:
            self.disk_cache.set(cache_key, serializable)
        return result

    def _extract_text(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return str(payload)
        if isinstance(payload.get("choices"), list) and payload["choices"]:
            choice = payload["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                text = content.get("text") or content.get("content")
                if isinstance(text, str):
                    return text
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item.get("content"), str):
                            parts.append(item["content"])
                if parts:
                    return "\n".join(parts)
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content
            if isinstance(choice.get("text"), str):
                return choice["text"]
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text
        if isinstance(payload.get("content"), str):
            return payload["content"]
        if isinstance(payload.get("text"), str):
            return payload["text"]
        return str(payload)
