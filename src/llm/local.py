"""Ollama HTTP API 기반 LLM 구현체."""
from collections.abc import AsyncIterator

import httpx

from config.settings import settings
from src.llm.base import LLMProvider
from src.models import Message

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3"


class OllamaProvider(LLMProvider):
    """Ollama 로컬 LLM provider."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_OLLAMA_URL).rstrip("/")
        self._model = model or _DEFAULT_OLLAMA_MODEL
        self._client = httpx.AsyncClient(timeout=120.0)

    def _build_messages(
        self, system_prompt: str, messages: list[Message]
    ) -> list[dict[str, str]]:
        """Ollama /api/chat 포맷으로 메시지를 변환한다."""
        result: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in messages:
            result.append({"role": m.role, "content": m.content})
        return result

    async def generate(
        self, system_prompt: str, messages: list[Message]
    ) -> str:
        """단일 응답 생성."""
        payload = {
            "model": self._model,
            "messages": self._build_messages(system_prompt, messages),
            "stream": False,
        }
        response = await self._client.post(
            f"{self._base_url}/api/chat", json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    async def stream(
        self, system_prompt: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성."""
        payload = {
            "model": self._model,
            "messages": self._build_messages(system_prompt, messages),
            "stream": True,
        }
        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            import json

            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
