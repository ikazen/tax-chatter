"""Gemini API LLM 구현체."""
from collections.abc import AsyncIterator

import google.generativeai as genai

from config.settings import settings
from src.llm.llm_provider import LLMProvider
from src.models import Message


class GeminiProvider(LLMProvider):
    """Google Gemini API 기반 LLM provider."""

    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            model_name=settings.gemini_model,
        )

    def _to_gemini_history(self, messages: list[Message]) -> list[dict]:
        role_map = {"user": "user", "assistant": "model"}
        return [
            {"role": role_map[m.role], "parts": [m.content]}
            for m in messages[:-1]  # 마지막 메시지는 send_message로 전달
        ]

    async def generate(self, system_prompt: str, messages: list[Message]) -> str:
        """단일 응답 생성."""
        chat = self._model.start_chat(history=self._to_gemini_history(messages))
        response = await chat.send_message_async(
            messages[-1].content,
            generation_config={"system_instruction": system_prompt},
        )
        return response.text

    async def stream(
        self, system_prompt: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성."""
        chat = self._model.start_chat(history=self._to_gemini_history(messages))
        async for chunk in await chat.send_message_async(
            messages[-1].content,
            generation_config={"system_instruction": system_prompt},
            stream=True,
        ):
            if chunk.text:
                yield chunk.text
