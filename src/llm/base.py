"""LLM provider 추상 기반 클래스."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.models import Message


class LLMProvider(ABC):
    """모든 LLM 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        """단일 응답 생성."""
        ...

    @abstractmethod
    async def stream(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성."""
        ...
