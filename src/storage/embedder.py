"""임베딩 모델 추상 기반 클래스."""
from abc import ABC, abstractmethod


class Embedder(ABC):
    """모든 임베딩 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """복수 텍스트를 배치 임베딩한다."""
        ...
