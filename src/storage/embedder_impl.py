"""SentenceTransformers 기반 임베딩 구현체."""
import asyncio

from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.storage.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """SentenceTransformers 라이브러리를 사용하는 임베더."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedder_model
        self._model = SentenceTransformer(self._model_name)

    async def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""
        result = await asyncio.to_thread(self._model.encode, text)
        return result.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """복수 텍스트를 배치 임베딩한다."""
        results = await asyncio.to_thread(self._model.encode, texts)
        return results.tolist()
