"""벡터 스토어 추상 기반 클래스."""
from abc import ABC, abstractmethod

from src.models import SourceChunk


class VectorStore(ABC):
    """모든 벡터 DB 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    async def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """청크 및 메타데이터를 인덱싱한다."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[SourceChunk]:
        """유사 청크를 검색한다."""
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """컬렉션 전체를 삭제한다 (재인덱싱 시 사용)."""
        ...
