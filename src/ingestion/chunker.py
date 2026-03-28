"""문서 청킹 모듈."""
import copy
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkWithMetadata:
    """청크 텍스트와 메타데이터."""

    text: str
    metadata: dict


class DocumentChunker:
    """문서를 일정 크기의 청크로 분할한다."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict) -> list[ChunkWithMetadata]:
        """단일 문서를 청크로 분할한다."""
        if not text or not text.strip():
            return []
        splits = self._splitter.split_text(text)
        return [
            ChunkWithMetadata(text=s, metadata=copy.copy(metadata))
            for s in splits
        ]

    def chunk_batch(
        self, documents: list[tuple[str, dict]]
    ) -> list[ChunkWithMetadata]:
        """복수 문서를 일괄 청킹한다."""
        result: list[ChunkWithMetadata] = []
        for text, metadata in documents:
            result.extend(self.chunk(text, metadata))
        return result
