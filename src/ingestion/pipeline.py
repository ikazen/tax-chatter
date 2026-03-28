"""문서 수집 파이프라인."""
import asyncio
import hashlib

from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader, LoadedPage, PDFLoader
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore


class IngestionPipeline:
    """문서 로드 → 청킹 → 벡터 DB 인덱싱 파이프라인."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        chunker: DocumentChunker | None = None,
        loader: DocumentLoader | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunker = chunker or DocumentChunker()
        self._loader = loader or PDFLoader()

    @staticmethod
    def _generate_chunk_id(text: str, metadata: dict) -> str:
        """결정적 청크 ID를 생성한다 (재인덱싱 시 동일 ID)."""
        source = metadata.get("source", "")
        page = metadata.get("page", "")
        key = f"{source}:{page}:{text[:200]}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def ingest_pages(self, pages: list[LoadedPage]) -> int:
        """로드된 페이지를 청킹 후 벡터 DB에 인덱싱한다. 청크 수를 반환한다."""
        if not pages:
            return 0

        documents = [(p.text, p.metadata) for p in pages]
        chunks = await asyncio.to_thread(self._chunker.chunk_batch, documents)

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [self._generate_chunk_id(c.text, c.metadata) for c in chunks]

        await self._vector_store.add_documents(texts, metadatas, ids)
        return len(chunks)

    async def ingest_file(self, file_path: str) -> int:
        """단일 파일을 인덱싱한다. 청크 수를 반환한다."""
        pages = await asyncio.to_thread(self._loader.load, file_path)
        return await self.ingest_pages(pages)

    async def ingest_directory(self, dir_path: str, glob: str = "*.pdf") -> int:
        """디렉토리 내 파일을 일괄 인덱싱한다. 청크 수를 반환한다."""
        pages = await asyncio.to_thread(self._loader.load_directory, dir_path, glob)
        return await self.ingest_pages(pages)
