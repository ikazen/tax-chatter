"""ChromaDB 기반 벡터 스토어 구현체."""
import asyncio

import chromadb

from config.settings import settings
from src.models import SourceChunk
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore


class ChromaStore(VectorStore):
    """ChromaDB를 사용하는 벡터 스토어."""

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or "tax_docs"
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

    async def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """청크를 임베딩하여 ChromaDB에 인덱싱한다."""
        if not chunks:
            return
        embeddings = await self._embedder.embed_batch(chunks)
        await asyncio.to_thread(
            self._collection.add,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[SourceChunk]:
        """유사 청크를 검색한다."""
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[SourceChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i, doc in enumerate(documents):
            distance = distances[i]
            score = 1.0 / (1.0 + distance)
            if score < score_threshold:
                continue
            meta = metadatas_list[i] if metadatas_list else {}
            chunks.append(
                SourceChunk(
                    chunk_id=ids[i],
                    content=doc,
                    source=meta.get("source", ""),
                    page=meta.get("page"),
                    score=score,
                )
            )

        return chunks

    async def delete_collection(self) -> None:
        """컬렉션 전체를 삭제한다."""
        await asyncio.to_thread(
            self._client.delete_collection,
            name=self._collection_name,
        )
