"""RAG 파이프라인 오케스트레이터."""
from config.settings import settings
from src.models import RAGResponse, SourceChunk
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore

RAG_SYSTEM_PROMPT_TEMPLATE = """당신은 세무 전문가 보조 AI입니다.
반드시 아래 [참조 문서]에 포함된 내용만을 근거로 답변하십시오.
[참조 문서]에 없는 내용은 "제공된 자료에서 확인할 수 없습니다"라고 답변하십시오.
답변 마지막에 근거가 된 문서명과 페이지를 반드시 명시하십시오.

[참조 문서]
{context}
"""


class RAGEngine:
    """검색 → 컨텍스트 주입 → LLM 응답 파이프라인."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    async def retrieve(self, query: str) -> list[SourceChunk]:
        """질의에 관련된 청크를 검색한다."""
        query_embedding = await self._embedder.embed(query)
        return await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
        )

    def build_context(self, chunks: list[SourceChunk]) -> str:
        """청크 목록을 프롬프트 컨텍스트 문자열로 변환한다."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" p.{chunk.page}" if chunk.page else ""
            parts.append(f"[{i}] {chunk.source}{page_info}\n{chunk.content}")
        return "\n\n".join(parts)

    def build_system_prompt(self, chunks: list[SourceChunk]) -> str:
        """RAG 시스템 프롬프트를 조립한다."""
        context = self.build_context(chunks)
        return RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context)

    async def query(self, question: str) -> tuple[list[SourceChunk], str]:
        """검색 수행 후 (청크, 시스템 프롬프트) 튜플을 반환한다.

        실제 LLM 호출은 Application layer에서 수행한다.
        RAGEngine은 검색과 프롬프트 조립만 담당한다.
        """
        chunks = await self.retrieve(question)
        system_prompt = self.build_system_prompt(chunks)
        return chunks, system_prompt
