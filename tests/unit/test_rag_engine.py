"""src/rag/engine.py 단위 테스트."""
import pytest

from src.models import SourceChunk
from src.rag.engine import RAGEngine, RAG_SYSTEM_PROMPT_TEMPLATE


@pytest.fixture
def rag_engine(mock_vector_store, mock_embedder):
    return RAGEngine(vector_store=mock_vector_store, embedder=mock_embedder)


def test_build_context_includes_source_and_page(rag_engine, sample_chunks) -> None:
    context = rag_engine.build_context(sample_chunks)
    assert "소득세법.pdf" in context
    assert "p.42" in context
    assert "6%" in context


def test_build_context_empty_chunks(rag_engine) -> None:
    context = rag_engine.build_context([])
    assert context == ""


def test_build_system_prompt_contains_context(rag_engine, sample_chunks) -> None:
    prompt = rag_engine.build_system_prompt(sample_chunks)
    assert "참조 문서" in prompt
    assert "소득세법.pdf" in prompt


def test_build_system_prompt_contains_constraint(rag_engine, sample_chunks) -> None:
    """프롬프트에 hallucination 방지 constraint가 포함되어야 한다."""
    prompt = rag_engine.build_system_prompt(sample_chunks)
    assert "제공된 자료에서 확인할 수 없습니다" in prompt


@pytest.mark.asyncio
async def test_retrieve_calls_embedder_and_vector_store(
    rag_engine, mock_embedder, mock_vector_store, sample_chunks
) -> None:
    result = await rag_engine.retrieve("소득세율 알려줘")
    mock_embedder.embed.assert_called_once_with("소득세율 알려줘")
    mock_vector_store.search.assert_called_once()
    assert result == sample_chunks


@pytest.mark.asyncio
async def test_query_returns_chunks_and_prompt(rag_engine, sample_chunks) -> None:
    chunks, system_prompt = await rag_engine.query("소득세율이 어떻게 되나요?")
    assert len(chunks) == len(sample_chunks)
    assert "참조 문서" in system_prompt
