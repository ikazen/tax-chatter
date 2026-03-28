"""RAG 파이프라인 통합 테스트 (mock LLM + mock VectorStore)."""
import pytest

from src.rag.engine import RAGEngine
from src.application.router import is_tax_query
from src.application.session import SessionManager


@pytest.mark.asyncio
async def test_full_rag_query_flow(
    mock_vector_store, mock_embedder, mock_llm, sample_chunks
) -> None:
    """질의 → 검색 → 프롬프트 조립 → LLM 응답 전체 흐름 검증."""
    rag_engine = RAGEngine(vector_store=mock_vector_store, embedder=mock_embedder)
    session_manager = SessionManager()
    user_id = "test-user"
    question = "소득세율이 어떻게 되나요?"

    # 라우팅
    assert is_tax_query(question) is True

    # RAG 검색 + 프롬프트 조립
    chunks, system_prompt = await rag_engine.query(question)
    assert len(chunks) > 0
    assert "참조 문서" in system_prompt

    # 세션에 질문 추가
    session_manager.append_message(user_id, "user", question)
    history = session_manager.get_history(user_id)

    # LLM 호출 (mock)
    answer = await mock_llm.generate(system_prompt, history)
    assert answer  # 빈 문자열이 아님

    # 세션에 답변 추가
    session_manager.append_message(user_id, "assistant", answer)
    assert len(session_manager.get_history(user_id)) == 2


@pytest.mark.asyncio
async def test_rag_pipeline_with_no_results(
    mock_vector_store, mock_embedder, mock_llm
) -> None:
    """검색 결과가 없을 때 빈 컨텍스트로 프롬프트가 구성되는지 확인."""
    mock_vector_store.search.return_value = []
    rag_engine = RAGEngine(vector_store=mock_vector_store, embedder=mock_embedder)

    chunks, system_prompt = await rag_engine.query("존재하지 않는 세법 조항")
    assert chunks == []
    # 컨텍스트가 비어있어도 constraint 프롬프트는 유지되어야 함
    assert "제공된 자료에서 확인할 수 없습니다" in system_prompt
