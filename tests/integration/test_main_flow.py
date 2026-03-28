"""main 흐름 end-to-end 통합 테스트.

사용자 메시지 → 라우터 분기 → RAG/일반 → LLM → 응답 전체 흐름.
모든 외부 의존성은 mock으로 대체한다.
"""
from unittest.mock import AsyncMock

from src.application.prompt_builder import build_general_prompt
from src.application.router import is_tax_query
from src.application.session import SessionManager
from src.rag.engine import RAGEngine


async def test_tax_query_flow(
    mock_vector_store: AsyncMock,
    mock_embedder: AsyncMock,
    mock_llm: AsyncMock,
    sample_chunks,
) -> None:
    """세금 질의 → RAG 검색 → LLM 응답 전체 흐름."""
    rag_engine = RAGEngine(vector_store=mock_vector_store, embedder=mock_embedder)
    session_manager = SessionManager()
    user_id = "user-1"
    text = "소득세율이 어떻게 되나요?"

    # 라우터 분기
    assert is_tax_query(text) is True

    # 세션에 메시지 추가
    session_manager.append_message(user_id, "user", text)
    history = session_manager.get_history(user_id)

    # RAG 경로
    chunks, system_prompt = await rag_engine.query(text)
    assert len(chunks) > 0
    assert "참조 문서" in system_prompt

    # LLM 호출
    answer = await mock_llm.generate(system_prompt, history)
    assert answer

    # 세션에 응답 추가
    session_manager.append_message(user_id, "assistant", answer)
    assert len(session_manager.get_history(user_id)) == 2


async def test_general_query_flow(
    mock_llm: AsyncMock,
) -> None:
    """일반 대화 → 일반 프롬프트 → LLM 응답 흐름."""
    session_manager = SessionManager()
    user_id = "user-2"
    text = "오늘 날씨가 어때요?"

    # 라우터 분기
    assert is_tax_query(text) is False

    # 세션에 메시지 추가
    session_manager.append_message(user_id, "user", text)
    history = session_manager.get_history(user_id)

    # 일반 대화 경로
    system_prompt = build_general_prompt()
    assert "세법" in system_prompt

    # LLM 호출
    answer = await mock_llm.generate(system_prompt, history)
    assert answer

    # 세션에 응답 추가
    session_manager.append_message(user_id, "assistant", answer)
    assert len(session_manager.get_history(user_id)) == 2


async def test_handle_message_function(
    mock_vector_store: AsyncMock,
    mock_embedder: AsyncMock,
    mock_llm: AsyncMock,
    sample_chunks,
) -> None:
    """main.py의 handle_message와 동일한 로직을 테스트한다."""
    rag_engine = RAGEngine(vector_store=mock_vector_store, embedder=mock_embedder)
    session_manager = SessionManager()

    async def handle_message(user_id: str, text: str) -> str:
        """main.py의 handle_message 로직 재현."""
        session_manager.append_message(user_id, "user", text)
        history = session_manager.get_history(user_id)

        if is_tax_query(text):
            _chunks, system_prompt = await rag_engine.query(text)
        else:
            system_prompt = build_general_prompt()

        answer = await mock_llm.generate(system_prompt, history)
        session_manager.append_message(user_id, "assistant", answer)
        return answer

    # 세금 질의
    answer1 = await handle_message("user-A", "소득세 신고 방법")
    assert answer1
    assert len(session_manager.get_history("user-A")) == 2

    # 일반 대화
    answer2 = await handle_message("user-B", "안녕하세요")
    assert answer2
    assert len(session_manager.get_history("user-B")) == 2

    # 동일 사용자 연속 질의 — 세션 히스토리 누적
    await handle_message("user-A", "부가세도 알려주세요")
    assert len(session_manager.get_history("user-A")) == 4
