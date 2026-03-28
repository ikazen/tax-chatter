"""애플리케이션 진입점 — 모든 의존성을 조립하고 봇을 실행한다."""
import asyncio
import logging

from config.settings import settings
from src.application.prompt_builder import build_general_prompt
from src.application.router import is_tax_query
from src.application.session import SessionManager
from src.factory import create_chat_adapter, create_embedder, create_llm, create_vector_store
from src.models import Message
from src.rag.engine import RAGEngine

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


async def main() -> None:
    """의존성 조립 후 봇을 시작한다."""
    # 의존성 생성
    llm = create_llm()
    embedder = create_embedder()
    vector_store = create_vector_store(embedder)
    chat_adapter = create_chat_adapter()

    rag_engine = RAGEngine(vector_store=vector_store, embedder=embedder)
    session_manager = SessionManager()

    async def handle_message(user_id: str, text: str) -> str:
        """사용자 메시지를 처리하여 응답을 반환한다."""
        session_manager.append_message(user_id, "user", text)
        history = session_manager.get_history(user_id)

        if is_tax_query(text):
            chunks, system_prompt = await rag_engine.query(text)
        else:
            system_prompt = build_general_prompt()

        answer = await llm.generate(system_prompt, history)
        session_manager.append_message(user_id, "assistant", answer)
        return answer

    chat_adapter.register_handler(handle_message)

    logger.info("봇을 시작합니다...")
    await chat_adapter.start()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await chat_adapter.stop()
        logger.info("봇이 종료되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
