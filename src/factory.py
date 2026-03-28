"""의존성 팩토리 — settings 값으로 구현체 인스턴스를 생성한다.

새 구현체 추가 시 이 파일만 수정하면 된다.
"""
from config.settings import settings
from src.interfaces.base import ChatAdapter
from src.llm.llm_provider import LLMProvider
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore


def create_llm() -> LLMProvider:
    """settings.llm_backend에 따라 LLM 구현체를 반환한다."""
    if settings.llm_backend == "gemini":
        from src.llm.gemini import GeminiProvider

        return GeminiProvider()
    if settings.llm_backend == "ollama":
        from src.llm.local import OllamaProvider

        return OllamaProvider()
    raise ValueError(f"지원하지 않는 LLM 백엔드: {settings.llm_backend}")


def create_embedder() -> Embedder:
    """settings.embedder_backend에 따라 Embedder 구현체를 반환한다."""
    if settings.embedder_backend == "sentence_transformers":
        from src.storage.sentence_transformers import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder()
    raise ValueError(f"지원하지 않는 Embedder 백엔드: {settings.embedder_backend}")


def create_vector_store(embedder: Embedder) -> VectorStore:
    """settings.vector_store_backend에 따라 VectorStore 구현체를 반환한다."""
    if settings.vector_store_backend == "chroma":
        from src.storage.chroma_store import ChromaStore

        return ChromaStore(embedder=embedder)
    raise ValueError(
        f"지원하지 않는 VectorStore 백엔드: {settings.vector_store_backend}"
    )


def create_chat_adapter() -> ChatAdapter:
    """settings.chat_interface에 따라 ChatAdapter 구현체를 반환한다."""
    if settings.chat_interface == "telegram":
        from src.interfaces.telegram_adapter import TelegramAdapter

        return TelegramAdapter()
    raise ValueError(
        f"지원하지 않는 채팅 인터페이스: {settings.chat_interface}"
    )
