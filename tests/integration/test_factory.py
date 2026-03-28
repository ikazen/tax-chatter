"""팩토리 함수 통합 테스트 — 설정값에 따라 올바른 구현체 반환 검증."""
from unittest.mock import patch

import pytest

from src.factory import create_chat_adapter, create_embedder, create_llm, create_vector_store
from src.interfaces.base import ChatAdapter
from src.llm.base import LLMProvider
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore


@patch("src.factory.settings")
@patch("src.llm.gemini.genai")
def test_create_llm_gemini(mock_genai, mock_settings) -> None:
    """llm_backend=gemini이면 GeminiProvider를 반환한다."""
    mock_settings.llm_backend = "gemini"
    mock_settings.gemini_api_key = "test-key"
    mock_settings.gemini_model = "gemini-2.0-flash"

    llm = create_llm()

    assert isinstance(llm, LLMProvider)
    from src.llm.gemini import GeminiProvider

    assert isinstance(llm, GeminiProvider)


@patch("src.factory.settings")
def test_create_llm_ollama(mock_settings) -> None:
    """llm_backend=ollama이면 OllamaProvider를 반환한다."""
    mock_settings.llm_backend = "ollama"

    llm = create_llm()

    assert isinstance(llm, LLMProvider)
    from src.llm.local import OllamaProvider

    assert isinstance(llm, OllamaProvider)


@patch("src.factory.settings")
def test_create_llm_unsupported_raises(mock_settings) -> None:
    """지원하지 않는 llm_backend는 ValueError를 발생시킨다."""
    mock_settings.llm_backend = "unsupported"

    with pytest.raises(ValueError, match="지원하지 않는 LLM"):
        create_llm()


@patch("src.factory.settings")
@patch("src.storage.embedder_impl.SentenceTransformer")
def test_create_embedder_sentence_transformers(
    mock_st, mock_settings
) -> None:
    """embedder_backend=sentence_transformers이면 해당 구현체를 반환한다."""
    mock_settings.embedder_backend = "sentence_transformers"
    mock_settings.embedder_model = "test-model"

    embedder = create_embedder()

    assert isinstance(embedder, Embedder)


@patch("src.factory.settings")
def test_create_embedder_unsupported_raises(mock_settings) -> None:
    """지원하지 않는 embedder_backend는 ValueError를 발생시킨다."""
    mock_settings.embedder_backend = "unsupported"

    with pytest.raises(ValueError, match="지원하지 않는 Embedder"):
        create_embedder()


@patch("src.factory.settings")
@patch("src.storage.chroma_store.chromadb")
def test_create_vector_store_chroma(mock_chromadb, mock_settings) -> None:
    """vector_store_backend=chroma이면 ChromaStore를 반환한다."""
    mock_settings.vector_store_backend = "chroma"
    mock_settings.chroma_persist_dir = "/tmp/test_chroma"
    mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = (
        None
    )

    from unittest.mock import AsyncMock

    mock_embedder = AsyncMock(spec=Embedder)
    store = create_vector_store(mock_embedder)

    assert isinstance(store, VectorStore)


@patch("src.factory.settings")
def test_create_vector_store_unsupported_raises(mock_settings) -> None:
    """지원하지 않는 vector_store_backend는 ValueError를 발생시킨다."""
    mock_settings.vector_store_backend = "unsupported"

    from unittest.mock import AsyncMock

    mock_embedder = AsyncMock(spec=Embedder)

    with pytest.raises(ValueError, match="지원하지 않는 VectorStore"):
        create_vector_store(mock_embedder)


@patch("src.factory.settings")
def test_create_chat_adapter_telegram(mock_settings) -> None:
    """chat_interface=telegram이면 TelegramAdapter를 반환한다."""
    mock_settings.chat_interface = "telegram"
    mock_settings.telegram_bot_token = "test-token"

    adapter = create_chat_adapter()

    assert isinstance(adapter, ChatAdapter)
    from src.interfaces.telegram_adapter import TelegramAdapter

    assert isinstance(adapter, TelegramAdapter)


@patch("src.factory.settings")
def test_create_chat_adapter_unsupported_raises(mock_settings) -> None:
    """지원하지 않는 chat_interface는 ValueError를 발생시킨다."""
    mock_settings.chat_interface = "unsupported"

    with pytest.raises(ValueError, match="지원하지 않는 채팅"):
        create_chat_adapter()
