"""공통 pytest fixture."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.chunker import ChunkWithMetadata
from src.ingestion.loader import LoadedPage
from src.llm.base import LLMProvider
from src.models import SourceChunk
from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore


@pytest.fixture
def sample_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="chunk-001",
            content="소득세법 제55조에 따라 종합소득 과세표준이 1,400만원 이하인 경우 세율은 6%이다.",
            source="소득세법.pdf",
            page=42,
            score=0.91,
        ),
        SourceChunk(
            chunk_id="chunk-002",
            content="소득세법 제55조 제2항에 따라 4,600만원 초과 8,800만원 이하의 경우 세율은 24%이다.",
            source="소득세법.pdf",
            page=42,
            score=0.87,
        ),
    ]


@pytest.fixture
def mock_llm() -> LLMProvider:
    """실제 API를 호출하지 않는 mock LLM provider."""
    mock = AsyncMock(spec=LLMProvider)
    mock.generate.return_value = "소득세율은 과세표준에 따라 6%~45%입니다. [소득세법.pdf p.42]"
    mock.stream.return_value = AsyncMock()
    return mock


@pytest.fixture
def mock_vector_store(sample_chunks) -> VectorStore:
    """mock 벡터 스토어 — 고정된 청크를 반환한다."""
    mock = AsyncMock(spec=VectorStore)
    mock.search.return_value = sample_chunks
    mock.add_documents.return_value = None
    return mock


@pytest.fixture
def mock_embedder() -> Embedder:
    """mock 임베더 — 고정된 벡터를 반환한다."""
    mock = AsyncMock(spec=Embedder)
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.return_value = [[0.1] * 768]
    return mock


@pytest.fixture
def sample_loaded_pages() -> list[LoadedPage]:
    """테스트용 로드된 페이지."""
    return [
        LoadedPage(text="소득세법 제55조 내용", metadata={"source": "소득세법.pdf", "page": 1}),
        LoadedPage(text="법인세법 제13조 내용", metadata={"source": "법인세법.pdf", "page": 5}),
    ]


@pytest.fixture
def mock_chunker() -> MagicMock:
    """mock 청커."""
    mock = MagicMock()
    mock.chunk_batch.return_value = [
        ChunkWithMetadata(text="청크1", metadata={"source": "test.pdf", "page": 1}),
    ]
    return mock


@pytest.fixture
def mock_loader() -> MagicMock:
    """mock PDF 로더."""
    mock = MagicMock()
    mock.load.return_value = [
        LoadedPage(text="페이지 내용", metadata={"source": "test.pdf", "page": 1}),
    ]
    mock.load_directory.return_value = [
        LoadedPage(text="페이지 내용", metadata={"source": "test.pdf", "page": 1}),
    ]
    return mock
