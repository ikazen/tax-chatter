"""IngestionPipeline 단위 테스트."""
from unittest.mock import AsyncMock, MagicMock

from src.ingestion.chunker import ChunkWithMetadata
from src.ingestion.loader import LoadedPage
from src.ingestion.pipeline import IngestionPipeline


async def test_ingest_pages_empty_input(
    mock_vector_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """빈 입력은 0을 반환한다."""
    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )

    result = await pipeline.ingest_pages([])

    assert result == 0
    mock_vector_store.add_documents.assert_not_called()


async def test_ingest_pages_processes_chunks(
    mock_vector_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """정상 입력 시 청킹 후 벡터 DB에 인덱싱한다."""
    mock_chunker = MagicMock()
    mock_chunker.chunk_batch.return_value = [
        ChunkWithMetadata(text="청크1", metadata={"source": "a.pdf", "page": 1}),
        ChunkWithMetadata(text="청크2", metadata={"source": "a.pdf", "page": 2}),
    ]

    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        chunker=mock_chunker,
    )

    pages = [
        LoadedPage(text="페이지 텍스트", metadata={"source": "a.pdf", "page": 1}),
    ]
    result = await pipeline.ingest_pages(pages)

    assert result == 2
    mock_vector_store.add_documents.assert_called_once()
    call_args = mock_vector_store.add_documents.call_args
    assert call_args[0][0] == ["청크1", "청크2"]  # chunks
    assert len(call_args[0][2]) == 2  # ids


async def test_ingest_file(
    mock_vector_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """ingest_file이 loader와 ingest_pages를 호출한다."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        LoadedPage(text="내용", metadata={"source": "test.pdf", "page": 1}),
    ]
    mock_chunker = MagicMock()
    mock_chunker.chunk_batch.return_value = [
        ChunkWithMetadata(text="내용", metadata={"source": "test.pdf", "page": 1}),
    ]

    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        chunker=mock_chunker,
        loader=mock_loader,
    )

    result = await pipeline.ingest_file("/tmp/test.pdf")

    assert result == 1
    mock_loader.load.assert_called_once()


async def test_ingest_directory(
    mock_vector_store: AsyncMock, mock_embedder: AsyncMock
) -> None:
    """ingest_directory가 loader.load_directory를 호출한다."""
    mock_loader = MagicMock()
    mock_loader.load_directory.return_value = [
        LoadedPage(text="내용1", metadata={"source": "a.pdf", "page": 1}),
        LoadedPage(text="내용2", metadata={"source": "b.pdf", "page": 1}),
    ]
    mock_chunker = MagicMock()
    mock_chunker.chunk_batch.return_value = [
        ChunkWithMetadata(text="내용1", metadata={"source": "a.pdf", "page": 1}),
        ChunkWithMetadata(text="내용2", metadata={"source": "b.pdf", "page": 1}),
    ]

    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        chunker=mock_chunker,
        loader=mock_loader,
    )

    result = await pipeline.ingest_directory("/tmp/docs")

    assert result == 2
    mock_loader.load_directory.assert_called_once()


def test_chunk_id_deterministic() -> None:
    """동일 입력에 대해 동일한 ID가 생성된다."""
    id1 = IngestionPipeline._generate_chunk_id("텍스트", {"source": "a.pdf", "page": 1})
    id2 = IngestionPipeline._generate_chunk_id("텍스트", {"source": "a.pdf", "page": 1})
    assert id1 == id2


def test_chunk_id_differs_for_different_input() -> None:
    """다른 입력에 대해 다른 ID가 생성된다."""
    id1 = IngestionPipeline._generate_chunk_id("텍스트A", {"source": "a.pdf", "page": 1})
    id2 = IngestionPipeline._generate_chunk_id("텍스트B", {"source": "a.pdf", "page": 1})
    assert id1 != id2
