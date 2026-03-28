"""scripts/ingest.py CLI 단위 테스트."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@patch("scripts.ingest.IngestionPipeline")
@patch("scripts.ingest.ChromaStore")
@patch("scripts.ingest.SentenceTransformersEmbedder")
@patch("scripts.ingest.PDFLoader")
async def test_ingest_file(
    mock_loader_cls: MagicMock,
    mock_embedder_cls: MagicMock,
    mock_store_cls: MagicMock,
    mock_pipeline_cls: MagicMock,
    tmp_path,
) -> None:
    """단일 파일 경로를 인자로 받으면 ingest_file을 호출한다."""
    # tmp_path에 파일 생성
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"dummy")

    mock_pipeline = MagicMock()
    mock_pipeline.ingest_file = AsyncMock(return_value=5)
    mock_pipeline_cls.return_value = mock_pipeline

    from scripts.ingest import main

    await main(str(test_file))

    mock_pipeline.ingest_file.assert_called_once_with(str(test_file))


@patch("scripts.ingest.IngestionPipeline")
@patch("scripts.ingest.ChromaStore")
@patch("scripts.ingest.SentenceTransformersEmbedder")
@patch("scripts.ingest.PDFLoader")
async def test_ingest_directory(
    mock_loader_cls: MagicMock,
    mock_embedder_cls: MagicMock,
    mock_store_cls: MagicMock,
    mock_pipeline_cls: MagicMock,
    tmp_path,
) -> None:
    """디렉토리 경로를 인자로 받으면 ingest_directory를 호출한다."""
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_directory = AsyncMock(return_value=10)
    mock_pipeline_cls.return_value = mock_pipeline

    from scripts.ingest import main

    await main(str(tmp_path))

    mock_pipeline.ingest_directory.assert_called_once_with(str(tmp_path))


@patch("scripts.ingest.IngestionPipeline")
@patch("scripts.ingest.ChromaStore")
@patch("scripts.ingest.SentenceTransformersEmbedder")
@patch("scripts.ingest.PDFLoader")
async def test_ingest_invalid_path_exits(
    mock_loader_cls: MagicMock,
    mock_embedder_cls: MagicMock,
    mock_store_cls: MagicMock,
    mock_pipeline_cls: MagicMock,
) -> None:
    """존재하지 않는 경로는 SystemExit을 발생시킨다."""
    from scripts.ingest import main

    with pytest.raises(SystemExit):
        await main("/nonexistent/path/to/file.pdf")
