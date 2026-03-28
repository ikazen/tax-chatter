"""DocumentChunker 단위 테스트."""
from src.ingestion.chunker import ChunkWithMetadata, DocumentChunker


def test_long_text_splits_into_multiple_chunks() -> None:
    """긴 텍스트가 복수 청크로 분할된다."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "가" * 500
    result = chunker.chunk(text, {"source": "test.pdf"})

    assert len(result) > 1
    assert all(isinstance(c, ChunkWithMetadata) for c in result)


def test_short_text_single_chunk() -> None:
    """짧은 텍스트는 단일 청크로 반환된다."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    text = "짧은 텍스트입니다."
    result = chunker.chunk(text, {"source": "test.pdf"})

    assert len(result) == 1
    assert result[0].text == text


def test_empty_text_returns_empty_list() -> None:
    """빈 텍스트는 빈 리스트를 반환한다."""
    chunker = DocumentChunker()
    assert chunker.chunk("", {"source": "test.pdf"}) == []
    assert chunker.chunk("   ", {"source": "test.pdf"}) == []


def test_metadata_preserved_in_chunks() -> None:
    """메타데이터가 각 청크에 보존된다."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "가" * 500
    metadata = {"source": "소득세법.pdf", "page": 42}
    result = chunker.chunk(text, metadata)

    for chunk in result:
        assert chunk.metadata["source"] == "소득세법.pdf"
        assert chunk.metadata["page"] == 42


def test_metadata_isolation() -> None:
    """청크별 메타데이터가 독립적이다 (shallow copy)."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "가" * 500
    metadata = {"source": "test.pdf"}
    result = chunker.chunk(text, metadata)

    result[0].metadata["extra"] = "modified"
    assert "extra" not in result[1].metadata
    assert "extra" not in metadata


def test_chunk_batch() -> None:
    """chunk_batch가 복수 문서를 일괄 처리한다."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    documents = [
        ("텍스트 A", {"source": "a.pdf"}),
        ("텍스트 B", {"source": "b.pdf"}),
    ]
    result = chunker.chunk_batch(documents)

    assert len(result) == 2
    assert result[0].metadata["source"] == "a.pdf"
    assert result[1].metadata["source"] == "b.pdf"
