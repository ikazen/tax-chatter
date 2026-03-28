"""ChromaStore 단위 테스트."""
from unittest.mock import AsyncMock, MagicMock, patch

from src.storage.chroma_store import ChromaStore
from src.storage.embedder import Embedder


def _make_mock_embedder() -> Embedder:
    """mock 임베더를 생성한다."""
    mock = AsyncMock(spec=Embedder)
    mock.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return mock


@patch("src.storage.chroma_store.chromadb")
async def test_add_documents_calls_embed_and_add(mock_chromadb: MagicMock) -> None:
    """add_documents가 embed_batch 후 collection.add를 호출한다."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    await store.add_documents(
        chunks=["텍스트1", "텍스트2"],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
        ids=["id1", "id2"],
    )

    embedder.embed_batch.assert_called_once_with(["텍스트1", "텍스트2"])
    mock_collection.add.assert_called_once_with(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["텍스트1", "텍스트2"],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
        ids=["id1", "id2"],
    )


@patch("src.storage.chroma_store.chromadb")
async def test_add_documents_empty_chunks(mock_chromadb: MagicMock) -> None:
    """빈 청크 리스트는 아무 작업도 하지 않는다."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    await store.add_documents(chunks=[], metadatas=[], ids=[])

    embedder.embed_batch.assert_not_called()


@patch("src.storage.chroma_store.chromadb")
async def test_search_converts_distance_to_score(mock_chromadb: MagicMock) -> None:
    """search가 distance를 1/(1+d)로 score 변환한다."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["문서1", "문서2"]],
        "metadatas": [[{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 2}]],
        "distances": [[0.1, 0.5]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    results = await store.search(query_embedding=[0.1, 0.2], top_k=5, score_threshold=0.0)

    assert len(results) == 2
    # 1/(1+0.1) ≈ 0.909
    assert abs(results[0].score - 1.0 / 1.1) < 1e-6
    # 1/(1+0.5) ≈ 0.667
    assert abs(results[1].score - 1.0 / 1.5) < 1e-6
    assert results[0].source == "a.pdf"
    assert results[0].page == 1


@patch("src.storage.chroma_store.chromadb")
async def test_search_filters_by_threshold(mock_chromadb: MagicMock) -> None:
    """score_threshold 이하의 결과는 필터링된다."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["문서1", "문서2"]],
        "metadatas": [[{"source": "a.pdf", "page": 1}, {"source": "b.pdf"}]],
        "distances": [[0.1, 10.0]],  # 10.0 → score 0.09 < threshold
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    results = await store.search(query_embedding=[0.1], top_k=5, score_threshold=0.5)

    assert len(results) == 1
    assert results[0].chunk_id == "id1"


@patch("src.storage.chroma_store.chromadb")
async def test_search_empty_results(mock_chromadb: MagicMock) -> None:
    """빈 결과를 올바르게 처리한다."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    results = await store.search(query_embedding=[0.1], top_k=5, score_threshold=0.5)

    assert results == []


@patch("src.storage.chroma_store.chromadb")
async def test_search_page_metadata_none(mock_chromadb: MagicMock) -> None:
    """page 메타데이터가 없는 경우 None으로 처리한다."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["문서1"]],
        "metadatas": [[{"source": "a.pdf"}]],  # page 없음
        "distances": [[0.1]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test")

    results = await store.search(query_embedding=[0.1], top_k=5, score_threshold=0.0)

    assert results[0].page is None


@patch("src.storage.chroma_store.chromadb")
async def test_delete_collection(mock_chromadb: MagicMock) -> None:
    """delete_collection이 client.delete_collection을 호출한다."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    embedder = _make_mock_embedder()
    store = ChromaStore(embedder=embedder, persist_dir="/tmp/test", collection_name="test_col")

    await store.delete_collection()

    mock_client.delete_collection.assert_called_once_with(name="test_col")
