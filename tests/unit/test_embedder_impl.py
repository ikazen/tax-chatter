"""SentenceTransformersEmbedder 단위 테스트."""
from unittest.mock import MagicMock, patch

import numpy as np

from src.storage.sentence_transformers import SentenceTransformersEmbedder


@patch("src.storage.sentence_transformers.SentenceTransformer")
async def test_embed_returns_list_of_floats(mock_st_class: MagicMock) -> None:
    """embed가 list[float]를 반환하는지 확인한다."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st_class.return_value = mock_model

    embedder = SentenceTransformersEmbedder(model_name="test-model")
    result = await embedder.embed("안녕하세요")

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)
    mock_model.encode.assert_called_once_with("안녕하세요")


@patch("src.storage.sentence_transformers.SentenceTransformer")
async def test_embed_batch_returns_list_of_lists(mock_st_class: MagicMock) -> None:
    """embed_batch가 list[list[float]]를 반환하는지 확인한다."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st_class.return_value = mock_model

    embedder = SentenceTransformersEmbedder(model_name="test-model")
    result = await embedder.embed_batch(["텍스트1", "텍스트2"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]


@patch("src.storage.sentence_transformers.SentenceTransformer")
@patch("src.storage.sentence_transformers.settings")
async def test_default_model_from_settings(
    mock_settings: MagicMock, mock_st_class: MagicMock
) -> None:
    """모델명을 지정하지 않으면 settings에서 가져온다."""
    mock_settings.embedder_model = "ko-sroberta"
    mock_st_class.return_value = MagicMock()

    SentenceTransformersEmbedder()

    mock_st_class.assert_called_once_with("ko-sroberta")


@patch("src.storage.sentence_transformers.SentenceTransformer")
async def test_custom_model_override(mock_st_class: MagicMock) -> None:
    """커스텀 모델명이 settings를 오버라이드한다."""
    mock_st_class.return_value = MagicMock()

    SentenceTransformersEmbedder(model_name="custom-model")

    mock_st_class.assert_called_once_with("custom-model")
