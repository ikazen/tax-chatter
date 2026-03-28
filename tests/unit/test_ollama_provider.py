"""OllamaProvider 단위 테스트."""
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.base import LLMProvider
from src.llm.local import OllamaProvider
from src.models import Message


def test_ollama_provider_is_llm_provider() -> None:
    """OllamaProvider가 LLMProvider ABC의 서브클래스이다."""
    assert issubclass(OllamaProvider, LLMProvider)


def test_build_messages() -> None:
    """_build_messages가 system + user/assistant 메시지를 올바르게 변환한다."""
    provider = OllamaProvider(base_url="http://localhost:11434", model="test")
    messages = [
        Message(role="user", content="안녕하세요"),
        Message(role="assistant", content="네, 안녕하세요"),
        Message(role="user", content="세율 알려주세요"),
    ]
    result = provider._build_messages("시스템 프롬프트", messages)

    assert result[0] == {"role": "system", "content": "시스템 프롬프트"}
    assert result[1] == {"role": "user", "content": "안녕하세요"}
    assert result[2] == {"role": "assistant", "content": "네, 안녕하세요"}
    assert result[3] == {"role": "user", "content": "세율 알려주세요"}
    assert len(result) == 4


async def test_generate_calls_ollama_api() -> None:
    """generate가 /api/chat 엔드포인트를 호출한다."""
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"role": "assistant", "content": "답변입니다"}
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        messages = [Message(role="user", content="질문")]
        result = await provider.generate("시스템", messages)

    assert result == "답변입니다"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["model"] == "llama3"
    assert call_kwargs[1]["json"]["stream"] is False
