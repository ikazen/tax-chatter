"""TelegramAdapter 단위 테스트."""
from unittest.mock import AsyncMock, MagicMock, patch

from src.interfaces.base import ChatAdapter
from src.interfaces.telegram_adapter import TelegramAdapter


def test_telegram_adapter_is_chat_adapter() -> None:
    """TelegramAdapter가 ChatAdapter ABC의 서브클래스이다."""
    assert issubclass(TelegramAdapter, ChatAdapter)


def test_register_handler() -> None:
    """register_handler가 핸들러를 저장한다."""
    adapter = TelegramAdapter(token="test-token")

    async def handler(user_id: str, text: str) -> str:
        return "ok"

    adapter.register_handler(handler)
    assert adapter._handler is handler


async def test_send_message_raises_when_not_started() -> None:
    """봇 시작 전 send_message는 RuntimeError를 발생시킨다."""
    adapter = TelegramAdapter(token="test-token")
    import pytest

    with pytest.raises(RuntimeError, match="봇이 시작되지 않았습니다"):
        await adapter.send_message("123", "hello")


async def test_on_message_calls_handler() -> None:
    """_on_message가 등록된 핸들러를 호출하고 reply_text로 응답한다."""
    adapter = TelegramAdapter(token="test-token")
    handler = AsyncMock(return_value="응답입니다")
    adapter.register_handler(handler)

    # mock update
    mock_update = MagicMock()
    mock_update.message.text = "안녕하세요"
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()

    await adapter._on_message(mock_update, MagicMock())

    handler.assert_called_once_with("12345", "안녕하세요")
    mock_update.message.reply_text.assert_called_once_with("응답입니다")


async def test_on_message_ignores_when_no_handler() -> None:
    """핸들러가 없으면 _on_message가 아무 작업도 하지 않는다."""
    adapter = TelegramAdapter(token="test-token")

    mock_update = MagicMock()
    mock_update.message.text = "안녕"
    # 핸들러 미등록 상태에서 예외 없이 완료
    await adapter._on_message(mock_update, MagicMock())
