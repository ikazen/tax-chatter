"""ChatAdapter 구현체 계약 검증 테스트.

새 어댑터를 추가할 때 이 테스트를 통과해야 한다.
실제 봇 토큰 없이 동작 검증만 수행한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.interfaces.base import ChatAdapter, MessageHandler


class MockChatAdapter(ChatAdapter):
    """테스트용 최소 구현체."""

    def __init__(self) -> None:
        self._handler: MessageHandler | None = None
        self.sent_messages: list[tuple[str, str]] = []
        self.started = False
        self.stopped = False

    def register_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def send_message(self, user_id: str, text: str) -> None:
        self.sent_messages.append((user_id, text))

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def simulate_incoming(self, user_id: str, text: str) -> str:
        """인바운드 메시지 시뮬레이션 — 테스트 전용."""
        assert self._handler is not None, "핸들러가 등록되지 않음"
        return await self._handler(user_id, text)


@pytest.fixture
def adapter() -> MockChatAdapter:
    return MockChatAdapter()


@pytest.mark.asyncio
async def test_adapter_lifecycle(adapter: MockChatAdapter) -> None:
    await adapter.start()
    assert adapter.started

    await adapter.stop()
    assert adapter.stopped


@pytest.mark.asyncio
async def test_adapter_handler_registration(adapter: MockChatAdapter) -> None:
    received: list[tuple[str, str]] = []

    async def handler(user_id: str, text: str) -> str:
        received.append((user_id, text))
        return "응답"

    adapter.register_handler(handler)
    result = await adapter.simulate_incoming("user-1", "안녕")

    assert received == [("user-1", "안녕")]
    assert result == "응답"


@pytest.mark.asyncio
async def test_adapter_send_message(adapter: MockChatAdapter) -> None:
    await adapter.send_message("user-1", "테스트 메시지")
    assert ("user-1", "테스트 메시지") in adapter.sent_messages


@pytest.mark.asyncio
async def test_adapter_simulate_roundtrip(adapter: MockChatAdapter) -> None:
    """인바운드 메시지 → 핸들러 → send_message 왕복 시뮬레이션."""

    async def echo_handler(user_id: str, text: str) -> str:
        response = f"에코: {text}"
        await adapter.send_message(user_id, response)
        return response

    adapter.register_handler(echo_handler)
    result = await adapter.simulate_incoming("user-2", "세금 질문")

    assert result == "에코: 세금 질문"
    assert ("user-2", "에코: 세금 질문") in adapter.sent_messages
