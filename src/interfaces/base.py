"""채팅 인터페이스 추상 기반 클래스."""
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable

from src.models import Message


# 메시지 핸들러 타입: (user_id, text) -> 응답 문자열
MessageHandler = Callable[[str, str], Awaitable[str]]


class ChatAdapter(ABC):
    """모든 채팅 인터페이스 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    def register_handler(self, handler: MessageHandler) -> None:
        """메시지 수신 시 호출될 핸들러를 등록한다."""
        ...

    @abstractmethod
    async def send_message(self, user_id: str, text: str) -> None:
        """사용자에게 메시지를 전송한다."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """봇/서버를 시작한다."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """봇/서버를 종료한다."""
        ...
