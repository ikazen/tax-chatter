"""대화 세션 관리 (in-memory, 추후 Redis 등으로 교체 가능)."""
import uuid
from datetime import datetime, timedelta

from src.models import Message, Session

SESSION_TTL_MINUTES = 60


class SessionManager:
    """사용자별 대화 히스토리를 관리한다."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, user_id: str) -> Session:
        """세션을 가져오거나 새로 생성한다. TTL 초과 시 새 세션을 반환한다."""
        session = self._sessions.get(user_id)
        if session and self._is_expired(session):
            del self._sessions[user_id]
            session = None

        if session is None:
            session = Session(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
            )
            self._sessions[user_id] = session

        return session

    def append_message(self, user_id: str, role: str, content: str) -> None:
        """세션에 메시지를 추가한다."""
        session = self.get_or_create(user_id)
        session.history.append(Message(role=role, content=content))

    def get_history(self, user_id: str) -> list[Message]:
        """대화 히스토리를 반환한다."""
        return self.get_or_create(user_id).history

    def clear(self, user_id: str) -> None:
        """세션을 초기화한다."""
        self._sessions.pop(user_id, None)

    @staticmethod
    def _is_expired(session: Session) -> bool:
        cutoff = datetime.now() - timedelta(minutes=SESSION_TTL_MINUTES)
        last_activity = (
            session.history[-1].timestamp if session.history else session.created_at
        )
        return last_activity < cutoff
