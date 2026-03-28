"""src/application/session.py 단위 테스트."""
from datetime import datetime, timedelta

import pytest

from src.application.session import SessionManager, SESSION_TTL_MINUTES
from src.models import Session


def test_get_or_create_new_session() -> None:
    manager = SessionManager()
    session = manager.get_or_create("user-1")
    assert session.user_id == "user-1"
    assert session.history == []


def test_get_or_create_returns_same_session() -> None:
    manager = SessionManager()
    s1 = manager.get_or_create("user-1")
    s2 = manager.get_or_create("user-1")
    assert s1.session_id == s2.session_id


def test_append_message() -> None:
    manager = SessionManager()
    manager.append_message("user-1", "user", "안녕하세요")
    manager.append_message("user-1", "assistant", "안녕하세요!")
    history = manager.get_history("user-1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_clear_session() -> None:
    manager = SessionManager()
    manager.append_message("user-1", "user", "테스트")
    manager.clear("user-1")
    # clear 후 새 세션이 생성되어야 함
    session = manager.get_or_create("user-1")
    assert session.history == []


def test_expired_session_creates_new(monkeypatch) -> None:
    manager = SessionManager()
    session = manager.get_or_create("user-1")
    manager.append_message("user-1", "user", "오래된 메시지")

    # 마지막 메시지 타임스탬프를 TTL 이전으로 조작
    expired_time = datetime.now() - timedelta(minutes=SESSION_TTL_MINUTES + 1)
    session.history[-1].timestamp = expired_time

    new_session = manager.get_or_create("user-1")
    assert new_session.session_id != session.session_id
    assert new_session.history == []


def test_clear_nonexistent_session_does_not_raise() -> None:
    manager = SessionManager()
    manager.clear("nonexistent-user")  # 예외 없이 통과해야 함
