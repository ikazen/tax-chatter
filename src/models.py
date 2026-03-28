"""도메인 모델 — 레이어 간 데이터 전달에 사용되는 공통 타입."""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SourceChunk:
    """검색된 문서 청크 및 출처 정보."""
    chunk_id: str
    content: str
    source: str          # 파일명 또는 URL
    page: int | None
    score: float         # 유사도 점수


@dataclass
class RAGResponse:
    """RAG 엔진의 최종 응답."""
    answer: str
    source_chunks: list[SourceChunk]
    is_grounded: bool    # 출처 내에서 답했는지 여부


@dataclass
class Message:
    """사용자/어시스턴트 메시지."""
    role: str            # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """대화 세션."""
    session_id: str
    user_id: str
    history: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
