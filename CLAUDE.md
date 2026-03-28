# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요
세무사(3인 이하) 전용 RAG 기반 LLM 서비스.
세법 / 판례 문서를 벡터 DB에 인덱싱하고, 사용자 질의에 대해 검색된 청크 내에서만 답변한다.

## 명령어

```bash
# 의존성 설치
uv sync --all-extras

# 테스트 (unit + integration, mock만 사용, API 호출 없음)
uv run pytest tests/unit tests/integration

# 단일 테스트 파일 실행
uv run pytest tests/unit/test_router.py

# 단일 테스트 함수 실행
uv run pytest tests/unit/test_router.py::test_tax_query_detection -v

# RAG 품질 평가 (실제 API 호출)
uv run pytest tests/eval -m eval

# 커버리지
uv run pytest --cov=src --cov-report=term-missing

# 린트
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# 타입 체크
uv run mypy src/
```

## 아키텍처

### ABC 기반 추상화 레이어
모든 외부 의존성은 ABC 뒤에 숨긴다. application/rag 레이어에서 구현체를 직접 import하면 안 된다.

| ABC | 위치 | 구현체 |
|-----|------|--------|
| `LLMProvider` | `src/llm/base.py` | `src/llm/gemini.py` |
| `VectorStore` | `src/storage/vector_store.py` | `src/storage/chroma_store.py`, `qdrant_store.py` |
| `Embedder` | `src/storage/embedder.py` | (미구현) |
| `ChatAdapter` | `src/interfaces/base.py` | `telegram_adapter.py`, `discord_adapter.py` |

```python
# 올바른 방법: ABC를 통해 접근
from src.llm.base import LLMProvider
from src.storage.vector_store import VectorStore

# 금지: application layer에서 구현체 직접 import
from src.llm.gemini import GeminiProvider  # 금지
import chromadb  # 금지: storage layer 밖에서 직접 import
```

### 데이터 흐름
1. `src/ingestion/` — 문서 수집 → 청킹 → 벡터 DB 인덱싱
2. `src/application/router.py` — 사용자 질의를 세금 관련 / 일반 대화로 분류
3. `src/rag/engine.py` — retrieve → context 조립 → LLM 호출 → `RAGResponse` 반환
4. `src/interfaces/` — Telegram/Discord를 통해 사용자에게 응답

### 핵심 모델 (`src/models.py`)
- `SourceChunk`: 검색된 문서 청크 메타데이터 (source, page, content, score)
- `RAGResponse`: 답변 + source_chunks + grounded 플래그
- `Session`: 사용자별 대화 히스토리 (SessionManager가 in-memory 관리, TTL 60분)

### 설정 (`config/settings.py`)
환경변수는 반드시 `settings` 객체를 통해 접근한다. `os.getenv()` 직접 사용 금지.

```python
from config.settings import settings
api_key = settings.gemini_api_key
```

구현체 교체는 `.env` 또는 환경변수로:
```
LLM_BACKEND=gemini            # gemini | ollama | openai
VECTOR_STORE_BACKEND=chroma   # chroma | qdrant
EMBEDDER_BACKEND=sentence_transformers  # sentence_transformers | openai | local
CHAT_INTERFACE=telegram       # telegram | discord | cli
```

## 핵심 규칙

- **비동기(async/await) 기본**: LLM/DB I/O 모두 비동기로 작성
- **타입 힌트**: 모든 함수에 명시
- **docstring**: 한국어로 작성
- **RAG 출처 필수**: `RAGResponse`에 항상 `source_chunks`를 포함하고, LLM 프롬프트에 "제공된 문서 외의 내용으로 답하지 마시오" constraint 명시
- **테스트에서 LLM mock**: `@pytest.mark.eval` 마커가 붙은 테스트만 실제 API 호출. 기본 pytest 실행 시 eval 테스트는 자동 skip
- **새 파일 생성 전** 해당 레이어의 ABC가 존재하는지 확인할 것

## 테스트

- `asyncio_mode = "auto"` — 테스트에서 별도 `@pytest.mark.asyncio` 불필요
- `tests/conftest.py`에 공통 fixture: `mock_llm`, `mock_vector_store`, `mock_embedder`, `sample_chunks`
- Ruff 규칙: `E`, `F`, `I`, `UP` / line-length 100
- MyPy: strict 모드, `ignore_missing_imports = true`
