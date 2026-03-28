# CLAUDE.md

## 프로젝트 개요
세무사(3인 이하) 전용 RAG 기반 LLM 서비스.
세법 / 판례 문서를 벡터 DB에 인덱싱하고, 사용자 질의에 대해 검색된 청크 내에서만 답변한다.

## 기술 스택
- **Python**: 3.11+, 패키지 매니저 `uv`
- **LLM**: `google-generativeai` (Gemini 2.0 Flash, 기본값) — `LLMProvider` ABC로 추상화
- **Vector DB**: `chromadb` (기본) → `qdrant-client` (옵션) — `VectorStore` ABC로 추상화
- **Embedding**: `sentence-transformers` — `Embedder` ABC로 추상화
- **Chat Interface**: `python-telegram-bot` / `discord.py` — `ChatAdapter` ABC로 추상화
- **Config**: `pydantic-settings` (`.env` 기반)
- **Test**: `pytest`, `pytest-asyncio`, `pytest-mock`, `ragas` (eval only)

## 디렉터리 구조
```
tax-llm/
├── CLAUDE.md
├── pyproject.toml
├── .env.example
├── config/
│   └── settings.py            # Pydantic BaseSettings, 환경별 분기
├── src/
│   ├── interfaces/
│   │   ├── base.py             # ChatAdapter ABC
│   │   ├── telegram_adapter.py
│   │   └── discord_adapter.py
│   ├── application/
│   │   ├── session.py          # 대화 세션 관리 (in-memory, 확장 가능)
│   │   ├── router.py           # RAG / 일반 대화 분기
│   │   └── prompt_builder.py   # 시스템 프롬프트 조립
│   ├── rag/
│   │   ├── engine.py           # RAG 파이프라인 오케스트레이션
│   │   ├── retriever.py        # retrieve + rerank
│   │   └── reranker.py         # Reranker ABC + 구현체
│   ├── llm/
│   │   ├── base.py             # LLMProvider ABC
│   │   ├── gemini.py           # Gemini 구현체
│   │   └── local.py            # Ollama 등 로컬 모델 구현체
│   ├── ingestion/
│   │   ├── loader.py           # PDF / HWPX / 웹 문서 로더
│   │   ├── chunker.py          # RecursiveCharacterTextSplitter 래퍼
│   │   └── pipeline.py         # 수집 → 청킹 → 인덱싱 파이프라인
│   └── storage/
│       ├── vector_store.py     # VectorStore ABC
│       ├── embedder.py         # Embedder ABC
│       ├── chroma_store.py     # Chroma 구현체
│       └── qdrant_store.py     # Qdrant 구현체
├── scripts/
│   └── ingest.py               # 문서 수집 CLI (python scripts/ingest.py --path ...)
└── tests/
    ├── conftest.py             # 공통 fixture (mock LLM, mock VectorStore 등)
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_prompt_builder.py
    │   └── test_router.py
    ├── integration/
    │   ├── test_rag_pipeline.py
    │   └── test_adapter_contract.py
    └── eval/
        ├── README.md           # eval 실행 방법 및 점수 기준
        ├── test_faithfulness.py
        └── test_retrieval_quality.py
```

## 핵심 설계 원칙

### 1. 모든 외부 의존성은 ABC 뒤에 숨긴다
```python
# 올바른 방법
from src.llm.base import LLMProvider
from src.storage.vector_store import VectorStore

# 금지: application layer에서 직접 import
from src.llm.gemini import GeminiProvider  # 금지
import chromadb  # 금지: storage layer 밖에서 직접 import
```

### 2. 환경변수는 settings 객체로만 접근한다
```python
# 올바른 방법
from config.settings import settings
api_key = settings.gemini_api_key

# 금지
import os
api_key = os.getenv("GEMINI_API_KEY")  # 금지
```

### 3. RAG 답변은 반드시 출처를 포함한다
RAG engine이 반환하는 `RAGResponse`에는 항상 `source_chunks: list[SourceChunk]`가 포함되어야 한다.
LLM 프롬프트에는 "제공된 문서 외의 내용으로 답하지 마시오" constraint를 명시한다.

### 4. LLM API를 호출하는 테스트는 mock 처리한다
`@pytest.mark.eval` 마커가 붙은 테스트만 실제 API를 호출한다.
기본 `pytest` 실행 시 eval 마커 테스트는 skip된다.

## 구현체 교체 방법
`config/settings.py`의 아래 필드만 변경하면 된다:
```
VECTOR_STORE_BACKEND=chroma   # chroma | qdrant
EMBEDDER_BACKEND=sentence_transformers  # sentence_transformers | openai | local
LLM_BACKEND=gemini            # gemini | ollama | openai
CHAT_INTERFACE=telegram       # telegram | discord | cli
```

## 테스트 실행
```bash
uv run pytest tests/unit tests/integration          # 빠른 테스트 (mock, API 호출 없음)
uv run pytest tests/eval -m eval                    # RAG 품질 평가 (실제 API 호출)
uv run pytest --cov=src --cov-report=term-missing   # 커버리지 포함
```

## 작업 시 주의사항
- 새 파일 생성 전 해당 레이어의 ABC가 존재하는지 확인할 것
- 비동기(async/await)를 기본으로 작성할 것 (LLM/DB I/O 모두 비동기)
- 타입 힌트를 모든 함수에 명시할 것
- docstring은 한국어로 작성할 것
