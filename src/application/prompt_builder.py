"""일반 대화용 시스템 프롬프트 빌더.

RAGEngine.build_system_prompt는 검색된 문서 컨텍스트를 주입하는 RAG 전용 프롬프트를 생성한다.
이 모듈은 세금 관련이 아닌 일반 대화에서 사용하는 시스템 프롬프트를 조립한다.
"""

GENERAL_SYSTEM_PROMPT = """당신은 세무사 사무실의 업무 보조 AI입니다.
세법 관련 질문에는 반드시 RAG 시스템을 통해 답변해야 하므로,
일반 대화에서 세법 관련 답변을 하지 마십시오.
일반 대화에서는 친절하고 간결하게 답변하십시오."""


def build_general_prompt() -> str:
    """일반 대화용 시스템 프롬프트를 반환한다."""
    return GENERAL_SYSTEM_PROMPT
