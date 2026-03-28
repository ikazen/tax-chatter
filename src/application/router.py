"""RAG 질의 vs. 일반 대화 분기 라우터."""

# 세법 관련 키워드 — 이 단어가 포함된 질의는 RAG 경로로 처리
TAX_KEYWORDS: frozenset[str] = frozenset([
    "세금", "세법", "소득세", "법인세", "부가가치세", "부가세",
    "양도세", "상속세", "증여세", "종합소득세", "원천징수",
    "세율", "공제", "감면", "납세", "신고", "납부", "환급",
    "판례", "예규", "유권해석", "국세청", "세무서",
])


def is_tax_query(text: str) -> bool:
    """세법 관련 질의인지 판단한다.

    키워드 매칭은 빠르지만 recall이 낮으므로,
    추후 LLM 기반 분류기로 교체할 수 있도록 함수로 분리한다.
    """
    return any(keyword in text for keyword in TAX_KEYWORDS)
