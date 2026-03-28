"""src/application/router.py 단위 테스트."""
import pytest
from src.application.router import is_tax_query


@pytest.mark.parametrize("text, expected", [
    ("소득세율이 어떻게 되나요?", True),
    ("법인세 신고 기한이 언제인가요?", True),
    ("부가가치세 환급 절차를 알려주세요", True),
    ("판례 검색 부탁드립니다", True),
    ("오늘 날씨가 어때요?", False),
    ("점심 메뉴 추천해줘", False),
    ("엑셀 함수 알려줘", False),
])
def test_is_tax_query(text: str, expected: bool) -> None:
    assert is_tax_query(text) == expected


def test_is_tax_query_empty_string() -> None:
    assert is_tax_query("") is False


def test_is_tax_query_keyword_boundary() -> None:
    """키워드가 단어 중간에 포함된 경우도 매칭되는지 확인."""
    # 현재 구현은 substring 매칭 — 향후 형태소 분석으로 개선 가능
    assert is_tax_query("부가세포함가격") is True
