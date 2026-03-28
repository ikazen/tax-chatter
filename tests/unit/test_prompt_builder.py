"""prompt_builder 단위 테스트."""
from src.application.prompt_builder import GENERAL_SYSTEM_PROMPT, build_general_prompt


def test_build_general_prompt_returns_string() -> None:
    """build_general_prompt가 문자열을 반환한다."""
    result = build_general_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_general_prompt_contains_constraint() -> None:
    """일반 대화 프롬프트에 세법 답변 금지 제약이 포함된다."""
    result = build_general_prompt()
    assert "세법" in result


def test_general_prompt_matches_constant() -> None:
    """build_general_prompt가 GENERAL_SYSTEM_PROMPT 상수를 반환한다."""
    assert build_general_prompt() == GENERAL_SYSTEM_PROMPT
