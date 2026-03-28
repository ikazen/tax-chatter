"""PDFLoader 단위 테스트."""
from unittest.mock import MagicMock, patch

from src.ingestion.loader import PDFLoader


def _make_mock_page(text: str) -> MagicMock:
    """mock PDF 페이지를 생성한다."""
    page = MagicMock()
    page.extract_text.return_value = text
    return page


@patch("src.ingestion.loader.PdfReader")
def test_load_extracts_pages(mock_reader_class: MagicMock) -> None:
    """load가 페이지별 텍스트를 추출한다."""
    mock_reader = MagicMock()
    mock_reader.pages = [
        _make_mock_page("페이지1 내용"),
        _make_mock_page("페이지2 내용"),
    ]
    mock_reader_class.return_value = mock_reader

    loader = PDFLoader()
    pages = loader.load("/tmp/test.pdf")

    assert len(pages) == 2
    assert pages[0].text == "페이지1 내용"
    assert pages[1].text == "페이지2 내용"


@patch("src.ingestion.loader.PdfReader")
def test_load_metadata(mock_reader_class: MagicMock) -> None:
    """load가 올바른 메타데이터를 생성한다."""
    mock_reader = MagicMock()
    mock_reader.pages = [_make_mock_page("내용")]
    mock_reader_class.return_value = mock_reader

    loader = PDFLoader()
    pages = loader.load("/data/docs/소득세법.pdf")

    assert pages[0].metadata["source"] == "소득세법.pdf"
    assert pages[0].metadata["page"] == 1


@patch("src.ingestion.loader.PdfReader")
def test_load_skips_empty_pages(mock_reader_class: MagicMock) -> None:
    """빈 페이지를 건너뛴다."""
    mock_reader = MagicMock()
    mock_reader.pages = [
        _make_mock_page("내용"),
        _make_mock_page(""),
        _make_mock_page("   "),
        _make_mock_page("실제 내용"),
    ]
    mock_reader_class.return_value = mock_reader

    loader = PDFLoader()
    pages = loader.load("/tmp/test.pdf")

    assert len(pages) == 2
    assert pages[0].text == "내용"
    assert pages[1].text == "실제 내용"


@patch("src.ingestion.loader.PdfReader")
def test_load_strips_whitespace(mock_reader_class: MagicMock) -> None:
    """페이지 텍스트의 앞뒤 공백을 제거한다."""
    mock_reader = MagicMock()
    mock_reader.pages = [_make_mock_page("  공백 텍스트  \n")]
    mock_reader_class.return_value = mock_reader

    loader = PDFLoader()
    pages = loader.load("/tmp/test.pdf")

    assert pages[0].text == "공백 텍스트"


@patch("src.ingestion.loader.PdfReader")
def test_load_page_numbers_are_1_indexed(mock_reader_class: MagicMock) -> None:
    """페이지 번호가 1부터 시작한다."""
    mock_reader = MagicMock()
    mock_reader.pages = [
        _make_mock_page("첫번째"),
        _make_mock_page("두번째"),
        _make_mock_page("세번째"),
    ]
    mock_reader_class.return_value = mock_reader

    loader = PDFLoader()
    pages = loader.load("/tmp/test.pdf")

    assert [p.metadata["page"] for p in pages] == [1, 2, 3]
