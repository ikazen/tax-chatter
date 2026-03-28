"""문서 로더 ABC 및 PDF 구현체."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class LoadedPage:
    """로드된 문서 페이지."""

    text: str
    metadata: dict  # {"source": "파일명.pdf", "page": 1}


class DocumentLoader(ABC):
    """모든 문서 로더 구현체가 따라야 하는 인터페이스."""

    @abstractmethod
    def load(self, file_path: str | Path) -> list[LoadedPage]:
        """단일 파일을 로드한다."""
        ...

    @abstractmethod
    def load_directory(
        self, dir_path: str | Path, glob: str
    ) -> list[LoadedPage]:
        """디렉토리 내 파일을 일괄 로드한다."""
        ...


class PDFLoader(DocumentLoader):
    """PDF 파일에서 페이지별 텍스트를 추출한다."""

    def load(self, file_path: str | Path) -> list[LoadedPage]:
        """단일 PDF 파일을 로드한다."""
        path = Path(file_path)
        reader = PdfReader(path)
        pages: list[LoadedPage] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            pages.append(
                LoadedPage(
                    text=text,
                    metadata={"source": path.name, "page": i + 1},
                )
            )
        return pages

    def load_directory(
        self, dir_path: str | Path, glob: str = "*.pdf"
    ) -> list[LoadedPage]:
        """디렉토리 내 PDF 파일을 일괄 로드한다."""
        path = Path(dir_path)
        pages: list[LoadedPage] = []
        for file in sorted(path.glob(glob)):
            pages.extend(self.load(file))
        return pages
