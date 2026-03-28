"""문서 인덱싱 CLI 스크립트.

사용법:
    python scripts/ingest.py --path ./data/tax_docs
    python scripts/ingest.py --path ./data/tax_docs/소득세법.pdf
"""
import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import PDFLoader
from src.ingestion.pipeline import IngestionPipeline
from src.storage.chroma_store import ChromaStore
from src.storage.embedder_impl import SentenceTransformersEmbedder


async def main(path: str) -> None:
    """주어진 경로의 문서를 인덱싱한다."""
    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaStore(embedder=embedder)
    loader = PDFLoader()
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedder=embedder,
        loader=loader,
    )

    target = Path(path)
    if target.is_file():
        print(f"파일 인덱싱: {target}")
        count = await pipeline.ingest_file(str(target))
    elif target.is_dir():
        print(f"디렉토리 인덱싱: {target}")
        count = await pipeline.ingest_directory(str(target))
    else:
        print(f"경로를 찾을 수 없습니다: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"완료: {count}개 청크 인덱싱됨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 인덱싱 CLI")
    parser.add_argument(
        "--path",
        required=True,
        help="인덱싱할 PDF 파일 또는 디렉토리 경로",
    )
    args = parser.parse_args()
    asyncio.run(main(args.path))
