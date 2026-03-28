from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    llm_backend: Literal["gemini", "ollama", "openai"] = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Vector DB
    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tax_docs"

    # Embedding
    embedder_backend: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    embedder_model: str = "jhgan/ko-sroberta-multitask"

    # Chat Interface
    chat_interface: Literal["telegram", "discord", "cli"] = "telegram"
    telegram_bot_token: str = ""
    discord_bot_token: str = ""

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.5

    # App
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


settings = Settings()
