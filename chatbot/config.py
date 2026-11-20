"""Configuration loading and validation for DocuChat AI."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    gemini_api_key: str | None
    gemini_chat_model: str
    gemini_embedding_model: str
    gemini_embedding_dimensions: int
    chat_history_limit: int
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings safely without exposing credential values."""
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"
        ),
        gemini_embedding_dimensions=int(
            os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768")
        ),
        chat_history_limit=int(os.getenv("CHAT_HISTORY_LIMIT", "20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "4")),
    )
