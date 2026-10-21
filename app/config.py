from functools import lru_cache
from pathlib import Path
import os

from pydantic import BaseModel


class Settings(BaseModel):
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chroma_persist_directory: Path = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")
    )
    documents_directory: Path = Path(os.getenv("DOCUMENTS_DIRECTORY", "data/documents"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
