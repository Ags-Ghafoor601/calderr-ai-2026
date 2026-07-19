"""Enterprise Document Intelligence Platform — Application Configuration."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    groq_api_key: str = ""
    hf_token: str = ""

    # Application
    app_name: str = "Enterprise Document Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ChromaDB
    chroma_dir: str = str(Path(__file__).resolve().parent.parent / ".chromadb")

    # RAG Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6

    # Upload
    upload_dir: str = str(Path(__file__).resolve().parent.parent / "uploads")
    max_file_size_mb: int = 50

    model_config = {"env_file": str(Path(__file__).resolve().parent.parent.parent.parent / ".env")}


settings = Settings()

# Ensure directories exist
os.makedirs(settings.chroma_dir, exist_ok=True)
os.makedirs(settings.upload_dir, exist_ok=True)
