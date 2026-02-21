"""
Configuration settings for the PDF Chatbot system.

IMPORTANT:
We intentionally avoid `pydantic-settings` here because in some locked-down /
offline / conda environments it's not available (and pip install may fail).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def find_project_root() -> Path:
    """
    Find project root directory (prefer where `.env` is; otherwise where this file is).
    """
    current = Path(__file__).parent.absolute()
    for path in [current] + list(current.parents):
        if (path / ".env").exists():
            return path
    return current


PROJECT_ROOT = find_project_root()

# Load .env once (if present) and OVERRIDE any existing env vars.
# روی بعضی سیستم‌ها ممکن است قبلاً OPENAI_API_KEY در محیط ست شده باشد
# و باعث شود کلید اشتباه استفاده شود؛ با override=True مطمئن می‌شویم
# مقادیر داخل فایل .env بر همه چیز اولویت دارند.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o")

    # Document
    pdf_file_path: str = os.getenv("PDF_FILE_PATH", "shopmanual.pdf")
    chunk_size: int = _env_int("CHUNK_SIZE", 2500)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 500)

    # Retrieval & RAG
    top_k_results: int = _env_int("TOP_K_RESULTS", 5)
    similarity_threshold: float = _env_float("SIMILARITY_THRESHOLD", 0.7)
    rerank_initial_k: int = _env_int("RERANK_INITIAL_K", 40)  # تعداد کاندید برای ریرنک/MMR
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() in ("1", "true", "yes")
    mmr_lambda: float = _env_float("MMR_LAMBDA", 0.7)  # ۱=فقط شباهت، ۰=تنوع بیشتر

    # Vector DB
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "vector_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "shop_manual")

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    def resolve_paths(self) -> None:
        if not os.path.isabs(self.pdf_file_path):
            self.pdf_file_path = os.path.normpath(str(PROJECT_ROOT / self.pdf_file_path))
        if not os.path.isabs(self.vector_db_path):
            self.vector_db_path = os.path.normpath(str(PROJECT_ROOT / self.vector_db_path))


settings = Settings()
settings.resolve_paths()

