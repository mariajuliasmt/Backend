"""Configuração central do TutorIA.

Os defaults permitem importar o pipeline em ambientes de benchmark sem .env.
Valores específicos podem ser sobrescritos por variáveis de ambiente ou src/.env.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

SRC_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Banco / roteamento
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    WEIGHT_EMBEDDING: float = 0.80
    WEIGHT_KEYWORDS: float = 0.20
    DUPLICATION_THRESHOLD: float = 0.05
    BASE_FOLDER: str = str(SRC_DIR / "base")
    WINDOW_SIZE: int = 3
    chunk_SIZE: int = 500
    overlap: int = 75
    PERSIST_DIRECTORY: str = str(SRC_DIR / "DataBase")

    # RAG / LLM
    PATH_DB: str = str(SRC_DIR / "DataBase")
    temperature: float = 0.0
    DEFAULT_LLM_TIMEOUT_S: int = 120
    DEFAULT_LLM_MAX_TOKENS: int = 350
    DEFAULT_LLM_SLOW_THRESHOLD_S: int = 60
    LLM_MODEL: str = "llama-int8"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "EMPTY"

    # Retrieval
    k_retrieval: int = 3
    k_results: int = 4
    t: float = 0.5

    model_config = SettingsConfigDict(
        env_file=str(SRC_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
