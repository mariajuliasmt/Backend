from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    
    # ============ PIPELINE DO BANCO DE DADOS ============

    EMBEDDING_MODEL: str
    WEIGHT_EMBEDDING: float
    WEIGHT_KEYWORDS: float
    DUPLICATION_THRESHOLD: float

    BASE_FOLDER: str
    WINDOW_SIZE: int

    chunk_SIZE: int
    overlap: int

    PERSIST_DIRECTORY: str

    # ============ PIPELINE DO RAG ============
    PATH_DB: str
    temperature: float
    DEFAULT_LLM_TIMEOUT_S: int
    DEFAULT_LLM_MAX_TOKENS: int
    DEFAULT_LLM_SLOW_THRESHOLD_S: int 

    LLM_MODEL: str
    LLM_BASE_URL: str
    LLM_API_KEY: str

    # ============ EMBEDDING / RETRIEVAL ============
    k_retrieval: int
    k_results: int
    t: float 

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
