from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "AegisLLM"
    VERSION: str = "0.1.0"
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_PATH: Path = DATA_DIR / "faiss_index"
    
    # Gemini API
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str = "models/gemini-2.5-pro"
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 2048
    
    # Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Retrieval
    TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3
    CHUNK_SIZE: int = 512 # Token-aware roughly
    CHUNK_OVERLAP: int = 64
    
    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_TRACING: bool = True
    
    # Redis (Phase 3+)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_STREAM_NAME: str = "aegis_ingestion_stream"
    REDIS_CONSUMER_GROUP: str = "aegis_ingestion_group"
    
    # Phase 3: Continual Learning
    TEMPORAL_DECAY_RATE: float = 0.01
    HYBRID_FRESHNESS_WEIGHT: float = 0.3
    
    # Phase 4: Context Optimization
    CONTEXT_COMPRESSION_THRESHOLD: float = 0.85
    MAX_CONTEXT_TOKENS: int = 4000
    ENABLE_MAP_REDUCE: bool = True

    # Phase 6: Safety Layer (The Shield)
    ENABLE_SAFETY_GUARDRAILS: bool = True
    INJECTION_RISK_THRESHOLD: float = 0.7
    GROUNDING_SIMILARITY_THRESHOLD: float = 0.65
    SAFETY_BLOCK_RESPONSE: str = "Query blocked due to safety/grounding policy."
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Force directory creation
Settings().DATA_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
