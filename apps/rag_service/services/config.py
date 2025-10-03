from dataclasses import dataclass
import os


def _env_bool(key: str, default: str = "false") -> bool:
    value = os.getenv(key, default)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class RAGConfig:
    # Environment & model settings
    openai_api_key:   str = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    index_name:       str = os.getenv("PINECONE_INDEX")

    embedding_model:  str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    keyword_model:    str = os.getenv("KEYWORD_MODEL", "all-MiniLM-L6-v2")
    tokenizer_model:  str = os.getenv("TOKENIZER_MODEL", "gpt-4o")

    # Chunking controls
    chunk_size:        int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap:     int = int(os.getenv("CHUNK_OVERLAP", "100"))
    chunk_min_length:  int = int(os.getenv("CHUNK_MIN_LENGTH", "50"))

    # Concurrency / batching
    batch_size:               int = int(os.getenv("PINECONE_BATCH_SIZE", os.getenv("BATCH_SIZE", "100")))
    embedding_batch_size:     int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_request_tokens: int = int(os.getenv("EMBEDDING_REQUEST_TOKENS", "7000"))
    max_workers:              int = int(os.getenv("MAX_WORKERS", "6"))

    # Vector configuration
    embedding_dimension:    int = int(os.getenv("EMBEDDING_DIMENSION", "0"))
    vector_id_prefix_chars: int = int(os.getenv("VECTOR_ID_PREFIX_CHARS", "48"))
    vector_id_hash_length:  int = int(os.getenv("VECTOR_ID_HASH_LENGTH", "16"))

    # Feature flags
    enable_keyword_enrichment: bool = _env_bool("ENABLE_KEYWORD_ENRICHMENT", "true")
    enable_entity_enrichment: bool = _env_bool("ENABLE_ENTITY_ENRICHMENT", "false")

    # Timeouts
    openai_timeout:   int = int(os.getenv("OPENAI_TIMEOUT",   "30"))
    pinecone_timeout: int = int(os.getenv("PINECONE_TIMEOUT", "30"))

    # Pinecone deployment details
    pinecone_cloud:  str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    cache_dir:        str = os.getenv("CACHE_DIR", "/tmp/cache")


config = RAGConfig()
os.makedirs(config.cache_dir, exist_ok=True)
