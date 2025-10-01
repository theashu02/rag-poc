from dataclasses import dataclass
import os

@dataclass
class RAGConfig:
    # Environment & model settings
    openai_api_key:   str = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    index_name:       str = os.getenv("PINECONE_INDEX")

    embedding_model:  str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    keyword_model:    str = "all-MiniLM-L6-v2"

    chunk_size:       int = 500
    chunk_overlap:    int = 100 #150 - 200 for legal doc and madical
    batch_size:       int = int(os.getenv("BATCH_SIZE", "100"))
    max_workers:      int = int(os.getenv("MAX_WORKERS", "6"))

    openai_timeout:   int = int(os.getenv("OPENAI_TIMEOUT",   "30"))
    pinecone_timeout: int = int(os.getenv("PINECONE_TIMEOUT", "30"))
    cache_dir:        str = "/tmp/cache"

config = RAGConfig()
os.makedirs(config.cache_dir, exist_ok=True)