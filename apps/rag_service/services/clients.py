import time
from functools import lru_cache
from typing import Optional, Any, Dict

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import spacy
import yake
import tiktoken
from keybert import KeyBERT

from .config import config

_client: Optional[OpenAI] = None
_pc: Optional[Pinecone] = None
_nlp: Optional[Any] = None
_kw_extractor: Optional[yake.KeywordExtractor] = None
_keybert_model: Optional[KeyBERT] = None
_encoding: Optional[tiktoken.Encoding] = None

_MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-3-large-v1": 3072,
    "text-embedding-ada-002": 1536,
}


def _resolve_embedding_dimension() -> int:
    if config.embedding_dimension:
        return config.embedding_dimension
    if config.embedding_model in _MODEL_DIMENSIONS:
        return _MODEL_DIMENSIONS[config.embedding_model]
    return 1536


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key, timeout=config.openai_timeout)
    return _client


def get_pinecone_index():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=config.pinecone_api_key)

    dimension = _resolve_embedding_dimension()
    existing = {idx.name for idx in _pc.list_indexes()}
    if config.index_name not in existing:
        print(f"[Pinecone] creating index '{config.index_name}' with dim {dimension}.")
        _pc.create_index(
            name=config.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region),
        )
        while True:
            status = _pc.describe_index(config.index_name).status
            if status.get("ready"):
                break
            time.sleep(1)

    return _pc.Index(config.index_name)


@lru_cache(maxsize=1)
def _load_spacy_model() -> Any:
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        return spacy.load("en_core_web_sm")


def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp


def get_yake_extractor():
    global _kw_extractor
    if _kw_extractor is None:
        _kw_extractor = yake.KeywordExtractor(n=3, top=15, dedupLim=0.9)
    return _kw_extractor


def get_keybert_model():
    global _keybert_model
    if _keybert_model is None:
        _keybert_model = KeyBERT(model=config.keyword_model)
    return _keybert_model


def get_tiktoken_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        try:
            _encoding = tiktoken.encoding_for_model(config.tokenizer_model)
        except KeyError:
            _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


__all__ = [
    "get_openai_client",
    "get_pinecone_index",
    "get_spacy_nlp",
    "get_yake_extractor",
    "get_keybert_model",
    "get_tiktoken_encoding",
]
