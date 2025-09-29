import time
from typing import Optional, Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import spacy, yake, tiktoken
from keybert import KeyBERT

from .config import config

_client:          Optional[OpenAI]     = None
_pc:              Optional[Pinecone]   = None
_nlp:             Optional[Any]        = None
_kw_extractor:    Optional[yake.KeywordExtractor] = None
_keybert_model:   Optional[KeyBERT]    = None
_encoding:        Optional[tiktoken.Encoding]     = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key, timeout=config.openai_timeout)
    return _client


def get_pinecone_index():
    global _pc
    if _pc is None: _pc = Pinecone(api_key=config.pinecone_api_key)

    dim = 1536            # for text-embedding-3-small/large 3072
    if config.index_name not in _pc.list_indexes().names():
        print(f"[Pinecone] creating index '{config.index_name}' …")
        _pc.create_index(
            name=config.index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not _pc.describe_index(config.index_name).status['ready']:
            time.sleep(1)

    return _pc.Index(config.index_name)


def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
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


def get_tiktoken_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model("gpt-4o")
    return _encoding