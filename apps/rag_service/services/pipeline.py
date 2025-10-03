import hashlib
import os
import re
import time
from typing import Dict, List, Optional

from .chunker import chunker
from .document_readers import (
    read_pdf,
    read_txt,
    read_json,
    read_pptx,
    read_docx,
)
from .text_processor import processor
from .config import config


_ID_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]+")


def _read_by_extension(path: str, filename: str) -> Optional[str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".txt":
        return read_txt(path)
    if ext == ".json":
        return read_json(path)
    if ext == ".pptx":
        return read_pptx(path)
    if ext == ".docx":
        return read_docx(path)
    return None


def _safe_id_prefix(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    safe = _ID_SANITIZE.sub("-", stem).strip("-")
    if not safe:
        safe = "doc"
    return safe[: config.vector_id_prefix_chars]


def process_file(file_path: str, original_filename: str) -> List[Dict]:
    print(f"[RAG] Processing: {original_filename}")
    text = _read_by_extension(file_path, original_filename)
    if not text:
        print(f"[RAG] No text extracted from {original_filename}")
        return []

    base_meta = {
        "source": original_filename,
        "file_type": os.path.splitext(original_filename)[1][1:],
        "processing_timestamp": time.time(),
        "char_length": len(text),
    }

    chunks = chunker.chunk_with_context(text, base_meta)
    results: List[Dict] = []
    prefix = _safe_id_prefix(original_filename)

    for chunk in chunks:
        if len(chunk["text"]) < config.chunk_min_length:
            continue
        results.append(_process_chunk(chunk, prefix))

    return results


def _process_chunk(chunk_data: Dict, id_prefix: str) -> Dict:
    chunk_text = chunk_data["text"]
    chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    chunk_hash = chunk_hash[: config.vector_id_hash_length]
    chunk_id = f"{id_prefix}-{chunk_hash}"

    metadata = {k: v for k, v in chunk_data.items() if k != "text"}
    keywords = processor.extract_keywords(chunk_text)
    if keywords:
        metadata["chunk_keywords"] = keywords

    entities = processor.extract_entities(chunk_text)
    if entities:
        metadata["chunk_entities"] = entities

    metadata["char_length"] = len(chunk_text)

    return {
        "id": chunk_id,
        "text": chunk_text,
        "metadata": metadata,
    }


# Convenient one-liner ------------------------------------------------------- #

def ingest_file_to_pinecone(path: str, *, namespace: str = "default") -> int:
    """High-level helper: read → chunk → embed → upload. Returns uploaded count."""
    filename = os.path.basename(path)
    chunks = process_file(path, filename)
    from .uploader import upload_to_pinecone

    return upload_to_pinecone(chunks, namespace)
