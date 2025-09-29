import hashlib, os, time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .chunker import chunker
from .document_readers import (
    read_pdf, read_txt, read_json, read_pptx, read_docx, read_doc
)
from .text_processor import processor
from .uploader import upload_to_pinecone
from .config import config


def _read_by_extension(path: str, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if   ext == ".pdf":  return read_pdf(path)
    elif ext == ".txt":  return read_txt(path)
    elif ext == ".json": return read_json(path)
    elif ext == ".pptx": return read_pptx(path)
    elif ext == ".docx": return read_docx(path)
    elif ext == ".doc":  return read_doc(path)
    return None


def process_file(file_path: str, original_filename: str) -> List[Dict]:
    print(f"[RAG] Processing: {original_filename}")
    text = _read_by_extension(file_path, original_filename)
    if not text:
        return []

    base_meta = {
        "source": original_filename,
        "file_type": os.path.splitext(original_filename)[1][1:],
        "processing_timestamp": time.time(),
    }

    chunks  = chunker.chunk_with_context(text, base_meta)
    results = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futs = [
            pool.submit(_process_chunk, ch, original_filename)
            for ch in chunks if len(ch["text"]) >= 50
        ]
        for f in futs:
            results.append(f.result())

    return results


def _process_chunk(chunk_data: Dict, original_filename: str) -> Dict:
    chunk_hash = hashlib.sha256(chunk_data["text"].encode()).hexdigest()[:16]
    chunk_id   = f"{original_filename}-{chunk_hash}"

    metadata = {
        **chunk_data,
        "chunk_keywords": processor.extract_keywords(chunk_data["text"]),
    }

    return {
        "id":       chunk_id,
        "text":     chunk_data["text"],
        "metadata": metadata,
    }


# Convenient one-liner ------------------------------------------------------- #

def ingest_file_to_pinecone(path: str, *, namespace: str = "default") -> int:
    """
    High-level helper: read → chunk → embed → upload.  
    Returns number of vectors uploaded.
    """
    filename = os.path.basename(path)
    chunks   = process_file(path, filename)
    return upload_to_pinecone(chunks, namespace)