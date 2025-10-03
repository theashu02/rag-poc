from typing import Dict, Iterable, List

from tenacity import retry, stop_after_attempt, wait_exponential

from .clients import get_pinecone_index, get_tiktoken_encoding
from .config import config
from .embedder import embedder


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _upload_batch(index, batch, namespace):
    index.upsert(vectors=batch, namespace=namespace)
    return len(batch)


def _iter_embedding_batches(chunks: Iterable[Dict]) -> Iterable[List[Dict]]:
    tokenizer = get_tiktoken_encoding()
    batch: List[Dict] = []
    token_budget = 0

    for chunk in chunks:
        text = chunk["text"]
        tokens = len(tokenizer.encode(text))

        if batch and (
            len(batch) >= config.embedding_batch_size
            or token_budget + tokens > config.embedding_request_tokens
        ):
            yield batch
            batch = []
            token_budget = 0

        if tokens > config.embedding_request_tokens:
            print(f"[Uploader] chunk {chunk['id']} exceeds token budget; splitting upstream is recommended.")

        batch.append(chunk)
        token_budget += tokens

    if batch:
        yield batch


def _flush_vectors(index, vectors: List[Dict], namespace: str) -> int:
    uploaded = 0
    for start in range(0, len(vectors), config.batch_size):
        uploaded += _upload_batch(
            index,
            vectors[start : start + config.batch_size],
            namespace,
        )
    return uploaded


def upload_to_pinecone(chunks: List[Dict], namespace: str) -> int:
    if not chunks:
        return 0

    index = get_pinecone_index()
    uploaded = 0
    pending: List[Dict] = []

    for batch in _iter_embedding_batches(chunks):
        texts = [c["text"] for c in batch]
        embeddings = embedder.get_embeddings_batch(texts)
        if len(embeddings) != len(batch):
            print("[Uploader] Embeddings batch size mismatch; skipping batch.")
            continue

        for chunk, embedding in zip(batch, embeddings):
            if not embedding:
                continue
            meta = dict(chunk["metadata"])
            meta["text"] = chunk["text"]
            pending.append(
                {
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": meta,
                }
            )

        if len(pending) >= config.batch_size:
            uploaded += _flush_vectors(index, pending, namespace)
            pending = []

    if pending:
        uploaded += _flush_vectors(index, pending, namespace)

    return uploaded
