from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from .clients import get_pinecone_index
from .config import config
from .embedder import embedder


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def _upload_batch(batch, namespace):
    index = get_pinecone_index()
    index.upsert(vectors=batch, namespace=namespace)
    return len(batch)


def upload_to_pinecone(chunks: List[Dict], namespace: str) -> int:
    if not chunks:
        return 0

    texts      = [c["text"] for c in chunks]
    embeddings = embedder.get_embeddings_batch(texts)

    valid = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding:
            meta = chunk["metadata"]
            meta["text"] = chunk["text"]
            valid.append({
                "id":     chunk["id"],
                "values": embedding,
                "metadata": meta,
            })

    if not valid:
        return 0

    uploaded = 0
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = [
            pool.submit(_upload_batch, valid[i:i+config.batch_size], namespace)
            for i in range(0, len(valid), config.batch_size)
        ]
        for f in futures:
            uploaded += f.result()

    return uploaded