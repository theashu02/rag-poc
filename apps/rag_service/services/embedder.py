from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from .clients import get_openai_client
from .config import config

class EmbeddingGenerator:
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        client = get_openai_client()
        response = client.embeddings.create(
            model           = config.embedding_model,
            input           = texts,
            encoding_format = "float",
        )
        return [data.embedding for data in response.data]

embedder = EmbeddingGenerator()