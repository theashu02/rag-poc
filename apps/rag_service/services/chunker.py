from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .clients import get_tiktoken_encoding
from .config import config

class SemanticChunker:
    def __init__(self):
        encoding = get_tiktoken_encoding()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size     = config.chunk_size,
            chunk_overlap  = config.chunk_overlap,
            length_function= lambda text: len(encoding.encode(text)),
            separators     = ["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_with_context(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = self.splitter.split_text(text)
        return [
            {
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **metadata
            }
            for idx, chunk_text in enumerate(chunks)
        ]

chunker = SemanticChunker()