import os
import time
import hashlib
from typing import List, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
import spacy
import yake
import tiktoken
import nltk
from keybert import KeyBERT
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
import PyPDF2
import pdfplumber

# --- NLTK and spaCy model loading (should be pre-installed in Docker) ---
# No runtime downloads; Dockerfile handles this.

@dataclass
class RAGConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    index_name: str = os.getenv("PINECONE_INDEX")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    keyword_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    cache_dir: str = "/tmp/cache"

config = RAGConfig()

# --- Initialization ---
print("Initializing models and connections...")

client = OpenAI(api_key=config.openai_api_key)
pc = Pinecone(api_key=config.pinecone_api_key)

# NLP Models
nlp = spacy.load("en_core_web_lg")
kw_extractor = yake.KeywordExtractor(n=3, top=10, dedupLim=0.7)
keybert_model = KeyBERT(model=config.keyword_model)
cross_encoder = CrossEncoder(config.reranker_model)
encoding = tiktoken.encoding_for_model("gpt-4")

os.makedirs(config.cache_dir, exist_ok=True)

# --- Pinecone Index Setup ---
def setup_pinecone_index():
    dimension = 1536  # for text-embedding-3-small
    if config.index_name not in pc.list_indexes().names():
        print(f"Creating index {config.index_name}...")
        pc.create_index(
            name=config.index_name,
            dimension=dimension,
            metric="cosine",
            spec={ "pod": { "environment": "gcp-starter" } }
        )
        while not pc.describe_index(config.index_name).status['ready']:
            time.sleep(1)
    return pc.Index(config.index_name)

index = setup_pinecone_index()

# --- Document Readers ---
def read_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            return None
    return normalize_text(text) if text else None

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace('\x00', '').replace('\xa0', ' ')
    s = s.encode('utf-8', 'ignore').decode('utf-8')
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])

def read_txt(path):
    try:
        with open(path, "r", encoding='utf-8', errors="ignore") as f:
            return normalize_text(f.read())
    except Exception:
        return None

# --- Text Processing ---
class EnhancedTextProcessor:
    def extract_entities(self, text: str) -> List[str]:
        doc = nlp(text[:1000000])
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]]
        return list(set(entities))

    def extract_keywords(self, text: str) -> List[str]:
        keywords = set()
        yake_kws = [kw for kw, _ in kw_extractor.extract_keywords(text)]
        keywords.update(yake_kws[:5])
        try:
            keybert_kws = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)
            keywords.update([kw for kw, _ in keybert_kws])
        except Exception:
            pass
        return list(keywords)[:10]

    def generate_summary(self, text: str) -> str:
        if len(text) < 100:
            return text
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a 2-sentence summary focusing on key information."},
                    {"role": "user", "content": text[:2000]}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return '. '.join(text.split('.')[:2]).strip()

processor = EnhancedTextProcessor()

# --- Chunking ---
class SemanticChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=lambda text: len(encoding.encode(text)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_with_context(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = self.splitter.split_text(text)
        enhanced_chunks = []
        for idx, chunk_text in enumerate(chunks):
            enhanced_chunks.append({
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **metadata
            })
        return enhanced_chunks

chunker = SemanticChunker()

# --- Embedding Generation ---
class EmbeddingGenerator:
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = client.embeddings.create(
                model=config.embedding_model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return [None] * len(texts)

embedder = EmbeddingGenerator()

# --- Main Processing Pipeline ---
def process_file(file_path: str, original_filename: str) -> List[Dict]:
    print(f"Processing: {original_filename}")
    ext = os.path.splitext(original_filename)[1].lower()
    text = None
    if ext == ".pdf":
        text = read_pdf(file_path)
    elif ext == ".txt":
        text = read_txt(file_path)
    # Add other file types here...

    if not text:
        return []

    base_metadata = {
        "source": original_filename,
        "file_type": ext[1:],
        "processing_timestamp": time.time()
    }
    
    chunks = chunker.chunk_with_context(text, base_metadata)
    processed_chunks = []
    for chunk_data in chunks:
        if len(chunk_data["text"]) < 50:
            continue
        
        chunk_hash = hashlib.sha256(chunk_data["text"].encode()).hexdigest()[:16]
        chunk_id = f"{original_filename}-{chunk_hash}"
        
        metadata = {
            **chunk_data,
            "chunk_keywords": processor.extract_keywords(chunk_data["text"])
        }
        
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_data["text"],
            "metadata": metadata
        })
    return processed_chunks

def upload_to_pinecone(chunks: List[Dict], namespace: str) -> int:
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embedder.get_embeddings_batch(texts)
    
    valid_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding:
            metadata = chunk["metadata"]
            metadata['text'] = chunk["text"]
            valid_chunks.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata
            })

    if not valid_chunks:
        return 0
    
    uploaded_count = 0
    for i in range(0, len(valid_chunks), config.batch_size):
        batch = valid_chunks[i:i+config.batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            uploaded_count += len(batch)
            print(f"Uploaded {len(batch)} vectors to namespace '{namespace}'.")
        except Exception as e:
            print(f"Pinecone upload error: {e}")
    return uploaded_count



# import os
# import json
# import time
# import hashlib
# import pandas as pd
# import numpy as np
# import spacy
# import yake
# import tiktoken
# import pickle
# from typing import List, Dict
# from dataclasses import dataclass
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # Advanced libraries
# import nltk
# from keybert import KeyBERT
# from sentence_transformers import CrossEncoder
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from openai import OpenAI
# from pinecone import Pinecone
# import PyPDF2
# import pdfplumber
# import chardet

# # Download required NLTK data during container build time
# # Note: It's best practice to have these downloads in the Dockerfile,
# # but including them here provides a fallback.
# nltk.download('wordnet', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)

# @dataclass
# class RAGConfig:
#     openai_api_key: str = os.getenv("OPENAI_API_KEY")
#     pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
#     index_name: str = os.getenv("PINECONE_INDEX")
#     namespace_dense: str = "dense_vectors"
#     embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
#     reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
#     keyword_model: str = "all-MiniLM-L6-v2"
#     chunk_size: int = 800
#     chunk_overlap: int = 150
#     batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
#     cache_dir: str = "/tmp/cache" # Use the /tmp directory for cache in serverless environments

# config = RAGConfig()

# # === INITIALIZATION ===
# print("Initializing models and connections...")

# client = OpenAI(api_key=config.openai_api_key)
# pc = Pinecone(api_key=config.pinecone_api_key)

# # NLP Models
# try:
#     nlp = spacy.load("en_core_web_lg")
# except OSError:
#     print("Downloading spaCy large model...")
#     os.system("python -m spacy download en_core_web_lg")
#     nlp = spacy.load("en_core_web_lg")

# # Keyword extraction and reranking models
# kw_extractor = yake.KeywordExtractor(n=3, top=10, dedupLim=0.7)
# keybert_model = KeyBERT(model=config.keyword_model)
# cross_encoder = CrossEncoder(config.reranker_model)

# # Tokenizer
# encoding = tiktoken.encoding_for_model("gpt-4")

# # Create cache directory in the temporary filesystem
# os.makedirs(config.cache_dir, exist_ok=True)

# # === PINECONE INDEX SETUP ===
# def setup_pinecone_index():
#     dimension = 1536  # for text-embedding-3-small
#     if config.index_name not in pc.list_indexes().names():
#         print(f"Creating index {config.index_name}...")
#         pc.create_index(
#             name=config.index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec={ "pod": { "environment": "gcp-starter" } }
#         )
#         while not pc.describe_index(config.index_name).status['ready']:
#             time.sleep(1)
#     return pc.Index(config.index_name)

# index = setup_pinecone_index()

# # === DOCUMENT READERS ===
# def read_pdf(path):
#     text = ""
#     try:
#         with pdfplumber.open(path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#     except Exception:
#         try:
#             with open(path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 for page in pdf_reader.pages:
#                     text += page.extract_text() + "\n"
#         except Exception:
#             return None
#     return normalize_text(text) if text else None

# def normalize_text(s: str) -> str:
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     s = s.replace('\x00', '').replace('\xa0', ' ')
#     s = s.encode('utf-8', 'ignore').decode('utf-8')
#     return "\n".join([line.strip() for line in s.splitlines() if line.strip()])

# def read_txt(path):
#     try:
#         with open(path, "r", encoding='utf-8', errors="ignore") as f:
#             return normalize_text(f.read())
#     except Exception:
#         return None

# # Add other readers (read_json, read_tsv) here if needed...

# # === ADVANCED TEXT PROCESSING ===
# class EnhancedTextProcessor:
#     def extract_entities(self, text: str) -> List[str]:
#         doc = nlp(text[:1000000]) # Limit for performance
#         entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]]
#         return list(set(entities))

#     def extract_keywords(self, text: str) -> List[str]:
#         keywords = set()
#         yake_kws = [kw for kw, _ in kw_extractor.extract_keywords(text)]
#         keywords.update(yake_kws[:5])
#         try:
#             keybert_kws = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=5)
#             keywords.update([kw for kw, _ in keybert_kws])
#         except Exception:
#             pass
#         return list(keywords)[:10]

#     def generate_summary(self, text: str) -> str:
#         if len(text) < 100:
#             return text
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "Create a 2-sentence summary focusing on key information."},
#                     {"role": "user", "content": text[:2000]}
#                 ],
#                 max_tokens=100,
#                 temperature=0.3
#             )
#             return response.choices[0].message.content.strip()
#         except Exception:
#             return '. '.join(text.split('.')[:2]).strip()

# processor = EnhancedTextProcessor()

# # === INTELLIGENT CHUNKING ===
# class SemanticChunker:
#     def __init__(self):
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=config.chunk_size,
#             chunk_overlap=config.chunk_overlap,
#             length_function=lambda text: len(encoding.encode(text)),
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )

#     def chunk_with_context(self, text: str, metadata: Dict) -> List[Dict]:
#         chunks = self.splitter.split_text(text)
#         enhanced_chunks = []
#         for idx, chunk_text in enumerate(chunks):
#             enhanced_chunks.append({
#                 "text": chunk_text,
#                 "chunk_index": idx,
#                 "total_chunks": len(chunks),
#                 **metadata
#             })
#         return enhanced_chunks

# chunker = SemanticChunker()

# # === EMBEDDING GENERATION ===
# class EmbeddingGenerator:
#     def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
#         if not texts:
#             return []
#         try:
#             response = client.embeddings.create(
#                 model=config.embedding_model,
#                 input=texts,
#                 encoding_format="float"
#             )
#             return [data.embedding for data in response.data]
#         except Exception as e:
#             print(f"Batch embedding error: {e}")
#             return [None] * len(texts)

# embedder = EmbeddingGenerator()

# # === MAIN PROCESSING PIPELINE ===
# def process_file(file_path: str, original_filename: str) -> List[Dict]:
#     print(f"Processing: {original_filename}")
#     ext = os.path.splitext(original_filename)[1].lower()
#     text = None
#     if ext == ".pdf":
#         text = read_pdf(file_path)
#     elif ext == ".txt":
#         text = read_txt(file_path)
#     # Add other file types here...

#     if not text:
#         return []

#     base_metadata = {
#         "source": original_filename,
#         "file_type": ext[1:],
#         "processing_timestamp": time.time()
#     }
    
#     chunks = chunker.chunk_with_context(text, base_metadata)
#     processed_chunks = []
#     for chunk_data in chunks:
#         if len(chunk_data["text"]) < 50:
#             continue
        
#         chunk_hash = hashlib.sha256(chunk_data["text"].encode()).hexdigest()[:16]
#         chunk_id = f"{original_filename}-{chunk_hash}"
        
#         metadata = {
#             **chunk_data,
#             "chunk_keywords": processor.extract_keywords(chunk_data["text"])
#         }
        
#         processed_chunks.append({
#             "id": chunk_id,
#             "text": chunk_data["text"],
#             "metadata": metadata
#         })
#     return processed_chunks

# def upload_to_pinecone(chunks: List[Dict], namespace: str) -> int:
#     if not chunks:
#         return 0

#     texts = [c["text"] for c in chunks]
#     embeddings = embedder.get_embeddings_batch(texts)
    
#     valid_chunks = []
#     for chunk, embedding in zip(chunks, embeddings):
#         if embedding:
#             metadata = chunk["metadata"]
#             # Ensure text is stored in metadata
#             metadata['text'] = chunk["text"]
#             valid_chunks.append({
#                 "id": chunk["id"],
#                 "values": embedding,
#                 "metadata": metadata
#             })

#     if not valid_chunks:
#         return 0
    
#     uploaded_count = 0
#     for i in range(0, len(valid_chunks), config.batch_size):
#         batch = valid_chunks[i:i+config.batch_size]
#         try:
#             index.upsert(vectors=batch, namespace=namespace)
#             uploaded_count += len(batch)
#             print(f"Uploaded {len(batch)} vectors to namespace '{namespace}'.")
#         except Exception as e:
#             print(f"Pinecone upload error: {e}")
#     return uploaded_count