import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
import spacy
import yake
import tiktoken
import asyncio
import pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Advanced libraries
import nltk
from nltk.corpus import wordnet
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import PyPDF2
import pdfplumber
import chardet

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

load_dotenv()

@dataclass
class RAGConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    
    index_name: str = os.getenv("PINECONE_INDEX")
    namespace_dense: str = "dense_vectors"
    namespace_sparse: str = "sparse_vectors"
    
    data_folder: str = os.getenv("DATA_DIR")
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")
    
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    keyword_model: str = "all-MiniLM-L6-v2"
    
    chunk_size: int = 800
    chunk_overlap: int = 150
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    max_workers: int = 4
    
    initial_retrieval_k: int = 100
    rerank_k: int = 30
    final_k: int = 10
    hybrid_alpha: float = 0.7  # Weight for dense vs sparse

config = RAGConfig()

# === INITIALIZATION ===
print("Initializing models and connections...")

client = OpenAI(api_key=config.openai_api_key)
pc = Pinecone(api_key=config.pinecone_api_key)

# NLP Models
try:
    nlp = spacy.load("en_core_web_lg")  # Use large model for better NER
except:
    print("Downloading spaCy large model...")
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Keyword extraction models
kw_extractor = yake.KeywordExtractor(n=3, top=10, dedupLim=0.7)
keybert_model = KeyBERT(model=config.keyword_model)

# Cross-encoder for reranking
cross_encoder = CrossEncoder(config.reranker_model)

# Tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# Create cache directory
os.makedirs(config.cache_dir, exist_ok=True)

# === ENHANCED PINECONE INDEX ===
def setup_pinecone_index():
    """Create optimized Pinecone index with metadata configuration"""
    dimension = 3072 if "large" in config.embedding_model else 1536
    
    if config.index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating index {config.index_name}...")
        pc.create_index(
            name=config.index_name,
            dimension=dimension,
            metric="cosine",
            cloud="aws",
            region="us-east-1",
            spec={
                "pod": {
                    "replicas": 1,
                    "shards": 1,
                    "pods": 1,
                    "pod_type": "p1.x1",
                    "metadata_config": {
                        "indexed": ["file_type", "source", "chunk_index"]
                    }
                }
            }
        )
        while not pc.describe_index(config.index_name).status['ready']:
            time.sleep(1)
    
    return pc.Index(config.index_name)

index = setup_pinecone_index()

# === DOCUMENT READERS ===
def detect_encoding(file_path):
    """Detect file encoding"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_pdf(path):
    """Enhanced PDF reader with fallback methods"""
    text = ""
    try:
        # Try pdfplumber first (better for tables)
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        try:
            # Fallback to PyPDF2
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except:
            return None
    
    return normalize_text(text) if text else None

def normalize_text(s: str) -> str:
    """Enhanced text normalization"""
    if not s:
        return ""
    
    # Remove excessive whitespace while preserving structure
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove null bytes and other problematic characters
    s = s.replace('\x00', '').replace('\xa0', ' ')
    
    # Normalize unicode
    s = s.encode('utf-8', 'ignore').decode('utf-8')
    
    # Clean up lines
    lines = []
    for line in s.split("\n"):
        line = line.strip()
        if line and not line.isspace():
            lines.append(line)
    
    return "\n".join(lines)

def read_txt(path):
    """Enhanced text reader with encoding detection"""
    try:
        encoding = detect_encoding(path)
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return normalize_text(f.read())
    except:
        return None

def read_json(path):
    """Enhanced JSON reader with nested structure handling"""
    try:
        encoding = detect_encoding(path)
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            data = json.load(f)
        
        # Enhanced extraction with path preservation
        extracted = extract_json_with_context(data)
        return normalize_text("\n".join(extracted))
    except:
        return None

def extract_json_with_context(obj, path=""):
    """Extract text from JSON while preserving hierarchical context"""
    texts = []
    keys = {"text", "content", "context", "body", "chunk", "page_content", 
            "data", "message", "description", "summary"}
    
    def walk(o, current_path):
        if isinstance(o, dict):
            for k, v in o.items():
                new_path = f"{current_path}.{k}" if current_path else k
                if k.lower() in keys and isinstance(v, str) and v.strip():
                    texts.append(f"[{new_path}]: {v}")
                else:
                    walk(v, new_path)
        elif isinstance(o, list):
            for i, item in enumerate(o):
                walk(item, f"{current_path}[{i}]")
        elif isinstance(o, str) and o.strip():
            texts.append(o)
    
    walk(obj, path)
    return texts

def read_tsv(path):
    """Enhanced TSV reader with column context preservation"""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        
        # Create semantic documents from rows
        texts = []
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() 
                                  if pd.notna(val) and str(val).strip()])
            if row_text:
                texts.append(row_text)
        
        return normalize_text("\n".join(texts))
    except:
        return None

# === ADVANCED TEXT PROCESSING ===
class EnhancedTextProcessor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.bm25_index = None
        self.corpus_for_bm25 = []
        
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities with confidence filtering"""
        doc = nlp(text[:1000000])  # Limit for performance
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW"]:
                entities.append(ent.text)
        return list(set(entities))
    
    def extract_keywords(self, text: str) -> List[str]:
        """Multi-method keyword extraction"""
        keywords = set()
        
        # YAKE keywords
        yake_kws = [kw for kw, _ in kw_extractor.extract_keywords(text)]
        keywords.update(yake_kws[:5])
        
        # KeyBERT keywords
        try:
            keybert_kws = keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english',
                top_n=5
            )
            keywords.update([kw for kw, _ in keybert_kws])
        except:
            pass
        
        return list(keywords)[:10]
    
    def generate_summary(self, text: str) -> str:
        """Generate concise summary using GPT-4"""
        if len(text) < 100:
            return text
        
        try:
            # Truncate to avoid token limits
            truncated = text[:2000]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a 2-sentence summary focusing on key information."},
                    {"role": "user", "content": truncated}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except:
            # Fallback to extractive summary
            sentences = text.split('.')[:2]
            return '. '.join(sentences).strip()
    
    def create_sparse_vector(self, text: str) -> Dict[int, float]:
        """Create sparse vector for hybrid search"""
        try:
            # Use TF-IDF for sparse representation
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Convert to Pinecone sparse vector format
            indices = tfidf_vector.nonzero()[1]
            values = tfidf_vector.data
            
            sparse_dict = {int(idx): float(val) for idx, val in zip(indices, values)}
            return sparse_dict
        except:
            return {}

processor = EnhancedTextProcessor()

# === INTELLIGENT CHUNKING ===
class SemanticChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=self.token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def token_length(self, text: str) -> int:
        return len(encoding.encode(text))
    
    def chunk_with_context(self, text: str, metadata: Dict) -> List[Dict]:
        """Create chunks with enhanced metadata"""
        chunks = self.splitter.split_text(text)
        
        enhanced_chunks = []
        for idx, chunk in enumerate(chunks):
            # Add context from surrounding chunks
            context_before = chunks[idx-1][-100:] if idx > 0 else ""
            context_after = chunks[idx+1][:100] if idx < len(chunks)-1 else ""
            
            enhanced_chunks.append({
                "text": chunk,
                "context": f"{context_before} [CHUNK] {context_after}",
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **metadata
            })
        
        return enhanced_chunks

chunker = SemanticChunker()

# === EMBEDDING GENERATION WITH CACHING ===
class EmbeddingGenerator:
    def __init__(self):
        self.cache_file = os.path.join(config.cache_dir, "embeddings_cache.pkl")
        self.cache = self.load_cache()
    
    def load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        try:
            response = client.embeddings.create(
                model=config.embedding_model,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            self.cache[text_hash] = embedding
            return embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation with retry logic"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            for i in range(0, len(uncached_texts), 20):  # Process in smaller batches
                batch = uncached_texts[i:i+20]
                retries = 3
                
                while retries > 0:
                    try:
                        response = client.embeddings.create(
                            model=config.embedding_model,
                            input=batch,
                            encoding_format="float"
                        )
                        
                        for j, embedding_data in enumerate(response.data):
                            idx = uncached_indices[i+j]
                            embeddings[idx] = embedding_data.embedding
                            # Cache the embedding
                            text_hash = hashlib.md5(texts[idx].encode()).hexdigest()
                            self.cache[text_hash] = embedding_data.embedding
                        break
                    except Exception as e:
                        print(f"Batch embedding error: {e}, retrying...")
                        retries -= 1
                        time.sleep(2)
        
        self.save_cache()
        return embeddings

embedder = EmbeddingGenerator()

# === QUERY EXPANSION AND PROCESSING ===
class QueryProcessor:
    def expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with WordNet synonyms"""
        tokens = nltk.word_tokenize(query.lower())
        expanded_tokens = []
        
        for token in tokens:
            expanded_tokens.append(token)
            synsets = wordnet.synsets(token)
            if synsets:
                # Add first synonym from each synset
                for synset in synsets[:2]:
                    for lemma in synset.lemmas()[:2]:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token:
                            expanded_tokens.append(synonym)
        
        return ' '.join(expanded_tokens)
    
    def reformulate_query(self, query: str) -> List[str]:
        """Generate multiple query perspectives using LLM"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate 3 alternative phrasings of the query. Return only the alternatives, one per line."},
                    {"role": "user", "content": query}
                ],
                max_tokens=150,
                temperature=0.7
            )
            alternatives = response.choices[0].message.content.strip().split('\n')
            return [query] + [alt.strip() for alt in alternatives if alt.strip()][:2]
        except:
            return [query]
    
    def extract_query_intent(self, query: str) -> Dict:
        """Extract intent and key components from query"""
        doc = nlp(query)
        
        return {
            "entities": [ent.text for ent in doc.ents],
            "keywords": processor.extract_keywords(query),
            "expanded": self.expand_query_with_synonyms(query),
            "alternatives": self.reformulate_query(query)
        }

query_processor = QueryProcessor()

# === HYBRID SEARCH WITH RERANKING ===
class HybridSearchEngine:
    def __init__(self):
        self.processor = processor
        self.embedder = embedder
        self.query_processor = query_processor
        self.cross_encoder = cross_encoder
    
    def dense_search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Semantic similarity search"""
        query_embedding = self.embedder.get_embedding(query)
        if not query_embedding:
            return []
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"].get("text", ""),
                "metadata": match["metadata"]
            }
            for match in results.get("matches", [])
        ]
    
    def keyword_search(self, query: str, documents: List[Dict], top_k: int = 30) -> List[Dict]:
        """BM25 keyword search on retrieved documents"""
        if not documents:
            return []
        
        # Create BM25 index from documents
        corpus = [doc["text"] for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Search with expanded query
        expanded_query = self.query_processor.expand_query_with_synonyms(query)
        query_tokens = expanded_query.lower().split()
        
        scores = bm25.get_scores(query_tokens)
        
        # Combine with original scores
        for i, doc in enumerate(documents):
            doc["bm25_score"] = scores[i]
        
        return sorted(documents, key=lambda x: x.get("bm25_score", 0), reverse=True)[:top_k]
    
    def rerank_with_cross_encoder(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank using cross-encoder for better relevance"""
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
    
    def reciprocal_rank_fusion(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """Fuse multiple result lists using RRF"""
        fused_scores = defaultdict(float)
        all_docs = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                all_docs[doc_id] = doc
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        return [all_docs[doc_id] for doc_id in sorted_ids]
    
    def search(self, query: str) -> List[Dict]:
        """Main search pipeline with all optimizations"""
        print(f"Processing query: {query}")
        
        # 1. Query understanding
        query_intent = self.query_processor.extract_query_intent(query)
        
        # 2. Multi-query search
        all_results = []
        
        # Search with original and alternative queries
        for q in query_intent["alternatives"][:2]:
            results = self.dense_search(q, config.initial_retrieval_k)
            all_results.append(results)
        
        # 3. Fusion
        fused_results = self.reciprocal_rank_fusion(all_results)[:config.rerank_k]
        
        # 4. Keyword reranking
        keyword_reranked = self.keyword_search(query, fused_results, config.rerank_k)
        
        # 5. Cross-encoder reranking
        final_results = self.rerank_with_cross_encoder(query, keyword_reranked, config.final_k)
        
        # 6. Add diversity using MMR
        diverse_results = self.mmr_selection(query, final_results, lambda_param=0.7)
        
        return diverse_results
    
    def mmr_selection(self, query: str, documents: List[Dict], lambda_param: float = 0.7) -> List[Dict]:
        """Maximal Marginal Relevance for diversity"""
        if len(documents) <= 1:
            return documents
        
        # Get query embedding
        query_embedding = np.array(self.embedder.get_embedding(query))
        
        # Get document embeddings
        doc_embeddings = []
        for doc in documents:
            emb = self.embedder.get_embedding(doc["text"])
            if emb:
                doc_embeddings.append(np.array(emb))
            else:
                doc_embeddings.append(np.zeros_like(query_embedding))
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Calculate similarity to query
        query_similarities = np.dot(doc_embeddings, query_embedding)
        
        selected = []
        selected_indices = []
        
        # Select first document (highest relevance)
        first_idx = np.argmax(query_similarities)
        selected.append(documents[first_idx])
        selected_indices.append(first_idx)
        
        # Select remaining documents
        while len(selected) < min(len(documents), config.final_k):
            remaining_indices = [i for i in range(len(documents)) if i not in selected_indices]
            
            if not remaining_indices:
                break
            
            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Max similarity to selected documents
                max_sim = 0
                for sel_idx in selected_indices:
                    sim = np.dot(doc_embeddings[idx], doc_embeddings[sel_idx])
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select document with highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(documents[best_idx])
            selected_indices.append(best_idx)
        
        return selected

search_engine = HybridSearchEngine()

# === MAIN PROCESSING PIPELINE ===
def process_file(file_path: str) -> List[Dict]:
    """Process a single file and return chunks"""
    print(f"Processing: {file_path}")
    
    # Read file based on extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        text = read_pdf(file_path)
    elif ext == ".txt":
        text = read_txt(file_path)
    elif ext == ".json":
        text = read_json(file_path)
    elif ext in [".tsv", ".csv"]:
        text = read_tsv(file_path)
    else:
        return []
    
    if not text:
        return []
    
    # Create base metadata
    base_metadata = {
        "source": os.path.relpath(file_path, config.data_folder),
        "file_type": ext[1:],
        "file_size": os.path.getsize(file_path),
        "processing_timestamp": time.time()
    }
    
    # Generate document-level summary
    doc_summary = processor.generate_summary(text[:3000])
    base_metadata["document_summary"] = doc_summary
    
    # Extract document-level entities
    doc_entities = processor.extract_entities(text[:5000])
    base_metadata["document_entities"] = doc_entities[:20]
    
    # Create chunks with context
    chunks = chunker.chunk_with_context(text, base_metadata)
    
    # Process each chunk
    processed_chunks = []
    for chunk_data in chunks:
        chunk_text = chunk_data["text"]
        
        # Skip if too short
        if len(chunk_text) < 50:
            continue
        
        # Generate chunk ID
        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
        chunk_id = f"{os.path.basename(file_path)}-{chunk_hash}"
        
        # Extract chunk-level features
        chunk_entities = processor.extract_entities(chunk_text)
        chunk_keywords = processor.extract_keywords(chunk_text)
        chunk_summary = processor.generate_summary(chunk_text)
        
        # Build chunk metadata
        metadata = {
            **chunk_data,
            "chunk_entities": chunk_entities[:10],
            "chunk_keywords": chunk_keywords[:10],
            "chunk_summary": chunk_summary,
            "token_count": chunker.token_length(chunk_text)
        }
        
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": metadata
        })
    
    return processed_chunks

def upload_to_pinecone(chunks: List[Dict]):
    """Upload chunks to Pinecone with optimizations"""
    if not chunks:
        return 0
    
    # Generate embeddings in batch
    texts = [c["text"] for c in chunks]
    embeddings = embedder.get_embeddings_batch(texts)
    
    # Filter out failed embeddings
    valid_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding:
            valid_chunks.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"]  # Store text in metadata for retrieval
                }
            })
    
    if not valid_chunks:
        return 0
    
    # Upload in batches
    uploaded = 0
    for i in range(0, len(valid_chunks), config.batch_size):
        batch = valid_chunks[i:i+config.batch_size]
        try:
            response = index.upsert(vectors=batch)
            uploaded += len(batch)
            print(f"  Uploaded {len(batch)} vectors (Total: {uploaded}/{len(valid_chunks)})")
        except Exception as e:
            print(f"  Upload error: {e}")
            time.sleep(2)
    
    return uploaded

def process_directory_parallel(directory: str):
    """Process all files in directory using parallel processing"""
    # Collect all files
    files_to_process = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.txt', '.json', '.tsv', '.csv', '.pdf')):
                files_to_process.append(os.path.join(root, file))
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process files in parallel
    all_chunks = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files_to_process}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                print(f"  ✓ Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ Error processing {file_path}: {e}")
    
    # Upload all chunks
    print(f"\nUploading {len(all_chunks)} total chunks to Pinecone...")
    total_uploaded = upload_to_pinecone(all_chunks)
    
    # Save TF-IDF model for sparse search
    if all_chunks:
        corpus = [c["text"] for c in all_chunks]
        processor.tfidf_vectorizer.fit(corpus)
        
        # Save the fitted vectorizer
        import joblib
        joblib.dump(processor.tfidf_vectorizer, os.path.join(config.cache_dir, 'tfidf_model.pkl'))
        print(f"  ✓ Saved TF-IDF model for sparse search")
    
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"  - Files processed: {len(files_to_process)}")
    print(f"  - Chunks created: {len(all_chunks)}")
    print(f"  - Vectors uploaded: {total_uploaded}")
    print(f"{'='*50}")
    
    return total_uploaded


# === MAIN EXECUTION ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG Pipeline")
    parser.add_argument("--mode", choices=["index"], 
                       default="index", help="Operation mode")
    
    args = parser.parse_args()
    
    if args.mode == "index":
        # Process and index documents
        print("Starting document processing and indexing...")
        print(f"Data directory: {config.data_folder}")
        print(f"Using embedding model: {config.embedding_model}")
        print(f"Batch size: {config.batch_size}")
        print("-" * 50)
        
        start_time = time.time()
        total_uploaded = process_directory_parallel(config.data_folder)
        
        print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
        print(f"Embeddings cache saved to: {config.cache_dir}")
        
    
    print("\n✨ Operation completed successfully!")
    
    # Save embedding cache
    embedder.save_cache()