#!/usr/bin/env python3
"""
Optimised RAG indexing pipeline.

• Reads PDF / TXT files (easy to extend for CSV / JSON …)
• Cleans & semantic-chunks text with context windows
• Enriches every chunk with entities, keywords, summaries, token-count
• Generates dense OpenAI embeddings  (cached) + sparse TF-IDF vectors
• Uploads to Pinecone under a user-supplied namespace
• Retry logic & progress output
"""

import os, time, json, hashlib, argparse
from dataclasses import dataclass
from typing import List, Dict

import pdfplumber, PyPDF2, spacy, yake, tiktoken, nltk
from keybert import KeyBERT
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, Index

# ───────────────────────────────────────── Config
@dataclass
class RAGConfig:
    openai_api_key : str = os.getenv("OPENAI_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    index_name     : str = os.getenv("PINECONE_INDEX")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    keyword_model  : str = "all-MiniLM-L6-v2"
    reranker_model : str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    chunk_size     : int = 800
    chunk_overlap  : int = 150
    batch_size     : int = 50
    cache_dir      : str = "/tmp/cache"

cfg = RAGConfig()
os.makedirs(cfg.cache_dir, exist_ok=True)

# ────────────────────────────────── Model / service init
print("⏳  Loading models …")
nlp            = spacy.load("en_core_web_lg")
kw_extractor   = yake.KeywordExtractor(n=3, top=10, dedupLim=0.7)
keybert_model  = KeyBERT(model=cfg.keyword_model)
cross_encoder  = CrossEncoder(cfg.reranker_model)

encoding       = tiktoken.encoding_for_model("gpt-4")
client         = OpenAI(api_key=cfg.openai_api_key)
pc             = Pinecone(api_key=cfg.pinecone_api_key)

# ────────────────────────────────── Pinecone index
def setup_index() -> Index:
    if cfg.index_name not in pc.list_indexes().names():
        print(f"📗 Creating Pinecone index '{cfg.index_name}' …")
        pc.create_index(
            name      = cfg.index_name,
            dimension = 1536,
            metric    = "cosine",
            spec      = { "pod": { "environment": "gcp-starter" } }
        )
        while not pc.describe_index(cfg.index_name).status['ready']:
            time.sleep(1)
    return pc.Index(cfg.index_name)

index = setup_index()

# ────────────────────────────────── Utilities
tfidf_vectorizer = TfidfVectorizer(max_features=1_000, stop_words='english')

def make_sparse(text:str)->Dict[int,float]:
    vec = tfidf_vectorizer.transform([text])
    if vec.nnz == 0: return {}
    idx, vals = vec.indices, vec.data
    return { int(i): float(v) for i,v in zip(idx, vals) }

def normalize(txt:str)->str:
    txt = txt.replace("\r\n","\n").replace("\r","\n")
    txt = txt.replace('\x00','').replace('\xa0',' ')
    return "\n".join(line.strip() for line in txt.splitlines() if line.strip())

def read_pdf(path:str)->str|None:
    try:
        with pdfplumber.open(path) as pdf:
            return normalize("\n".join(p.extract_text() or '' for p in pdf.pages))
    except Exception:
        try:
            with open(path,'rb') as f:
                return normalize("\n".join(p.extract_text() or '' for p in PyPDF2.PdfReader(f).pages))
        except Exception:
            return None

def read_txt(path:str)->str|None:
    try:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            return normalize(f.read())
    except Exception:
        return None

splitter = RecursiveCharacterTextSplitter(
    chunk_size = cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
    length_function = lambda t: len(encoding.encode(t)),
    separators = ["\n\n", "\n", ". ", " ", ""]
)

# ────────────────────────────────── Embedding cache
EMB_CACHE_FILE = os.path.join(cfg.cache_dir,"embeddings.pkl")
EMB_CACHE: Dict[str,List[float]] = {}
if os.path.exists(EMB_CACHE_FILE):
    import pickle; EMB_CACHE.update(pickle.load(open(EMB_CACHE_FILE,'rb')))

def batch_embed(texts:List[str])->List[List[float]|None]:
    out=[None]*len(texts)
    uncached, idx = [], []
    for i,t in enumerate(texts):
        h=hashlib.md5(t.encode()).hexdigest()
        if h in EMB_CACHE: out[i]=EMB_CACHE[h]
        else: uncached.append(t); idx.append(i)

    for start in range(0,len(uncached),20):
        batch = uncached[start:start+20]; retries=3
        while retries:
            try:
                resp = client.embeddings.create(
                    model=cfg.embedding_model, input=batch, encoding_format="float")
                for j,d in enumerate(resp.data):
                    out_idx = idx[start+j]
                    out[out_idx]=d.embedding
                    EMB_CACHE[hashlib.md5(batch[j].encode()).hexdigest()] = d.embedding
                break
            except Exception as e:
                print("⚠️ embed retry:",e); retries-=1; time.sleep(2)
    return out

def save_cache():
    import pickle; pickle.dump(EMB_CACHE, open(EMB_CACHE_FILE,'wb'))

# ────────────────────────────────── Processing
def extract_keywords(text:str)->List[str]:
    kws = [kw for kw,_ in kw_extractor.extract_keywords(text)][:5]
    try:
        kb = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1,3),
                                            stop_words='english', top_n=5)
        kws.extend([kw for kw,_ in kb])
    except Exception: pass
    return list(set(kws))[:10]

def extract_entities(text:str)->List[str]:
    doc = nlp(text[:1_000_000])
    ents = [e.text for e in doc.ents if e.label_ in
           ("PERSON","ORG","GPE","PRODUCT","EVENT","LAW")]
    return list(set(ents))[:10]

def gpt_summary(text:str)->str:
    if len(text)<100: return text
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Give a 2-sentence summary."},
                {"role":"user","content":text[:2000]}
            ],
            max_tokens=100, temperature=0.3)
        return resp.choices[0].message.content.strip()
    except Exception:
        return '. '.join(text.split('.')[:2]).strip()

def chunk_document(text:str, meta:Dict)->List[Dict]:
    chunks = splitter.split_text(text)
    out=[]
    for i,chunk in enumerate(chunks):
        if len(chunk)<50: continue
        ctx_before = chunks[i-1][-100:] if i>0 else ""
        ctx_after  = chunks[i+1][:100]  if i+1<len(chunks) else ""
        chunk_id = f"{meta['source']}-{hashlib.sha256(chunk.encode()).hexdigest()[:12]}"
        out.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                **meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_keywords": extract_keywords(chunk),
                "chunk_entities": extract_entities(chunk),
                "chunk_summary": gpt_summary(chunk),
                "context_before": ctx_before,
                "context_after": ctx_after,
                "token_count": len(encoding.encode(chunk))
            }
        })
    return out

def process_file(path:str)->List[Dict]:
    ext=os.path.splitext(path)[1].lower()
    text = read_pdf(path) if ext==".pdf" else read_txt(path)
    if not text: return []

    title = next((ln for ln in text.splitlines() if ln.strip()), os.path.basename(path))[:120]
    base_meta={
        "source": os.path.basename(path),
        "file_type": ext[1:],
        "title": title,
        "doc_summary": gpt_summary(text[:3000]),
        "ts": time.time()
    }
    return chunk_document(text, base_meta)

def upload_chunks(chunks:List[Dict], ns:str)->int:
    if not chunks: return 0
    dense = batch_embed([c["text"] for c in chunks])
    vectors=[]
    for c,e in zip(chunks, dense):
        if not e: continue
        sparse = make_sparse(c["text"])
        vectors.append({
            "id": c["id"],
            "values": e,
            "sparse_values":{
                "indices": list(sparse.keys()),
                "values": list(sparse.values())
            },
            "metadata": { **c["metadata"], "text": c["text"] }
        })
    uploaded=0
    for i in range(0,len(vectors),cfg.batch_size):
        try:
            index.upsert(vectors=vectors[i:i+cfg.batch_size], namespace=ns)
            uploaded+=len(vectors[i:i+cfg.batch_size])
        except Exception as e:
            print("⚠️  upsert retry:",e); time.sleep(2)
    return uploaded

def walk(folder:str)->List[str]:
    return [os.path.join(r,f)
            for r,_,fs in os.walk(folder)
            for f in fs if f.lower().endswith(('.pdf','.txt'))]

def main(data_dir:str, namespace:str):
    files = walk(data_dir)
    total_chunks=total_vecs=0
    for p in files:
        ch = process_file(p)
        total_chunks+=len(ch)
        total_vecs += upload_chunks(ch, namespace)
        print(f"✓ {os.path.basename(p):35}  chunks:{len(ch):3}  totalVec:{total_vecs}")
    save_cache()
    print(f"\n🎉 Done. Files:{len(files)}  Chunks:{total_chunks}  Vectors:{total_vecs}")

# ────────────────────────────────── CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimised RAG pipeline")
    ap.add_argument("--data",      default="./data", help="Directory with docs")
    ap.add_argument("--namespace", required=True,   help="Pinecone namespace / userID")
    args = ap.parse_args()
    main(args.data, args.namespace)