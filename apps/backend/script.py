import os
import json
import time
import hashlib
import pandas as pd
import spacy
import yake
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX")
data_folder = os.getenv("DATA_DIR")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")  # higher accuracy
batch_size = int(os.getenv("BATCH_SIZE", "100"))
start_batch = int(os.getenv("START_BATCH", "0"))

# === INIT ===
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
nlp = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor(n=2, top=5)
encoding = tiktoken.encoding_for_model(embedding_model)

# Create index if not exists
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072 if "large" in embedding_model else 1536,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pc.Index(index_name)

# === HELPERS ===
def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in s.split("\n")]
    return "\n".join([ln for ln in lines if ln])

def read_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_text(f.read())
    except:
        return None

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return normalize_text("\n".join(extract_strings_from_json(data)))
    except:
        return None

def extract_strings_from_json(obj):
    keys = {"text", "content", "context", "body", "chunk", "page_content", "data", "message"}
    preferred, others = [], []
    def walk(o, parent=None):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for i in o:
                walk(i, parent)
        elif isinstance(o, str) and o.strip():
            (preferred if parent and parent.lower() in keys else others).append(o)
    walk(obj)
    return preferred if preferred else others

def read_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        text_data = " ".join(df.fillna("").astype(str).agg(" ".join, axis=1))
        return normalize_text(text_data)
    except:
        return None

def iter_data_files(root):
    exts = (".txt", ".json", ".tsv")
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                yield os.path.join(dirpath, name)

def get_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents))

def get_keywords(text):
    return [kw for kw, score in kw_extractor.extract_keywords(text)]

def summarize_text(text):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Summarize in 1-2 sentences."},
                      {"role": "user", "content": text}],
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except:
        return ""

def get_token_length(text):
    return len(encoding.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=get_token_length
)

def get_embeddings_batch(texts):
    try:
        resp = client.embeddings.create(
            model=embedding_model,
            input=texts,
            encoding_format="float"
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# === MAIN PIPELINE ===
seen_hashes = set()
all_chunks = []
processed_files = 0
uploaded_vectors = 0

for file_path in iter_data_files(data_folder):
    if file_path.lower().endswith(".txt"):
        text = read_txt(file_path)
    elif file_path.lower().endswith(".json"):
        text = read_json(file_path)
    else:
        text = read_tsv(file_path)

    if not text:
        continue

    processed_files += 1
    chunks = splitter.split_text(text)

    for idx, chunk in enumerate(chunks):
        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        entities = get_entities(chunk)
        keywords = get_keywords(chunk)
        summary = summarize_text(chunk)

        all_chunks.append({
            "id": f"{os.path.basename(file_path)}-{h[:12]}",
            "text": chunk,
            "metadata": {
                "source": os.path.relpath(file_path, data_folder),
                "chunk_index": idx,
                "file_type": os.path.splitext(file_path)[1][1:],
                "chunk_length": get_token_length(chunk),
                "entities": entities,
                "keywords": keywords,
                "summary": summary
            }
        })

    # Stream upload in batches to save memory
    while len(all_chunks) >= batch_size:
        batch = all_chunks[:batch_size]
        embeddings = get_embeddings_batch([c["text"] for c in batch])
        if embeddings:
            resp = index.upsert(vectors=[
                {"id": c["id"], "values": e, "metadata": {**c["metadata"], "text": c["text"]}}
                for c, e in zip(batch, embeddings)
            ])
            uploaded_count = getattr(resp, "upserted_count", len(batch))
            uploaded_vectors += uploaded_count
            print(f"✅ Uploaded {uploaded_count} vectors. Total: {uploaded_vectors}")
        all_chunks = all_chunks[batch_size:]

# Upload any remaining chunks
print(f"Processed {processed_files} files from {data_folder}")
if all_chunks:
    embeddings = get_embeddings_batch([c["text"] for c in all_chunks])
    if embeddings:
        resp = index.upsert(vectors=[
            {"id": c["id"], "values": e, "metadata": {**c["metadata"], "text": c["text"]}}
            for c, e in zip(all_chunks, embeddings)
        ])
        uploaded_count = getattr(resp, "upserted_count", len(all_chunks))
        uploaded_vectors += uploaded_count
        print(f"✅ Uploaded {uploaded_count} vectors. Total: {uploaded_vectors}")

print(f"✅ Processed {processed_files} files into Pinecone.")

# import os
# import json
# import time
# import hashlib
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from openai import OpenAI
# from pinecone import Pinecone
# from dotenv import load_dotenv

# load_dotenv()

# # Configuration
# openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
# index_name = os.getenv("PINECONE_INDEX", "YOUR_INDEX_NAME")
# data_folder = os.getenv("DATA_DIR", r"YOUR_LOCAL_DATASET_DIRECTORY_PATH")
# embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
# pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # Default to us-east-1

# # Resume Support
# start_batch = int(os.getenv("START_BATCH", "0"))  # Can be overridden via env
# batch_size = int(os.getenv("BATCH_SIZE", "100"))  # Batch embeddings to reduce API calls

# # Initialize Clients
# client = OpenAI(api_key=openai_api_key)
# pc = Pinecone(api_key=pinecone_api_key)

# # Check if index exists, create if it doesn't
# try:
#     index_list = [idx.name for idx in pc.list_indexes()]
#     if index_name not in index_list:
#         print(f"Creating index '{index_name}' with 1536 dimensions...")
#         pc.create_index(
#             name=index_name,
#             dimension=1536,  # text-embedding-3-small dimension
#             metric="cosine",
#             cloud="aws",
#             region="us-east-1",
#         )
#         # Wait for index to be ready
#         while not pc.describe_index(index_name).status['ready']:
#             print("Waiting for index to be ready...")
#             time.sleep(1)
#         print(f"✅ Index '{index_name}' created successfully!")
#     else:
#         print(f"✅ Using existing index '{index_name}'")
# except Exception as e:
#     print(f"❌ Error with index setup: {e}")
#     exit(1)

# index = pc.Index(index_name)

# # -------- Helpers --------
# def normalize_text(s: str) -> str:
#     """Normalize text by cleaning line breaks and removing empty lines"""
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     lines = [line.strip() for line in s.split("\n")]
#     # Remove empty lines to compact chunks
#     non_empty = [ln for ln in lines if ln]
#     return "\n".join(non_empty)

# def read_txt(file_path: str) -> str | None:
#     """Read and normalize text from a .txt file"""
#     try:
#         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#             text = f.read()
#         text = normalize_text(text)
#         return text if text else None
#     except Exception as e:
#         print(f"⚠️ Failed to read TXT {file_path}: {e}")
#         return None

# def extract_strings_from_json(obj) -> list[str]:
#     """Extract text content from JSON, prioritizing content-related keys"""
#     preferred_keys = {"text", "content", "context", "body", "chunk", "page_content", "data", "message"}
#     preferred, others = [], []

#     def walk(o, parent_key=None):
#         if isinstance(o, dict):
#             for k, v in o.items():
#                 walk(v, k)
#         elif isinstance(o, list):
#             for item in o:
#                 walk(item, parent_key)
#         elif isinstance(o, str) and len(o.strip()) > 0:  # Only non-empty strings
#             if parent_key and parent_key.lower() in preferred_keys:
#                 preferred.append(o)
#             else:
#                 others.append(o)

#     walk(obj)
#     return preferred if preferred else others

# def read_json(file_path: str) -> str | None:
#     """Read and extract text content from a JSON file"""
#     try:
#         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#             data = json.load(f)
#         strings = extract_strings_from_json(data)
#         if not strings:
#             return None
#         text = normalize_text("\n".join(strings))
#         return text if text else None
#     except Exception as e:
#         print(f"⚠️ Failed to read JSON {file_path}: {e}")
#         return None

# def iter_data_files(root_dir: str):
#     """Iterator for all .txt and .json files in the directory tree"""
#     supported_extensions = ('.txt', '.json')
#     for dirpath, _, filenames in os.walk(root_dir):
#         for name in filenames:
#             if name.lower().endswith(supported_extensions):
#                 yield os.path.join(dirpath, name)

# # Validate data directory
# if not os.path.exists(data_folder):
#     print(f"❌ Data directory '{data_folder}' does not exist!")
#     exit(1)

# # Chunk Preparation with updated LangChain text splitter
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,      # slightly larger chunks for better throughput
#     chunk_overlap=100,   # moderate overlap for context continuity
#     length_function=len,
#     is_separator_regex=False,
# )

# all_chunks = []
# seen_hashes = set()
# processed_files = 0

# print(f"🔍 Scanning files in {data_folder}...")

# for file_path in iter_data_files(data_folder):
#     try:
#         if file_path.lower().endswith(".txt"):
#             text = read_txt(file_path)
#         else:
#             text = read_json(file_path)

#         if not text:
#             continue

#         processed_files += 1
#         splits = splitter.split_text(text)
        
#         for i, chunk in enumerate(splits):
#             chunk = chunk.strip()
#             if not chunk:
#                 continue
            
#             h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
#             if h in seen_hashes:
#                 continue
            
#             seen_hashes.add(h)
#             chunk_id = f"{os.path.basename(file_path)}-{h[:12]}"
            
#             all_chunks.append({
#                 "id": chunk_id,
#                 "text": chunk,
#                 "metadata": {
#                     "source": os.path.relpath(file_path, data_folder).replace("\\", "/"),
#                     "hash": h,
#                     "chunk_index": i,
#                     "file_type": "txt" if file_path.lower().endswith(".txt") else "json",
#                     "chunk_length": len(chunk)
#                 }
#             })
            
#     except Exception as e:
#         print(f"⚠️ Skipping {file_path}: {e}")

# if not all_chunks:
#     print("⚠️ No chunks prepared. Check your data directory and file formats.")
#     print(f"Processed {processed_files} files from {data_folder}")
#     exit(1)
# else:
#     print(f"✅ Prepared {len(all_chunks)} unique chunks from {processed_files} files")

# # -------- Embeddings (batched) --------
# def get_embeddings_batch(texts: list[str], retries: int = 3, delay: int = 5) -> list | None:
#     """Generate embeddings for a batch of texts with retry logic"""
#     for attempt in range(retries):
#         try:
#             resp = client.embeddings.create(
#                 model=embedding_model,
#                 input=texts,
#                 encoding_format="float"
#             )
#             # Align by index
#             return [d.embedding for d in resp.data]
#         except Exception as e:
#             print(f"⚠️ Embedding batch failed (attempt {attempt + 1}): {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#                 delay *= 2  # Exponential backoff
#             else:
#                 print("❌ Giving up on this embedding batch.")
#     return None

# # -------- Upload to Pinecone in Batches with Resume Support --------
# total_batches = (len(all_chunks) + batch_size - 1) // batch_size
# uploaded_vectors = 0

# print(f"🚀 Starting upload of {len(all_chunks)} chunks in {total_batches} batches...")
# print(f"📊 Starting from batch {start_batch + 1}")

# for i in range(start_batch * batch_size, len(all_chunks), batch_size):
#     batch_index = i // batch_size
#     batch = all_chunks[i:i + batch_size]
    
#     print(f"📤 Uploading batch {batch_index + 1} / {total_batches} (size={len(batch)})")

#     # Get embeddings for this batch
#     texts = [item["text"] for item in batch]
#     embeddings = get_embeddings_batch(texts)

#     if embeddings is None:
#         print(f"❌ Skipping upload for batch {batch_index + 1} due to embedding failure.")
#         print(f"❗ To resume, set START_BATCH={batch_index} in your .env file")
#         break

#     # Prepare vectors for Pinecone
#     vectors = []
#     for item, emb in zip(batch, embeddings):
#         if emb is None:
#             continue
#         vectors.append({
#             "id": item["id"],
#             "values": emb,
#             "metadata": {**item["metadata"], "text": item["text"]}
#         })

#     if not vectors:
#         print("⚠️ No vectors in this batch to upload.")
#         continue

#     # Upload to Pinecone
#     try:
#         upsert_response = index.upsert(vectors=vectors)
#         uploaded_count = upsert_response.upserted_count if hasattr(upsert_response, 'upserted_count') else len(vectors)
#         uploaded_vectors += uploaded_count
#         print(f"✅ Uploaded {uploaded_count} vectors. Total: {uploaded_vectors}")
        
#     except Exception as e:
#         print(f"❌ Failed to upload batch {batch_index + 1}: {e}")
#         print(f"❗ To resume, set START_BATCH={batch_index} in your .env file")
#         break

#     # Rate limiting - be gentle with APIs
#     time.sleep(0.5)

# # Final stats
# try:
#     stats = index.describe_index_stats()
#     total_vectors = stats.total_vector_count if hasattr(stats, 'total_vector_count') else "unknown"
#     print(f"📊 Index now contains {total_vectors} total vectors")
# except:
#     print("📊 Could not retrieve final index statistics")

# print("✅ Script execution completed!")


# some updation like token based chunking instead of char based
# Handles .txt, .json, .tsv.
# Uses token-based chunking instead of character-based.
# Runs NER, keyword extraction, and summarization per chunk.
# Stores rich metadata for hybrid retrieval in Pinecone.
# Works in streaming mode for 20GB datasets so you don’t blow RAM.
# Supports resume after failure.
# import os
# import json
# import time
# import hashlib
# import pandas as pd
# import spacy
# import yake
# import tiktoken
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from openai import OpenAI
# from pinecone import Pinecone
# from dotenv import load_dotenv

# load_dotenv()

# # === CONFIG ===
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# index_name = os.getenv("PINECONE_INDEX")
# data_folder = os.getenv("DATA_DIR")
# embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")  # higher accuracy
# batch_size = int(os.getenv("BATCH_SIZE", "100"))
# start_batch = int(os.getenv("START_BATCH", "0"))

# # === INIT ===
# client = OpenAI(api_key=openai_api_key)
# pc = Pinecone(api_key=pinecone_api_key)
# nlp = spacy.load("en_core_web_sm")
# kw_extractor = yake.KeywordExtractor(n=2, top=5)
# encoding = tiktoken.encoding_for_model(embedding_model)

# # Create index if not exists
# if index_name not in [idx.name for idx in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=3072 if "large" in embedding_model else 1536,
#         metric="cosine",
#         cloud="aws",
#         region="us-east-1",
#     )
#     while not pc.describe_index(index_name).status['ready']:
#         time.sleep(1)
# index = pc.Index(index_name)


# # === HELPERS ===
# def normalize_text(s: str) -> str:
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     lines = [line.strip() for line in s.split("\n")]
#     return "\n".join([ln for ln in lines if ln])


# def read_txt(path):
#     try:
#         with open(path, "r", encoding="utf-8", errors="ignore") as f:
#             return normalize_text(f.read())
#     except:
#         return None


# def read_json(path):
#     try:
#         with open(path, "r", encoding="utf-8", errors="ignore") as f:
#             data = json.load(f)
#         return normalize_text("\n".join(extract_strings_from_json(data)))
#     except:
#         return None


# def extract_strings_from_json(obj):
#     keys = {"text", "content", "context", "body", "chunk", "page_content", "data", "message"}
#     preferred, others = [], []
#     def walk(o, parent=None):
#         if isinstance(o, dict):
#             for k, v in o.items():
#                 walk(v, k)
#         elif isinstance(o, list):
#             for i in o:
#                 walk(i, parent)
#         elif isinstance(o, str) and o.strip():
#             (preferred if parent and parent.lower() in keys else others).append(o)
#     walk(obj)
#     return preferred if preferred else others


# def read_tsv(path):
#     try:
#         df = pd.read_csv(path, sep="\t", dtype=str)
#         text_data = " ".join(df.fillna("").astype(str).agg(" ".join, axis=1))
#         return normalize_text(text_data)
#     except:
#         return None


# def iter_data_files(root):
#     exts = (".txt", ".json", ".tsv")
#     for dirpath, _, filenames in os.walk(root):
#         for name in filenames:
#             if name.lower().endswith(exts):
#                 yield os.path.join(dirpath, name)


# def get_entities(text):
#     doc = nlp(text)
#     return list(set(ent.text for ent in doc.ents))


# def get_keywords(text):
#     return [kw for kw, score in kw_extractor.extract_keywords(text)]


# def summarize_text(text):
#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Summarize in 1-2 sentences."},
#                       {"role": "user", "content": text}],
#             max_tokens=80,
#         )
#         return resp.choices[0].message.content.strip()
#     except:
#         return ""


# def get_token_length(text):
#     return len(encoding.encode(text))


# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=100,
#     length_function=get_token_length
# )


# def get_embeddings_batch(texts):
#     try:
#         resp = client.embeddings.create(
#             model=embedding_model,
#             input=texts,
#             encoding_format="float"
#         )
#         return [d.embedding for d in resp.data]
#     except Exception as e:
#         print(f"Embedding error: {e}")
#         return None


# # === MAIN PIPELINE ===
# seen_hashes = set()
# all_chunks = []
# file_count = 0

# for file_path in iter_data_files(data_folder):
#     if file_path.lower().endswith(".txt"):
#         text = read_txt(file_path)
#     elif file_path.lower().endswith(".json"):
#         text = read_json(file_path)
#     else:
#         text = read_tsv(file_path)

#     if not text:
#         continue

#     file_count += 1
#     chunks = splitter.split_text(text)

#     for idx, chunk in enumerate(chunks):
#         h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
#         if h in seen_hashes:
#             continue
#         seen_hashes.add(h)

#         entities = get_entities(chunk)
#         keywords = get_keywords(chunk)
#         summary = summarize_text(chunk)

#         all_chunks.append({
#             "id": f"{os.path.basename(file_path)}-{h[:12]}",
#             "text": chunk,
#             "metadata": {
#                 "source": os.path.relpath(file_path, data_folder),
#                 "chunk_index": idx,
#                 "file_type": os.path.splitext(file_path)[1][1:],
#                 "chunk_length": get_token_length(chunk),
#                 "entities": entities,
#                 "keywords": keywords,
#                 "summary": summary
#             }
#         })

#     # Stream upload in batches to save memory
#     while len(all_chunks) >= batch_size:
#         batch = all_chunks[:batch_size]
#         embeddings = get_embeddings_batch([c["text"] for c in batch])
#         if embeddings:
#             index.upsert(vectors=[
#                 {"id": c["id"], "values": e, "metadata": {**c["metadata"], "text": c["text"]}}
#                 for c, e in zip(batch, embeddings)
#             ])
#         all_chunks = all_chunks[batch_size:]

# # Upload any remaining chunks
# if all_chunks:
#     embeddings = get_embeddings_batch([c["text"] for c in all_chunks])
#     if embeddings:
#         index.upsert(vectors=[
#             {"id": c["id"], "values": e, "metadata": {**c["metadata"], "text": c["text"]}}
#             for c, e in zip(all_chunks, embeddings)
#         ])

# print(f"✅ Processed {file_count} files into Pinecone.")