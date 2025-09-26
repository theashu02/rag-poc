FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl

# Download NLTK data
RUN python - <<'PY'
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
PY

RUN mkdir -p /models/hf-cache
ENV HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/models/hf-cache

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
from keybert import KeyBERT

SentenceTransformer("all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
KeyBERT(model="all-MiniLM-L6-v2")
PY

ENV HF_HUB_OFFLINE=1

COPY main.py .
COPY rag_pipeline.py .

EXPOSE 8080
CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]
