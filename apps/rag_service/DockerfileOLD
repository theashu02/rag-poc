FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        poppler-utils \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Preinstall spaCy model (fallback to small model if large is unavailable)
RUN pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl \
    || pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl

ENV HF_HOME=/workspace/models/hf-cache \
    TRANSFORMERS_CACHE=/workspace/models/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/workspace/models/hf-cache
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

COPY main.py ./
COPY services/ ./services/

EXPOSE 8080
ENV PORT=8080

CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]
