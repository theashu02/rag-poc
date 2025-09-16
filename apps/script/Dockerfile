FROM python:3.12-slim

# -------------------------------------------------
# Set working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# Install Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# -------------------------------------------------
# Download NLTK corpora during build
# -------------------------------------------------
RUN python - <<'PY'
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
PY

# -------------------------------------------------
# Prepare Hugging Face cache (but stay ONLINE for now)
# -------------------------------------------------
RUN mkdir -p /models/hf-cache
ENV HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/models/hf-cache

# -------------------------------------------------
# Download all required models while network is allowed
# -------------------------------------------------
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
from keybert import KeyBERT

SentenceTransformer("all-MiniLM-L6-v2")                     # encoder weights + configs
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")       # cross-encoder weights
KeyBERT(model="all-MiniLM-L6-v2")                           # KeyBERT cache priming
PY

# -------------------------------------------------
# Now lock the image into completely offline mode
# -------------------------------------------------
ENV HF_HUB_OFFLINE=1

# -------------------------------------------------
# Copy application source
# -------------------------------------------------
COPY main.py .
COPY rag_pipeline.py .

# -------------------------------------------------
# Expose port & start service
# -------------------------------------------------
EXPOSE 8080
CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]