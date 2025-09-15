FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_lg \
 && python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

ENV HF_HOME=/root/.cache/huggingface

RUN python -c "from keybert import KeyBERT; KeyBERT(model='all-MiniLM-L6-v2')"

RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"

COPY main.py .
COPY rag_pipeline.py .

EXPOSE 8080

CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]
