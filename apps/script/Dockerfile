FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_lg
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

COPY main.py .
COPY rag_pipeline.py .

EXPOSE 8080

CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]