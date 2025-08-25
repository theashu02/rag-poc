# 1. Start with an official Python base image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download and cache the ML models during the build process
RUN python -m spacy download en_core_web_lg
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 5. Copy your application code into the container
COPY main.py .
COPY rag_pipeline.py .

# 6. Expose the port the app will run on
EXPOSE 8080

# 7. Start the Functions Framework for CloudEvents
CMD ["functions-framework", "--target=handle_gcs_event", "--port=8080"]