FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 sentence-transformers
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

COPY . .

# Build the vector DB (and cache the embedding model) at image build time
RUN python ingest.py

EXPOSE 8000
# Hosts like Render set $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]