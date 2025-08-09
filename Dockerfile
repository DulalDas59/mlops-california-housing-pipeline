# Dockerfile

FROM python:3.10-slim

# Make Python output unbuffered so logs flush immediately
ENV PYTHONUNBUFFERED=1 \
    # Where our app writes logs (api/main.py uses this)
    LOG_DIR=/app/logs \
    # Where the model is loaded from (api/main.py uses this)
    MODEL_PATH=/app/models/dt.pkl

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: carry model hash in the image for visibility
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

# Copy source
COPY . .

# Ensure the logs directory exists and is writable
RUN mkdir -p ${LOG_DIR} && \
    chmod -R 755 ${LOG_DIR}

# (Optional but recommended) Add a simple healthcheck hitting /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

EXPOSE 8000

# Use a single worker to avoid SQLite write locks; enable access logs
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--access-log"]
