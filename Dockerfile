# Multi-stage lightweight Dockerfile for FinFraud API & Dashboard
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY run_pipeline.py .

EXPOSE 8000 8501

# Default command starts the FastAPI scoring service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
