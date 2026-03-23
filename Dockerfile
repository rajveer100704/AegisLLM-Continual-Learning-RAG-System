# Use a lightweight python image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for FAISS and Redis
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV PYTHONPATH=.
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command (API)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
