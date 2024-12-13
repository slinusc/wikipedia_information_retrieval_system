# Use an official Python image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install FAISS, FastAPI, and Python dependencies
RUN pip install --no-cache-dir faiss-cpu sentence-transformers numpy fastapi uvicorn

# Create a working directory
WORKDIR /app

# Copy the FAISS index into the container
COPY wikipedia_202307.index /app/wikipedia_202307.index

# Copy the app script into the container
COPY app.py /app/app.py

# Default command to run the FastAPI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
