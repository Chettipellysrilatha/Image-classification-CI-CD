# Use official Python image
FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Avoid overly chatty TensorFlow logs
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Default command to run training script
CMD ["python", "train_mnist.py"]
