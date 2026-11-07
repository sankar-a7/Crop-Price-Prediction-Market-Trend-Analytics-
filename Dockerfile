# Production Container Image for Crop Price Prediction System
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and artifacts
COPY . .

# Generate dataset and train models if not already pre-baked
RUN python generate_dataset.py && python train.py

# Expose Streamlit dashboard (8501) and Flask REST API (5000)
EXPOSE 8501 5000

# Default command starts Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

