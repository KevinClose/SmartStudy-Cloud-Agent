# Base image: Python 3.11 slim, keeps the image small (~150 MB)
# We don't use Python 3.13 because not all libs ship wheels for it yet on Linux.
FROM python:3.11-slim

# System: install only what's strictly needed
# build-essential is needed by some pip wheels that compile C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY app.py .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Start Streamlit:
#   - bind to all interfaces so Cloud Run can reach it
#   - on the PORT injected by Cloud Run
#   - disable usage stats and CORS for cleaner production behavior
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false