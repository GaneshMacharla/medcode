# MedCodeRL — Root Dockerfile for HF Spaces & OpenEnv validation
#
# Build:  docker build -t medcoderl .
# Run:    docker run -p 7680:7680 medcoderl

FROM python:3.11-slim

WORKDIR /app

# Ensure stdout/stderr are unbuffered for real-time logging
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (root requirements first for layer caching)
COPY requirements.txt /app/requirements.txt
COPY server/requirements.txt /app/server_requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/server_requirements.txt

# Copy all source code
COPY . /app/

# Ensure the package is importable
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7680/health || exit 1

EXPOSE 7680

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7680"]
