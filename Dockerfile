# syntax=docker/dockerfile:1.7

# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (improves caching)
COPY requirements.txt .

# Install OS deps, then Python deps with a persistent pip cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    # build deps (keep if you need to compile wheels) \
    g++ cmake libffi-dev libssl-dev \
    # TLS + certs \
    ca-certificates openssl \
    # network/debug tools \
    curl netcat-openbsd iputils-ping dnsutils iproute2 procps \
    # optional niceties \
    jq traceroute \
    # basic utilities you already like \
    wget nano \
  && rm -rf /var/lib/apt/lists/* \
  && update-ca-certificates \
  && python -m pip install --upgrade pip


# Python deps with pip cache (BuildKit)  ← the --mount must be here
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt 'huggingface_hub[hf_xet]'

# NLTK data (network fetch at build time)
RUN python -m nltk.downloader stopwords

# Copy the rest of the project files
COPY . .

# Expose FastAPI port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
