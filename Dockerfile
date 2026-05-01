# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssl \
    curl \
  && rm -rf /var/lib/apt/lists/* \
  && update-ca-certificates \
  && python -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

COPY . .

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
