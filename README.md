# EU FarmBook OpenSearch API

A FastAPI-based semantic search and recommendation service powered by OpenSearch neural search capabilities. This API provides hybrid search (BM25 + neural embeddings), k-NN recommendations, and AI-powered result summarization for the EU FarmBook knowledge platform.

## Overview

This service enables intelligent document search across agricultural knowledge objects (KOs) by combining:

- **Neural Semantic Search** - Vector similarity using sentence embeddings (MS MARCO, MPNet, MiniLM)
- **BM25 Keyword Matching** - Traditional text retrieval for exact term matching
- **Hybrid Search** - Score normalization and blending of both approaches
- **k-NN Recommendations** - Content-based similar document suggestions
- **AI Summarization** - LLM-powered result summaries via Hugging Face
- **Multi-language Support** - Automatic query translation (DeepL)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Access to an OpenSearch cluster with neural search plugins

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader stopwords

# Configure environment
cp .env.example .env
# Edit .env with your OpenSearch credentials and API keys

# Run the application
uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up -d

# Or build and push to registry
./deploy.sh
```

## API Endpoints

### Search Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/neural_search_relevant` | POST | Hybrid semantic search with explicit query control |
| `/neural_search_relevant_hybrid` | POST | Hybrid search with BM25-neural score normalization |
| `/neural_search_relevant_sparse` | POST | Neural sparse search (learned sparse retrieval) |
| `/neural_search_relevant_new` | POST | Context-aware neural search with smart fallback |

### Recommendation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend_similar_knn` | POST | k-NN semantic recommendations for a document |

### Request/Response Format

All search endpoints accept JSON payloads with the following common fields:

```json
{
  "search_term": "sustainable farming practices",
  "topics": ["agriculture", "sustainability"],
  "themes": ["environment"],
  "languages": ["en"],
  "category": ["research"],
  "project_type": ["H2020"],
  "project_acronym": ["EIP-AGRI"],
  "locations": ["EU"],
  "page": 1,
  "dev": true,
  "model": "msmarco",
  "include_fulltext": false,
  "include_summary": false,
  "sort_by": "score_desc",
  "access_token": "..."
}
```

Response structure:

```json
{
  "data": [...],
  "related_projects_from_this_page": [...],
  "related_projects_from_entire_resultset": [...],
  "pagination": {
    "total_records": 100,
    "current_page": 1,
    "total_pages": 10,
    "next_page": 2,
    "prev_page": null
  }
}
```

## Features

### Search Capabilities

- **Multi-field Neural Search** - Searches across title, subtitle, description, keywords, and content embeddings
- **Smart Query Detection** - Automatically detects acronyms, codes, IDs, and quoted phrases
- **Dynamic Semantic/BM25 Fallback** - Switches between semantic and keyword search based on query characteristics
- **Result Collapsing** - Groups chunk-level results by parent document
- **Multi-dimensional Sorting** - Sort by relevance score, creation date, or update date

### Filtering & Faceting

- Topics, themes, languages, locations
- Categories and subcategories
- Project types and acronyms
- Related project aggregation

### Debug & Analysis

- Query explanation (`debug_explain`)
- Execution profiling (`debug_profile`)
- LLM-powered explanation (`debug_llm_explain`)
- Debug data persistence (`debug_save`)

## Configuration

Key environment variables (see `.env` for full list):

| Variable | Description |
|----------|-------------|
| `OPENSEARCH_API` | OpenSearch host |
| `OPENSEARCH_USR` / `OPENSEARCH_PWD` | OpenSearch credentials |
| `OPENSEARCH_MSMARCO_MODEL_ID` | Neural search model ID |
| `DEEPL_API_KEY` | DeepL translation API key |
| `HF_TOKEN` / `HF_MODEL_ID` | Hugging Face credentials for summarization |
| `VALIDATE_ACCESS_TOKEN_DEV_URL` / `VALIDATE_ACCESS_TOKEN_PRD_URL` | Auth validation endpoints |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design, component interactions, and data flow diagrams.

## Project Structure

```
.
├── app.py                          # Main FastAPI application
├── main.py                         # Minimal FastAPI stub
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container build instructions
├── docker-compose.yml              # Docker Compose configuration
├── deploy.sh                       # Deployment script
├── services/                       # Business logic modules
│   ├── utils.py                    # Shared utilities & OpenSearch client
│   ├── neural_search_relevant.py   # Core neural search implementation
│   ├── neural_search_relevant_hybrid.py  # Hybrid search pipeline
│   ├── neural_search_relevant_sparse.py  # Sparse neural search
│   ├── neural_search_relevant_new.py     # Alternative search implementation
│   ├── recommender_knn.py          # k-NN recommendation engine
│   ├── search_endpoint_helpers.py  # Endpoint utility functions
│   ├── language_detect.py          # Translation & language detection
│   ├── summariser_hf.py            # Hugging Face summarization
│   └── clickhouse_logger.py        # Analytics logging (optional)
└── tools/                          # Debugging & development tools
    ├── debug_llm_summary.py        # Debug explanation builder
    ├── llm_explain_client.py       # LLM explanation client
    └── analyse_debug.py            # Debug analysis utilities
```

## Dependencies

- **FastAPI** - Web framework
- **OpenSearch-py** - OpenSearch client
- **DeepL** - Translation services
- **Hugging Face** - LLM summarization
- **NLTK** - Text processing
- **Pydantic** - Data validation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
