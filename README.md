# EU FarmBook OpenSearch API

A FastAPI-based search and recommendation service powered by OpenSearch neural search capabilities. The current default search path is semantic-first retrieval with lexical boosting, alongside experimental hybrid, sparse, advanced boolean-aware search, k-NN recommendations, and optional result summarization for the EU FarmBook knowledge platform.

## Overview

This service enables intelligent document search across agricultural knowledge objects (KOs) by combining:

- **Semantic-first Search** - Vector similarity using dense embeddings with lexical ranking boosts
- **Lexical Fallback** - Stricter BM25-style retrieval for acronym/code/exact-term queries
- **Hybrid Search** - OpenSearch-native BM25 + neural fusion with optional pipeline normalization
- **Advanced Search** - Boolean-aware experimental search with explicit operators and field scoping
- **Sparse Search** - Learned sparse retrieval over `content_sparse`
- **k-NN Recommendations** - Content-based similar document suggestions
- **AI Summarization** - LLM-powered result summaries via Hugging Face
- **Multi-language Support** - Translation-aware query handling for authenticated users

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
| `/neural_search_relevant` | POST | Default semantic-first search with lexical boosting |
| `/neural_search_relevant_advanced` | POST | Experimental boolean-aware search; can fall back to default mode |
| `/neural_search_relevant_hybrid` | POST | Experimental OpenSearch hybrid search with optional pipeline normalization |
| `/neural_search_relevant_sparse` | POST | Sparse semantic retrieval over `content_sparse` |
| `/neural_search_relevant_new` | POST | Experimental fragment-aware search path |

### Recommendation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend_similar_knn` | POST | k-NN semantic recommendations for a document |

### Request/Response Format

Most search endpoints accept JSON payloads based on the same core shape, with endpoint-specific additions:

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
  "debug_profile": false,
  "debug_explain": false,
  "sort_by": "score_desc",
  "access_token": "..."
}
```

Notes:

- `/neural_search_relevant_advanced` also accepts `advanced: true|false`.
- `/recommend_similar_knn` uses a different request body centered on `parent_id`, `mode`, `space`, and `mix`.
- `k` is present on some request models, but the main search endpoints still behave as paginated parent search unless explicit top-k handling is added back.

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
- **Query-intent Routing** - Switches between semantic-first and lexical-first behavior based on query characteristics
- **Result Collapsing** - Groups chunk-level results by parent document
- **Multi-dimensional Sorting** - Sort by relevance score, creation date, or update date
- **Advanced Boolean Parsing** - Available on the advanced endpoint with `AND`, `OR`, `NOT`, parentheses, field scoping, and project scoping

### Filtering & Faceting

- Topics, themes, languages, locations
- Categories and subcategories
- Project types and acronyms
- Related project aggregation

### Debug & Analysis

- Query explanation (`debug_explain`)
- Execution profiling (`debug_profile`)
- Best-effort LLM explanation automatically attempted when `debug_explain=true` and LLM config is available

## Configuration

Key environment variables (see `.env` for full list):

| Variable | Description |
|----------|-------------|
| `OPENSEARCH_API` | OpenSearch host |
| `OPENSEARCH_USR` / `OPENSEARCH_PWD` | OpenSearch credentials |
| `OPENSEARCH_MSMARCO_MODEL_ID` | Neural search model ID |
| `DEEPL_API_KEY` | DeepL translation API key |
| `HF_TOKEN` / `HF_MODEL_ID` | Hugging Face credentials for summarization |
| `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | Optional LLM debug explanation endpoint configuration |
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
│   ├── neural_search_relevant.py   # Default semantic-first search
│   ├── neural_search_relevant_advanced.py # Experimental advanced boolean-aware search
│   ├── neural_search_relevant_hybrid.py  # Experimental OpenSearch hybrid search
│   ├── neural_search_relevant_sparse.py  # Sparse neural search
│   ├── neural_search_relevant_new.py     # Experimental fragment-aware search
│   ├── recommender_knn.py          # k-NN recommendation engine
│   ├── search_endpoint_helpers.py  # Endpoint utility functions
│   ├── language_detect.py          # Translation & language detection
│   ├── summariser_hf.py            # Hugging Face summarization
│   └── clickhouse_logger.py        # Analytics logging (currently disabled at app level)
└── tools/                          # Debugging & development tools
    ├── debug_llm_summary.py        # Debug explanation builder
    ├── llm_explain_client.py       # LLM explanation client
    └── analyse_debug.py            # Debug analysis utilities
```

## Current Recommendation

- Use `/neural_search_relevant` as the default search endpoint.
- Treat `/neural_search_relevant_hybrid` as experimental for this corpus.
- Use `/neural_search_relevant_advanced` only when explicit boolean-aware control is needed.

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
