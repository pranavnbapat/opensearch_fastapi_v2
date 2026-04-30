# Architecture Documentation

## System Overview

The EU FarmBook OpenSearch API is a Python-based FastAPI application that provides search and recommendation capabilities over an OpenSearch cluster. The current primary retrieval strategy is semantic-first search with lexical boosting, alongside experimental hybrid, sparse, and advanced boolean-aware search paths.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Client Applications                               │
│         (EU FarmBook Portal, Admin Dashboard, Mobile Apps)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        EU FarmBook OpenSearch API                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Search    │  │   Hybrid    │  │   Sparse    │  │  Recommendation │    │
│  │  Endpoints  │  │   Search    │  │   Search    │  │    Endpoints    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     Service Layer (services/)                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │ Neural   │ │ Hybrid   │ │ Sparse   │ │ k-NN     │ │ Language │  │    │
│  │  │ Search   │ │ Search   │ │ Search   │ │ Recommend│ │ Detect   │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                            │    │
│  │  │Summarizer│ │  Auth    │ │ Utilities│                            │    │
│  │  │   (HF)   │ │ Middleware│ │          │                           │    │
│  │  └──────────┘ └──────────┘ └──────────┘                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │      OpenSearch         │    │         External Services               │ │
│  │  ┌─────────────────┐    │    │  ┌─────────────┐    ┌────────────────┐  │ │
│  │  │ Neural Index    │    │    │  │   DeepL     │    │  FarmBook Auth │  │ │
│  │  │ (Embeddings)    │◄───┼────┼──┤ Translation │    │   (Django)     │  │ │
│  │  ├─────────────────┤    │    │  └─────────────┘    └────────────────┘  │ │
│  │  │ Sparse Index    │    │    │  ┌─────────────┐    ┌────────────────┐  │ │
│  │  │ (Learned)       │◄───┼────┼──┤  Hugging    │    │  ClickHouse    │  │ │
│  │  ├─────────────────┤    │    │  │   Face      │    │ (Analytics)    │  │ │
│  │  │ BM25 Text       │    │    │  │Summarization│    │   (Optional)   │  │ │
│  │  │ (Inverted)      │    │    │  └─────────────┘    └────────────────┘  │ │
│  │  └─────────────────┘    │    │  ┌─────────────┐                        │ │
│  └─────────────────────────┘    │  │   LLM API   │                        │ │
│                                 │  │(Debug/Expln)│                        │ │
│                                 │  └─────────────┘                        │ │
│                                 └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`app.py`)

The main FastAPI application that defines all HTTP endpoints and orchestrates request handling.

**Responsibilities:**
- Route definition and request validation (Pydantic models)
- Authentication middleware (Basic Auth + time-based expiration)
- CORS configuration
- Request preprocessing (translation, query analysis)
- Response formatting and post-processing (summarization, pagination)

**Key Endpoints:**
- `/neural_search_relevant` - Default semantic-first search with lexical boosting
- `/neural_search_relevant_advanced` - Experimental boolean-aware search with explicit operators and field scoping
- `/neural_search_relevant_hybrid` - Experimental OpenSearch hybrid search with optional pipeline normalization
- `/neural_search_relevant_sparse` - Learned sparse retrieval (`neural_sparse`)
- `/neural_search_relevant_new` - Experimental fragment-aware search path
- `/recommend_similar_knn` - k-NN content-based recommendations

### 2. Service Layer

#### 2.1 Neural Search (`neural_search_relevant.py`)

Implements the current default semantic-first search algorithm using OpenSearch's neural query DSL.

**Key Features:**
- **Multi-field Embedding Search** - Queries across 5 embedding fields (title, subtitle, description, keywords, content) with field-specific boosts
- **Dynamic k-values** - Adjusts candidate pool sizes based on query length
- **Disjunction Max (dis_max)** - Allows the strongest field match to dominate scoring
- **Lexical Ranking Boosts** - Uses lexical signals to improve ordering without making them broad eligibility gates in semantic mode
- **Smart Semantic Toggle** - Disables semantic search for IDs, codes, acronyms, and quoted phrases

**Query Flow:**
```
User Query → Query Analysis → (Optional: Translation) →
Choose Semantic-first or Lexical-first Path →
Build Retrieval Clauses → Apply Filters →
Collapse by parent_id → Return Results
```

#### 2.2 Hybrid Search (`neural_search_relevant_hybrid.py`)

Uses OpenSearch's native hybrid query with optional search-pipeline-based score normalization.

**Key Features:**
- **Native Hybrid DSL** - Uses OpenSearch's `hybrid` query type
- **Search Pipeline Integration** - Uses the configured hybrid pipeline when enabled
- **BM25 + Neural** - Combines multi_match text queries with neural embedding queries
- **Pagination Depth Control** - Configurable depth for accurate result collapsing

This endpoint is currently considered experimental relative to the tuned default semantic-first endpoint.

#### 2.3 Advanced Search (`neural_search_relevant_advanced.py`)

Implements an isolated experimental search path for boolean-aware and field-scoped retrieval.

**Key Features:**
- **Explicit Boolean Parsing** - Supports uppercase `AND`, `OR`, `NOT`, and parentheses
- **Field-scoped Clauses** - Supports syntax like `title:"soil health"` and `keywords:"crop rotation"`
- **Project Scoping** - Supports `-project`, `--project`, `project:`, and `acronym:`
- **Mode Controls** - Supports `mode:strict|broad|semantic|lexical`
- **Fallback Switch** - With `advanced=false`, behaves like `/neural_search_relevant`

#### 2.4 Sparse Neural Search (`neural_search_relevant_sparse.py`)

Implements learned sparse retrieval using OpenSearch's `neural_sparse` query.

**Key Features:**
- **Doc-only Sparse Mode** - Uses pre-computed sparse vectors at index time
- **Analyzer-based Querying** - Query-time token expansion using BERT-compatible analyzers
- **Rank Features Scoring** - Efficient sparse dot-product scoring

#### 2.5 k-NN Recommender (`recommender_knn.py`)

Content-based recommendation engine using vector similarity.

**Key Features:**
- **Seed Document Lookup** - Fetches vector embeddings for the clicked document
- **Multiple Embedding Fields** - Falls back through content → description → title → keywords
- **Exact vs ANN Modes** - Supports both brute-force exact k-NN and approximate nearest neighbors
- **Weighted Multi-space** - Can combine multiple embedding fields with custom weights
- **Script Score Reranking** - Uses Painless scripts for weighted cosine similarity

**Modes:**
- `exact` - Full index scan with precise cosine similarity
- `ann` - Approximate nearest neighbors for large-scale retrieval

#### 2.6 Language Detection & Translation (`language_detect.py`)

Multi-provider translation service with fallback capabilities.

**Features:**
- **Dual Detection** - Uses both `langdetect` and `langid` for robust language identification
- **DeepL Primary** - First-choice translation via DeepL API
- **Google Fallback** - Google Cloud Translation as secondary provider
- **Exponential Backoff** - Retry logic for resilient API calls

#### 2.7 Summarization (`summariser_hf.py`)

LLM-powered result summarization using Hugging Face inference API.

**Features:**
- **Token Overlap Scoring** - Reranks results by query relevance before summarization
- **Concurrency Control** - Semaphore-based rate limiting
- **Structured Prompts** - Enforces factual, non-meta commentary in summaries
- **Configurable Timeouts** - Adjustable connection and read timeouts

#### 2.8 Utilities (`utils.py`)

Shared infrastructure and helper functions.

**Components:**
- **OpenSearch Client** - Singleton client with connection pooling
- **MultiUserTimedAuthMiddleware** - Basic auth with time-based access control
- **Result Grouping** - Collapses chunk-level hits by parent document
- **Sorting** - Multi-dimensional sort clause generation
- **Token Validation** - Async validation against FarmBook Django backend

### 3. External Integrations

#### 3.1 OpenSearch Cluster

The primary data store with multiple index types:

| Index Type | Purpose | Query Method |
|------------|---------|--------------|
| Dense Embeddings | Semantic similarity | `neural` queries |
| Sparse Features | Learned sparse retrieval | `neural_sparse` queries |
| BM25 Text | Keyword matching | `multi_match` queries |

**Index Naming:**
- Production: `neural_search_index_msmarco_distilbert`
- Development: `neural_search_index_msmarco_distilbert_dev`

#### 3.2 FarmBook Authentication

Token-based access control integration:
- Validates JWT access tokens against Django backend
- Enables translation and summarization features for authenticated users
- Extracts user_id from token claims for analytics

#### 3.3 Hugging Face

LLM inference for result summarization:
- Uses Hugging Face Inference API or dedicated endpoints
- Supports configurable model selection
- Implements concurrency controls and timeouts

#### 3.4 LLM Debug Explanation

Optional LLM-backed explanation for `debug_explain=true`:

- attempts a human-readable explanation when LLM config is available
- falls back silently to structured debug output if unavailable

#### 3.5 DeepL

Machine translation for multilingual search:
- Automatic query translation to English
- Supports 30+ languages
- HTML-aware translation preserving formatting

## Data Flow

### Search Request Flow

```
┌─────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Client │────►│   FastAPI   │────►│  Auth Middleware│────►│Query Analysis│
└─────────┘     └─────────────┘     └─────────────────┘     └──────────────┘
                                                                     │
                                                                     ▼
┌─────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Client │◄────│  Response   │◄────│ Result Grouping │◄────│OpenSearch Qry│
└─────────┘     │ Formatting  │     │   & Sorting     │     │   Execution  │
                └─────────────┘     └─────────────────┘     └──────────────┘
```

### Recommendation Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Client │────►│  /recommend  │────►│ Fetch Seed Doc   │────►│Get Embeddings│
│         │     │  _similar_knn│     │  (chunk_index=-1)│     │  (best field)│
└─────────┘     └──────────────┘     └──────────────────┘     └──────────────┘
                                                                     │
                                                                     ▼
┌─────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Client │◄────│ Format Resp │◄────│ k-NN Retrieval  │◄────│ Build Script │
└─────────┘     │  + Metadata │     │ (Exact or ANN)  │     │    Score     │
                └─────────────┘     └─────────────────┘     └──────────────┘
```

## Query Processing Pipeline

### 1. Query Analysis

The system analyzes incoming queries to determine optimal search strategy:

```python
# Detection patterns:
- looks_like_code_or_id: Contains digits, special chars (CVE, DOI, ISBN)
- looks_like_acronym: Short (<12 chars), multiple capitals (EIPAGRI)
- looks_like_quoted: Contains " or '
- very_short: ≤ 2 tokens

# Decision:
use_semantic = not (code_or_id or quoted or acronym) or (very_short and not code_or_id)
```

### 2. Translation (Authenticated Users)

```python
if access_token_valid and detected_lang != "en":
    query = deepl_translate(query, target="EN-GB")
```

### 3. Query Building

**Semantic Branch:**
- Build dis_max query across 5 embedding fields
- Field-specific boosts: title (1.4), description (1.1), content (1.0), subtitle (0.7), keywords (0.4)
- Dynamic k-values based on query length
- Add lexical boosts as secondary ranking clauses

**Lexical Fallback Branch:**
- Used for code-like, acronym-like, and exact-term-style queries
- Uses stricter lexical matching with project acronym boosts

**Advanced Branch:**
- Optional experimental parser for boolean-aware and field-scoped queries
- Can force semantic-only or lexical-only positive clause handling via `mode:`

**BM25 Branch:**
- Multi-match across text fields with English analyzer
- Reduced neural component (boost 0.3)

### 4. Result Processing

- **Collapsing** - Groups by `parent_id`, keeping best chunk per document
- **Sorting** - Applies user-specified sort (score, dates)
- **Pagination** - Calculates page boundaries
- **Enrichment** - Optionally fetches full text chunks

## Security Model

### Authentication Layers

1. **Basic Auth** (Middleware)
   - Username/password validation
   - Time-based access expiration
   - Configured per-user in `ALLOWED_USERS`

2. **Bearer Token** (Feature gates)
   - JWT access token validation
   - Enables premium features (translation, summarization)
   - User identification for analytics

### CORS Policy

Restricted to known origins:
- Local development (`127.0.0.1:8000`)
- Production API domains
- FarmBook backend instances (dev/prd)

## Error Handling

### OpenSearch Resilience

```python
try:
    exists = client.indices.exists(index=index_name, request_timeout=2)
except (ConnectionTimeout, ConnectionError):
    return {"_meta": {"error": "OpenSearch unreachable", "unreachable": True}, ...}
except TransportError:
    return {"_meta": {"error": "OpenSearch error", ...}, ...}
```

### Graceful Degradation

- Missing index → Empty results with metadata
- Translation failure → Original query used
- Summarization failure → Results without summary
- Token validation failure → Translation disabled

## Performance Considerations

### Caching

- BuildKit pip cache for Docker builds
- NLTK data downloaded at build time
- No runtime query caching (stateless design)

### Concurrency

- Async/await throughout the stack
- `aiohttp` for external API calls
- Connection pooling for OpenSearch
- Semaphore-controlled LLM calls

### Pagination Strategy

- **Deep Pagination** - Uses `from`/`size` with bounded limits
- **Result Collapsing** - Fetches more results than page size to ensure variety after grouping
- **Cursor Alternative** - Search After recommended for very deep paging

## Development Tools

### Debug Endpoints

All search endpoints support debug flags:

- `debug_explain` - Returns Lucene explanation for scoring
- `debug_profile` - Returns query execution profile
- `debug_llm_explain` - Uses LLM to explain results in natural language
- `debug_save` - Persists debug data to disk

### LLM Explanation Pipeline

```
Explain Output → Build Summary → Call LLM API → Natural Language Explanation
```

## Deployment Architecture

### Container Structure

```dockerfile
FROM python:3.11-slim
# Multi-stage build with BuildKit cache
# NLTK data pre-downloaded
# Exposed on port 10000
```

### Environment Strategy

- **Development** (`dev=true`) - Uses `_dev` index suffix
- **Production** (`dev=false`) - Uses production indexes

### Scaling Considerations

- Stateless application design
- Horizontal scaling via container orchestration
- OpenSearch cluster handles query load
- External services (DeepL, HF) may require rate limiting

## Future Enhancements

Potential architectural improvements:

1. **Query Caching** - Redis for frequent query results
2. **Result Caching** - CDN for popular document content
3. **Analytics Pipeline** - ClickHouse integration for search analytics
4. **A/B Testing Framework** - Multi-armed bandit for ranking experiments
5. **Federated Search** - Merge results from multiple indexes
