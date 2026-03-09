# OpenSearch ML and Pipeline Runbook

This runbook documents the exact setup sequence for EU-FarmBook neural/hybrid search, why each step exists, and how to operate/tune it safely.

## Scope

This covers:
- ML plugin readiness checks
- model group + model registration/deployment
- ingest pipeline creation (dense + sparse embeddings)
- query-time hybrid search pipeline creation (`v1`, `v2`)
- index creation requirements
- why the order matters
- how the API uses these pipelines (`HYBRID_SEARCH_PIPELINE` + fallback)

---

## Why This Order Matters

Order is important because each layer depends on previous layers:

1. **Plugin and node readiness**
2. **Model registry and deployments**
3. **Ingest pipeline that references deployed models**
4. **Query-time search pipeline for score blending**
5. **Index mappings/settings that match API query fields**
6. **Application configuration to use/tune pipelines**

If this order is broken, you can get: missing model errors, empty vectors, bad ranking, or no hybrid normalization.

---

## 1) Verify ML plugin is installed

```http
GET /_cat/plugins?v
```

Expected:
- `opensearch-ml`
- ideally also `opensearch-neural-search` and `opensearch-knn`

If missing:
- ML Commons is not installed; model groups/models cannot be used.

---

## 2) Verify ML plugin is loaded on nodes

```http
GET /_nodes/plugins
```

Expected:
- all nodes list `opensearch-ml`
- at least one node has `ml` role

If missing:
- cluster is not configured for ML workloads.

---

## 3) Check ML system indices

```http
GET /_cat/indices/.plugins-ml*?v
```

Expected (fresh cluster):
- `.plugins-ml-config` only

Expected (used cluster):
- additional indices (`.plugins-ml-model`, `.plugins-ml-task`, model-group related)

Interpretation:
- system/model indices are lazily created after first ML usage.

---

## 4) Check ML plugin status

```http
GET /_plugins/_ml/stats
```

Expected (fresh):
- `ml_config_index_status: green`
- model/task index status may be `non-existent`
- counts often `0`

Interpretation:
- plugin is healthy but unused.

---

## 5) Attempt to list model groups

```http
GET /_plugins/_ml/model_groups/_search
{
  "query": { "match_all": {} }
}
```

Outcomes:
- **Case A**: no groups yet (`hits.total=0`, `_shards.total=0`) -> backing index not created yet
- **Case B**: groups exist -> documents returned

---

## 6) Enable model access control

```http
PUT /_cluster/settings
{
  "transient": {
    "plugins.ml_commons.model_access_control_enabled": "true"
  }
}
```

Why:
- allows controlled model-group access modes and safer multi-tenant operation.

---

## Step 1: Register model group

```http
POST /_plugins/_ml/model_groups/_register
{
  "name": "eu-farmbook-semantic",
  "description": "Semantic embedding models used by EU-FarmBook for hybrid search (BM25 and neural) and content-based recommendations across Knowledge Objects.",
  "access_mode": "public"
}
```

Why:
- organizes related embedding models under one logical group.

---

## Step 2: Register dense model

```http
POST /_plugins/_ml/models/_register
{
  "name": "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
  "version": "1.0.3",
  "model_group_id": "<your_model_group_id>",
  "model_format": "TORCH_SCRIPT"
}
```

Then poll task:

```http
GET /_plugins/_ml/tasks/<task_id>
```

Expected final:
- `state: COMPLETED`
- includes `model_id` (example: `ovClkJsB_qkkFA9OzmRk`)

Why:
- this model generates dense vectors used by neural search.

---

## Step 3.1: Deploy dense model

```http
POST /_plugins/_ml/models/<model_id>/_deploy
```

Then poll:

```http
GET /_plugins/_ml/tasks/<deploy_task_id>
```

Expected:
- `state: COMPLETED`

Why:
- deployed models are needed for ingest/query-time inference.

---

## Step 3.2: Register+deploy sparse model

```http
POST /_plugins/_ml/models/_register?deploy=true
{
  "name": "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT"
}
```

Poll task to completion; capture sparse model id (example: `C6IJn5sB3NGHdteUcSl6`).

Why:
- sparse encoding improves lexical-semantic matching behavior with neural sparse retrieval.

---

## Step 4.1: Create ingest pipeline (index-time embeddings)

Create pipeline (example: `eufb_neural_ko_v1`) with:
- painless pre-processing (blank-safe/list-safe input fields)
- `sparse_encoding` processor for `content_sparse`
- `text_embedding` processor for dense vectors (`title/subtitle/description/keywords/content`)
- failure handler via `on_failure`

Why:
- converts source fields into vector fields consumed by search endpoints.

Important:
- model IDs in ingest pipeline must match deployed model IDs.

---

## Step 4.2: Create query-time hybrid search pipeline

### v1 (neural-heavy)

```http
PUT /_search/pipeline/eufb-hybrid-v1
{
  "description": "Hybrid normalisation for EUFB neural+BM25",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.30, 0.70]
          }
        }
      }
    }
  ]
}
```

### v2 (BM25-heavy)

```http
PUT /_search/pipeline/eufb-hybrid-v2
{
  "description": "Hybrid normalization BM25-heavy",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": { "weights": [0.70, 0.30] }
        }
      }
    }
  ]
}
```

Why:
- blends BM25 and neural branch scores into one rank list.

---

## Step 5: Create index with compatible mappings/settings

Use index settings:
- `index.knn: true`
- `default_pipeline: <your_ingest_pipeline>`

Define mappings for all fields your API queries and returns.

Critical compatibility rule:
- API currently queries fields like:
  - `title.en`, `subtitle.en`, `description.en`, `keywords.en`, `content_chunk.en`
  - `title_embedding`, `subtitle_embedding`, `description_embedding`, `keywords_embedding`, `content_embedding`
  - `parent_id`, `project_id`, `chunk_index`

If index mapping uses different names (for example `projectName` vs `project_name`, `content_pages` vs `content_chunk`), relevance degrades or query behavior becomes inconsistent.

---

## Application Configuration

### Hybrid pipeline selection

Set in `.env`:

```env
HYBRID_SEARCH_PIPELINE=eufb-hybrid-v2
```

Current code behavior:
- requested pipeline = `HYBRID_SEARCH_PIPELINE`
- fallback chain: requested -> `eufb-hybrid-v1` -> no pipeline
- response metadata includes used pipeline info

Why:
- safe A/B testing and graceful runtime fallback.

### Query-intent tuning

Global defaults:

```env
QUERY_CODE_HINT_REGEX=\\d|[_:/]|cve-|doi|isbn
QUERY_ACRONYM_MIN_LEN=2
QUERY_ACRONYM_MAX_LEN=12
QUERY_ACRONYM_MIN_CAPS=2
```

Hybrid-specific overrides (optional):

```env
HYBRID_QUERY_CODE_HINT_REGEX=...
HYBRID_QUERY_ACRONYM_MIN_LEN=...
HYBRID_QUERY_ACRONYM_MAX_LEN=...
HYBRID_QUERY_ACRONYM_MIN_CAPS=...
```

Why:
- tune hybrid fallback behavior independently from main relevant endpoint.

---

## Operational Checklist Before Evaluation

1. `GET /_search/pipeline/eufb-hybrid-v2` returns expected weights
2. `HYBRID_SEARCH_PIPELINE` is set and API restarted
3. target index exists and field mappings match API query fields
4. model deployment tasks are `COMPLETED`
5. run evaluation and compare:
   - `focus_correct@k`
   - `focus_incorrect@k`
   - `focus_nDCG@k`

---

## Notes on Interpreting Results

- Plugin/model setup success does **not** guarantee good ranking.
- Ranking quality depends heavily on:
  - branch score normalization strategy
  - branch weights
  - schema-field compatibility
  - query intent routing

Use evaluation outputs (`per_call.csv`, `query_verdicts.csv`, `page_gap_analysis.csv`) for iterative tuning.
