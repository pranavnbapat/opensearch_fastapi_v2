# Neural Search Relevant

`/neural_search_relevant` is the current default search endpoint.

## Goal

Provide semantic-first retrieval with enough lexical precision to keep result counts controlled and first-page ranking strong.

## Current strategy

The endpoint uses query-intent routing:

- normal topical queries:
  - semantic-first retrieval
- code-like / acronym-like / exact-term queries:
  - lexical-first fallback

Intent detection is done before the OpenSearch query is built.

## Semantic-first mode

Semantic-first mode requires the semantic branch and uses lexical evidence only to improve ordering.

### Semantic branch

Uses `dis_max` over:

- `title_embedding`
- `subtitle_embedding`
- `description_embedding`
- `keywords_embedding`
- `content_embedding`

### Lexical ranking boosts

Uses lexical evidence as `should` boosts:

- broad `multi_match`
- high-precision `AND`-style `multi_match`
- phrase `multi_match`
- exact `project_acronym` boost

This keeps semantic retrieval as the main gate while letting exact textual evidence improve ranking.

## Lexical-first fallback

Used for:

- acronym-like queries
- code-like queries
- exact-lookups where semantic expansion is less desirable

This fallback is now truly lexical-first. It no longer uses the hidden neural assist that previously made the fallback broader than intended.

## Filters

Request-level filters are supported through the normal request body:

- `topics`
- `themes`
- `languages`
- `category`
- `project_type`
- `project_acronym`
- `locations`

These are applied as OpenSearch filter clauses.

## Grouping

Results are retrieved at chunk level and collapsed to parent KO level:

- grouped by `parent_id`
- top chunks are retained as `inner_hits`

This is why `total_records` refers to parent objects, not raw chunks.

## Sorting

Supported via `sort_by`:

- score
- KO created/updated date
- project created/updated date

## Debug behavior

### `debug_explain`

When enabled:

- returns structured explain data for top hits
- automatically attempts an LLM explanation if LLM config is available
- falls back silently to structured debug if the LLM is unavailable

### `debug_profile`

When enabled:

- requests OpenSearch profile information
- mainly useful for query-performance debugging

## Current recommendation

This is the recommended default search endpoint for the current corpus.

Reasons:

- better relevance than the hybrid endpoint in evaluation
- controlled result counts after semantic-first tightening
- supports query-intent-aware fallback without exposing hybrid complexity

## Example request

```json
{
  "search_term": "soil health",
  "page": 1,
  "dev": false,
  "k": 0,
  "model": "msmarco",
  "include_fulltext": false,
  "include_summary": false,
  "debug_profile": false,
  "debug_explain": false,
  "sort_by": "score_desc",
  "access_token": "string"
}
```

## Notes

- `k` is present in the request model, but the main endpoint still behaves as paginated parent search unless explicit top-k handling is reintroduced
- slight result-count variation can still happen because of ANN retrieval, collapse behavior, and cardinality counting
