# Recommender KNN

`/recommend_similar_knn` is the content-based recommendation endpoint.

## Goal

Given a clicked or selected KO parent, return semantically similar KOs.

This endpoint is not a keyword search endpoint. It is a nearest-neighbour recommender.

## Input

Main request fields:

- `parent_id`
- `k`
- `k_candidates`
- `dev`
- `model`
- `include_fulltext`
- `mode`
- `space`
- `mix`

## Seed document

The recommender uses the KO meta document:

- `parent_id = requested parent`
- `chunk_index = -1`

This meta doc is used as the semantic seed.

## Embedding spaces

Available embedding spaces:

- `content`
- `description`
- `title`
- `keywords`
- `mix`

Priority map in the implementation:

- `content_embedding`
- `description_embedding`
- `title_embedding`
- `keywords_embedding`

## Modes

### `mode: "exact"`

Uses script-scored cosine similarity over the selected embedding space.

This is the default mode.

### `mode: "ann"`

Uses approximate nearest-neighbour candidate retrieval when `k_candidates` is set.

This is useful when you want broader candidate recall before parent grouping.

## Space selection

### Single space

Examples:

- `space: "content"`
- `space: "title"`

The seed vector is taken from the corresponding embedding field.

### Mixed space

Set:

- `space: "mix"`
- `mix: {"content": 0.7, "title": 0.3}`

The endpoint:

- keeps only valid positive weights
- normalizes them
- builds a weighted script score across the selected embedding fields

## Exclusions and grouping

The seed parent itself is excluded from recommendations.

Results are then:

- grouped by `parent_id`
- enriched from meta docs for display

So the output is KO-level recommendations, not raw chunk hits.

## Full text

If `include_fulltext=true`:

- chunks are fetched for returned parents
- full text is attached to the response

Otherwise:

- response stays lightweight

## Failure behavior

The service returns explicit metadata when:

- index is missing
- OpenSearch is unreachable
- seed meta doc is missing
- requested vectors are missing

## Example request

```json
{
  "parent_id": "669508a971b0ee89777c51a6",
  "k": 6,
  "k_candidates": 0,
  "dev": false,
  "model": "msmarco",
  "include_fulltext": false,
  "mode": "exact",
  "space": "content"
}
```

## When to use this endpoint

Use it for:

- related KO suggestions
- “more like this” panels
- follow-up discovery from a selected document

Do not use it as a replacement for query search. It is seeded by an existing KO, not by free-text intent.
