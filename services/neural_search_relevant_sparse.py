# services/neural_search_relevant_sparse.py

import logging
from typing import Dict, Any, Optional

from fastapi import HTTPException
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError

from services.utils import (
    PAGE_SIZE,
    client,
    group_hits_by_parent,
    build_sort,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def neural_search_relevant_sparse(
    index_name: str,
    query: str,
    filters: Dict[str, Any],
    page: int,
    analyzer: str = "bert-uncased",
    size_override: Optional[int] = None,
):
    """
    Neural sparse search (doc-only mode) over `content_sparse` (rank_features).

    - Uses `neural_sparse` query clause with `analyzer` (no model_id needed at query time).
    - Excludes meta docs (chunk_index == -1) when a query is provided.
    - Collapses by parent_id to return one result per KO
    """

    page_size = int(size_override) if isinstance(size_override, int) and size_override > 0 else PAGE_SIZE
    from_offset = (max(page, 1) - 1) * page_size

    filter_conditions = []
    if filters.get("topics"):
        filter_conditions.append({"terms": {"topics": filters["topics"]}})
    if filters.get("themes"):
        filter_conditions.append({"terms": {"themes": filters["themes"]}})
    if filters.get("languages"):
        filter_conditions.append({"terms": {"languages": filters["languages"]}})
    if filters.get("locations"):
        filter_conditions.append({"terms": {"locations": filters["locations"]}})
    if filters.get("category"):
        filter_conditions.append({"terms": {"category": filters["category"]}})
    if filters.get("project_type"):
        filter_conditions.append({"terms": {"project_type": filters["project_type"]}})
    if filters.get("project_acronym"):
        filter_conditions.append({"terms": {"project_acronym": filters["project_acronym"]}})

    # Exclude meta docs only when we actually have a query
    must_not = [] if not query else [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_clause = build_sort(str(raw_sort) if raw_sort is not None else "score_desc", has_query=bool(query))

    # -------------------- Query --------------------
    # IMPORTANT: Keep _source light but include the fields build_response_json expects later.
    # We collapse by parent_id so one "best" chunk per KO represents the parent.
    search_query = {
        "_source": {
            "includes": [
                "parent_id", "project_name", "project_acronym",

                "title", "subtitle", "description", "keywords",
                "title_original", "subtitle_original", "description_original", "keywords_original",

                "topics", "themes", "locations", "languages", "category", "subcategories",
                "date_of_completion", "creators", "intended_purposes", "project_id", "project_type",
                "project_url", "@id", "_orig_id", "ko_created_at", "ko_updated_at", "proj_created_at",
                "proj_updated_at",
            ],
        },
        "track_total_hits": True,
        "size": page_size,
        "from": from_offset,
        "query": (
            {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": filter_conditions,
                    "must_not": must_not,
                }
            }
            if not query
            else
            {
                "bool": {
                    "must": [
                        # Only search chunk docs (avoid meta)
                        {"range": {"chunk_index": {"gte": 0}}},
                        {
                            "neural_sparse": {
                                "content_sparse": {
                                    "query_text": query,
                                    "analyzer": analyzer,
                                }
                            }
                        },
                    ],
                    "filter": filter_conditions,
                    "must_not": must_not,
                }
            }
        ),
        "collapse": {
            "field": "parent_id",
            "inner_hits": {
                "name": "best_chunks",
                "size": 3,
                "_source": ["chunk_index", "content_chunk"],
            },
        },
        "aggs": {
            "unique_parents_total": {
                "cardinality": {"field": "parent_id", "precision_threshold": 40000}
            },
            "top_projects": {
                "terms": {"field": "project_id", "size": 3, "order": {"unique_parents": "desc"}},
                "aggs": {
                    "unique_parents": {
                        "cardinality": {"field": "parent_id", "precision_threshold": 40000}
                    }
                },
            },
        },
        "sort": sort_clause,
    }

    # ---- Fail fast if index doesn't exist ----
    try:
        exists = client.indices.exists(index=index_name, request_timeout=2)
        if not exists:
            return {
                "took": 0,
                "timed_out": False,
                "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                "aggregations": {"unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}},
                "grouped": {"total_parents": 0, "parents": []},
                "_meta": {"error": f"Index not found: {index_name}", "index_missing": True},
            }
    except (ConnectionTimeout, ConnectionError) as e:
        return {
            "took": 0,
            "timed_out": True,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "aggregations": {"unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}},
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"OpenSearch unreachable: {type(e).__name__}", "unreachable": True},
        }
    except TransportError as e:
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch error while checking index existence",
            "opensearch": detail,
        })

    # ---- Execute search ----
    try:
        response = client.search(index=index_name, body=search_query)
    except TransportError as e:
        logger.exception("Neural sparse search failed: %s", e)
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch neural sparse search failed",
            "opensearch": detail,
        })

    aggs = response.get("aggregations", {})
    total_parents_from_agg = aggs.get("unique_parents_total", {}).get("value") if aggs else None

    hits = response.get("hits", {}).get("hits", []) or []
    grouped = group_hits_by_parent(hits, parents_size=page_size)
    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped
    return response
