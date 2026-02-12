# services/neural_search_relevant_hybrid.py

import logging

from enum import Enum
from typing import List, Optional, Dict, Any, Union

from fastapi import HTTPException
from pydantic import BaseModel
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError

from services.utils import (PAGE_SIZE, client, group_hits_by_parent, build_sort_hybrid)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SortBy(str, Enum):
    score_desc = "score_desc"
    score_asc  = "score_asc"

    ko_created_at_desc  = "ko_created_at_desc"
    ko_created_at_asc   = "ko_created_at_asc"
    ko_updated_at_desc  = "ko_updated_at_desc"
    ko_updated_at_asc   = "ko_updated_at_asc"

    proj_created_at_desc = "proj_created_at_desc"
    proj_created_at_asc  = "proj_created_at_asc"
    proj_updated_at_desc = "proj_updated_at_desc"
    proj_updated_at_asc  = "proj_updated_at_asc"

NUMERIC_SORT_MAP = {
    1: SortBy.score_desc,          2: SortBy.score_asc,
    3: SortBy.ko_created_at_desc,  4: SortBy.ko_created_at_asc,
    5: SortBy.ko_updated_at_desc,  6: SortBy.ko_updated_at_asc,
    7: SortBy.proj_created_at_desc,8: SortBy.proj_created_at_asc,
    9: SortBy.proj_updated_at_desc,10: SortBy.proj_updated_at_asc,
}

PAGINATION_DEPTH = 200


def coerce_sort(raw: Union[None, int, str, SortBy]) -> str:
    """
    Accepts Enum | int | str | None and returns a canonical sort key string.
    Falls back to 'score_desc' on unknown values.
    """
    if raw is None:
        return SortBy.score_desc.value
    if isinstance(raw, SortBy):
        return raw.value
    # numerals (e.g., "3" or 3)
    try:
        i = int(raw)
        return NUMERIC_SORT_MAP.get(i, SortBy.score_desc).value
    except (TypeError, ValueError):
        pass
    s = str(raw).strip().lower()
    for member in SortBy:
        if s == member.value:
            return member.value
    return SortBy.score_desc.value

class RelevantSearchRequestHybrid(BaseModel):
    search_term: str
    topics: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    category: Optional[List[str]] = None
    project_type: Optional[List[str]] = None
    project_acronym: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    page: Optional[int] = 1
    dev: Optional[bool] = True
    k: Optional[int] = None
    model: Optional[str] = "msmarco"
    include_fulltext: Optional[bool] = False
    include_summary: Optional[bool] = False
    debug_profile: Optional[bool] = False
    debug_explain: Optional[bool] = False
    debug_llm_explain: Optional[bool] = False
    debug_analyze: Optional[bool] = False       # Currently not being used
    debug_field: Optional[str] = None           # Currently, can't be used as it depends on debug_analyze
    debug_save: Optional[bool] = False
    debug_save_dir: Optional[str] = None
    sort_by: Optional[Union[SortBy, int, str]] = SortBy.score_desc
    access_token: Optional[str] = None


def neural_search_relevant_hybrid(
        index_name: str,
        query: str,
        filters: Dict[str, Any],
        page: int,
        model_id: str,
        search_pipeline: str = "eufb-hybrid-v1",
        size_override: Optional[int] = None,
        debug_profile: bool = False,
        debug_explain: bool = False,
):
    """
    Hybrid query (BM25 + neural) with query-time score normalisation via a Search Pipeline.

    Requires:
      - a Search Pipeline named `search_pipeline` already created
      - the index to have `.en` subfields (title.en, subtitle.en, etc.) if you want stemming
    """

    page_size = int(size_override) if isinstance(size_override, int) and size_override > 0 else PAGE_SIZE
    from_offset = (page - 1) * page_size

    # candidate pools
    q_len = len((query or "").split())
    k_content = 300 if q_len >= 3 else 200

    # Filters
    filter_conditions = []
    if filters.get("topics"):
        filter_conditions.append({"terms": {"topics": filters["topics"]}})
    if filters.get("themes"):
        filter_conditions.append({"terms": {"themes": (filters["themes"])}})
    if filters.get("languages"):
        filter_conditions.append({"terms": {"languages": (filters["languages"])}})
    if filters.get("locations"):
        filter_conditions.append({"terms": {"locations": filters["locations"]}})
    if filters.get("category"):
        filter_conditions.append({"terms": {"category": filters["category"]}})
    if filters.get("project_type"):
        filter_conditions.append({"terms": {"project_type": filters["project_type"]}})
    if filters.get("project_acronym"):
        filter_conditions.append({"terms": {"project_acronym": filters["project_acronym"]}})

    bm25_text = query

    # Exclude meta docs only when we actually have a query
    must_not = [] if not bm25_text else [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_key = coerce_sort(raw_sort)

    pagination_depth = max(PAGINATION_DEPTH, from_offset + page_size + 200)

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
                    "must_not": must_not
                }
            }
            if not query
            else {
                "hybrid": {
                    "pagination_depth": pagination_depth,
                    "queries": [
                        {
                            "bool": {
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": bm25_text,
                                            "fields": [
                                                "title.en^10",
                                                "subtitle.en^9",
                                                "description.en^6",
                                                "keywords.en^4",
                                                "content_chunk.en^2"
                                            ],
                                            "operator": "or",
                                            "minimum_should_match": "70%",
                                            "type": "best_fields"
                                        }
                                    }
                                ],
                                "filter": filter_conditions,
                                "must_not": must_not
                            }
                        },
                        {
                            "bool": {
                                "must": [
                                    {
                                        "neural": {
                                            "content_embedding": {
                                                "query_text": query,
                                                "model_id": model_id,
                                                "k": k_content
                                            }
                                        }
                                    }
                                ],
                                "filter": filter_conditions,
                                "must_not": must_not
                            }
                        }
                    ]
                }
            }
        ),
        "aggs": {
            "unique_parents_total": {
                "cardinality": {
                    "field": "parent_id",
                    "precision_threshold": 40000
                }
            },
            "top_projects": {
                "terms": {
                    "field": "project_id",
                    "size": 3,
                    "order": {"unique_parents": "desc"}
                },
                "aggs": {
                    "unique_parents": {
                        "cardinality": {
                            "field": "parent_id",
                            "precision_threshold": 40000
                        }
                    }
                }
            }
        },
    }

    if debug_profile:
        logger.warning("debug_profile requested, but disabled for hybrid queries to avoid OpenSearch NPE.")

    if debug_explain:
        search_query["explain"] = True

    # Keep your existing sorting behaviour
    sort_clause = build_sort_hybrid(sort_key, has_query=bool(bm25_text))
    search_query["sort"] = sort_clause

    params = {}
    if query and search_pipeline:
        params["search_pipeline"] = search_pipeline
        params["error_trace"] = "true"

    debug_bundle = {
        "request_flags": {
            "debug_profile": bool(debug_profile),
            "debug_explain": bool(debug_explain),
        },
        # store the exact query we sent to OpenSearch (super useful later)
        "opensearch_request": {
            "index": index_name,
            "params": params,
            "body": search_query,
        },
    }

    # Fail fast if index missing (copy your existing block, unchanged)
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

    # The crucial bit: pass the search_pipeline param
    try:
        response = client.search(
            index=index_name,
            body=search_query,
            params=params or None,
        )

        hits = response.get("hits", {}).get("hits", [])
        logger.info("[HYBRID DEBUG] returned_hits=%d page_size=%d from=%d",
                    len(hits), page_size, from_offset)
        logger.info("[HYBRID DEBUG] sample parent_ids=%s",
                    [h.get("_source", {}).get("parent_id") for h in hits[:10]])


    except TransportError as e:
        logger.exception("Hybrid search failed: %s", e)
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch hybrid search failed",
            "opensearch": detail,
        })

    aggs = response.get("aggregations", {})
    total_parents_from_agg = aggs.get("unique_parents_total", {}).get("value") if aggs else None

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=page_size)
    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped

    if debug_profile or debug_explain:
        response["_debug_bundle"] = debug_bundle

    return response

