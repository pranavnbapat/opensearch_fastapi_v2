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
        use_semantic: bool = True,
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

    # Candidate pools
    q_len = len((query or "").split())
    k_content = 300 if q_len >= 3 else 200
    k_title = 100 if q_len >= 3 else 60
    k_subtitle = 120 if q_len >= 3 else 80
    k_description = 180 if q_len >= 3 else 120
    k_keywords = 140 if q_len >= 3 else 90
    bm25_msm = "70%" if q_len >= 5 else None

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

    semantic_neural_dismax = {
        "dis_max": {
            "tie_breaker": 0.1,
            "queries": [
                {
                    "neural": {
                        "title_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_title,
                            "boost": 1.4
                        }
                    }
                },
                {
                    "neural": {
                        "subtitle_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_subtitle,
                            "boost": 0.7
                        }
                    }
                },
                {
                    "neural": {
                        "description_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_description,
                            "boost": 1.1
                        }
                    }
                },
                {
                    "neural": {
                        "keywords_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_keywords,
                            "boost": 0.4
                        }
                    }
                },
                {
                    "neural": {
                        "content_embedding": {
                            "query_text": query,
                            "model_id": model_id,
                            "k": k_content,
                            "boost": 1.0
                        }
                    }
                },
            ]
        }
    }

    bm25_clause = {
        "multi_match": {
            "query": bm25_text,
            "fields": [
                "title.en^10",
                "subtitle.en^9",
                "description.en^6",
                "keywords.en^4"
            ],
            "operator": "or",
            **({"minimum_should_match": bm25_msm} if bm25_msm else {}),
            "type": "best_fields"
        }
    }

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
            else (
                {
                    "hybrid": {
                        "pagination_depth": pagination_depth,
                        "queries": [
                            {
                                "bool": {
                                    "must": [bm25_clause],
                                    "filter": filter_conditions,
                                    "must_not": must_not
                                }
                            },
                            {
                                "bool": {
                                    "must": [semantic_neural_dismax],
                                    "filter": filter_conditions,
                                    "must_not": must_not
                                }
                            }
                        ]
                    }
                }
                if use_semantic
                else {
                    "bool": {
                        "must": [bm25_clause],
                        "filter": filter_conditions,
                        "must_not": must_not
                    }
                }
            )
        ),
        "collapse": {
            "field": "parent_id",
            "inner_hits": {
                "name": "best_chunks",
                "size": 3,
                "_source": ["chunk_index", "content_chunk"]
            }
        },
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

    def _with_pipeline_params(pipeline_name: Optional[str]) -> Dict[str, str]:
        p: Dict[str, str] = {}
        if query and use_semantic and pipeline_name:
            p["search_pipeline"] = pipeline_name
            p["error_trace"] = "true"
        return p

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

    # Execute with graceful fallback:
    #   configured pipeline -> v1 pipeline -> no pipeline
    requested_pipeline = search_pipeline if (query and use_semantic and search_pipeline) else None
    attempts: list[Optional[str]] = []
    if requested_pipeline:
        attempts.append(requested_pipeline)
    if requested_pipeline != "eufb-hybrid-v1":
        attempts.append("eufb-hybrid-v1")
    attempts.append(None)

    # Deduplicate while preserving order
    seen: set[Optional[str]] = set()
    pipeline_attempts: list[Optional[str]] = []
    for p in attempts:
        if p not in seen:
            seen.add(p)
            pipeline_attempts.append(p)

    response = None
    used_pipeline: Optional[str] = None
    last_error: Optional[TransportError] = None

    try:
        for attempt_pipeline in pipeline_attempts:
            params = _with_pipeline_params(attempt_pipeline)
            try:
                response = client.search(
                    index=index_name,
                    body=search_query,
                    params=params or None,
                )
                used_pipeline = attempt_pipeline
                break
            except TransportError as e:
                last_error = e
                # If this was the final attempt, re-raise below as HTTPException.
                logger.warning(
                    "Hybrid search attempt failed (pipeline=%s): %s",
                    attempt_pipeline if attempt_pipeline else "none",
                    e,
                )
                continue

        if response is None:
            if last_error is not None:
                raise last_error
            raise HTTPException(status_code=502, detail={
                "message": "OpenSearch hybrid search failed (no response from fallback attempts)",
            })

        hits = response.get("hits", {}).get("hits", [])
        logger.info(
            "[HYBRID DEBUG] returned_hits=%d page_size=%d from=%d pipeline_used=%s",
            len(hits),
            page_size,
            from_offset,
            used_pipeline if used_pipeline else "none",
        )

    except TransportError as e:
        logger.exception("Hybrid search failed after fallback attempts: %s", e)
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch hybrid search failed (all pipeline fallbacks exhausted)",
            "opensearch": detail,
        })

    aggs = response.get("aggregations", {})
    total_parents_from_agg = aggs.get("unique_parents_total", {}).get("value") if aggs else None

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=page_size)
    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped

    # Always add pipeline metadata for observability.
    response_meta = response.setdefault("_meta", {})
    response_meta["requested_pipeline"] = requested_pipeline
    response_meta["used_pipeline"] = used_pipeline
    response_meta["pipeline_attempts"] = [p if p is not None else "none" for p in pipeline_attempts]

    if debug_profile or debug_explain:
        debug_bundle = {
            "request_flags": {
                "debug_profile": bool(debug_profile),
                "debug_explain": bool(debug_explain),
            },
            "opensearch_request": {
                "index": index_name,
                "params": _with_pipeline_params(used_pipeline),
                "body": search_query,
            },
            "pipeline_debug": {
                "requested_pipeline": requested_pipeline,
                "used_pipeline": used_pipeline,
                "pipeline_attempts": [p if p is not None else "none" for p in pipeline_attempts],
            },
        }
        response["_debug_bundle"] = debug_bundle

    return response
