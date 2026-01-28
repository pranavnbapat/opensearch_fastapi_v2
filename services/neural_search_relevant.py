# services/neural_search_relevant.py

from enum import Enum
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError

from services.utils import (PAGE_SIZE, remove_stopwords_from_query, client,
                            group_hits_by_parent, build_sort)


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

class RelevantSearchRequest(BaseModel):
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
    debug_analyze: Optional[bool] = False
    debug_field: Optional[str] = None
    sort_by: Optional[Union[SortBy, int, str]] = SortBy.score_desc
    access_token: Optional[str] = None


def neural_search_relevant(
        index_name: str,
        query: str,
        filters: Dict[str, Any],
        page: int,
        model_id: str,
        use_semantic: bool = True,
        size_override: Optional[int] = None,
        debug_profile: bool = False,
        debug_explain: bool = False,
):
    # Allow caller (endpoint) to request "top-k parents" by fetching more per page
    page_size = int(size_override) if isinstance(size_override, int) and size_override > 0 else PAGE_SIZE

    # Pagination offset
    from_offset = (page - 1) * page_size

    # --- Dynamic semantic recall ---
    # Candidate pool sizes for neural retrieval (independent of pagination)
    q_len = len(query.split())
    k_content = 300 if q_len >= 3 else 200
    k_title = 100 if q_len >= 3 else 60
    k_subtitle = 120 if q_len >= 3 else 80
    k_description = 180 if q_len >= 3 else 120
    k_keywords = 140 if q_len >= 3 else 90

    bm25_msm = "70%" if q_len >= 5 else None

    # Filter conditions
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

    # Decide query type based on whether search_term is provided
    if not query:
        query_part = {"match_all": {}}
    elif use_semantic:
        # Semantic: use dis_max so the strongest semantic field wins, with a small bonus if others agree.
        semantic_dismax = {
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

        bm25_multi_match = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title.en^10",
                    "subtitle.en^9",
                    "description.en^6",
                    "keywords.en^4",
                    "content_chunk.en^2",
                ],
                "operator": "or",
                **({"minimum_should_match": bm25_msm} if bm25_msm else {}),
                "type": "best_fields",
                "boost": 1.1
            }
        }

        query_part = {
            "bool": {
                "should": [
                    semantic_dismax,
                    bm25_multi_match,
                ],
                "minimum_should_match": 1
            }
        }
    else:
        filtered_query = query

        query_part = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": filtered_query,
                            # TEXT fields only
                            "fields": [
                                "title.en^10",
                                "subtitle.en^9",
                                "description.en^6",
                                "keywords.en^4",
                                "content_chunk.en^2",
                            ],
                            "operator": "or",
                            "type": "best_fields",
                            "boost": 1.0
                        }
                    },
                    {
                        "neural": {
                            "content_embedding": {
                                "query_text": query,
                                "model_id": model_id,
                                "k": k_content,
                                "boost": 0.3
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }

    if not query:
        must_not = []  # include meta docs so ALL KOs can show up
    else:
        must_not = [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_key = coerce_sort(raw_sort)

    # Final OpenSearch query
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
        "query": {
            "bool": {
                "must": [query_part],
                "filter": filter_conditions,
                "must_not": must_not
            }
        },
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
                    "order": { "unique_parents": "desc" }
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
        search_query["profile"] = True

    if debug_explain:
        search_query["explain"] = True

    search_query["sort"] = build_sort(sort_key, has_query=bool(query))

    # ---- Fail fast if index doesn't exist (e.g., dev suffix not created) ----
    try:
        # HEAD /<index>
        exists = client.indices.exists(index=index_name, request_timeout=2)
        if not exists:
            return {
                "took": 0,
                "timed_out": False,
                "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                "aggregations": { "unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}, },
                "grouped": {"total_parents": 0, "parents": []},
                "_meta": {"error": f"Index not found: {index_name}","index_missing": True},
            }
    except (ConnectionTimeout, ConnectionError) as e:
    # Don't fall through to normal search (it will just retry/hang)
        return {
            "took": 0,
            "timed_out": True,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "aggregations": { "unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}, },
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"OpenSearch unreachable: {type(e).__name__}", "unreachable": True},
        }
    except TransportError as e:
    # covers other HTTP-ish issues; keep it explicit
        return {
            "took": 0,
            "timed_out": False,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "aggregations": { "unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}, },
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"OpenSearch error: {type(e).__name__}", "transport_error": True},
        }

    response = client.search(index=index_name, body=search_query)

    aggs = response.get("aggregations", {})
    total_parents_from_agg = (
        aggs.get("unique_parents_total", {}).get("value") if aggs else None
    )

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=page_size)

    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped

    return response
