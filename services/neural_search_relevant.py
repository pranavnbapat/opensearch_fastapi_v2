# services/neural_search_relevant.py

import re
from enum import Enum
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError

from services.utils import (PAGE_SIZE, remove_stopwords_from_query, client,
                            group_hits_by_parent, build_sort, infer_query_intent)


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
    sort_by: Optional[Union[SortBy, int, str]] = SortBy.score_desc
    access_token: Optional[str] = None


QUESTION_PREFIX_RE = re.compile(
    r"^(what|which|when|where|why|how|can|could|should|would|is|are|do|does|did|who)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:kg/ha|t/ha|mm/yr|mm|cm|ha|m³/ha|m3/ha|c/n|ph|%)\b", re.IGNORECASE)
CODEISH_RE = re.compile(r"\b[A-Z]{2,}[-_/]?[A-Z0-9]{2,}\b")
NON_WORD_RE = re.compile(r"[^\w\s/%.-]+", re.UNICODE)
EXPLICIT_SINGLE_QUOTE_RE = re.compile(r"(?:^|\\s)'[^']{2,}'(?:\\s|$)")
QUESTION_NOISE_PREFIXES = [
    "what is", "what are", "what's", "which", "when", "where", "why", "how do i", "how can i",
    "how can", "how do", "can i", "could i", "should i", "would it", "is it", "are there",
    "i am looking for", "looking for", "find", "we are looking for", "we are trying to",
    "i'm wondering if", "i am wondering if", "can you explain", "can you check if", "does anyone know if",
    "so like", "um", "after the last treatment", "what about",
]
FULL_SENTENCE_LEADERS = [
    "we are trying to", "we are considering", "we are seeing", "we applied", "we used", "we noticed",
    "i used to", "i noticed", "i am looking for", "i'm looking for", "after the recent", "after the last",
    "based on", "looking for", "find", "the soil", "the new", "the application", "this study", "this project",
]
QUESTION_GENERIC_TOKENS = {
    "what", "what's", "which", "when", "where", "why", "how", "can", "could", "should", "would",
    "best", "recommended", "recommend", "explain", "difference", "signs", "mean", "means", "use",
    "using", "help", "helps", "still", "apply", "applied", "time", "way", "there", "any",
}


def _normalize_query_spaces(query: str) -> str:
    return " ".join((query or "").strip().split())


def _strip_leading_noise(query: str) -> str:
    cleaned = _normalize_query_spaces(query)
    lowered = cleaned.lower()
    for prefix in QUESTION_NOISE_PREFIXES:
        if lowered.startswith(prefix + " "):
            return cleaned[len(prefix):].strip(" ,.:;!?-")
    for prefix in FULL_SENTENCE_LEADERS:
        if lowered.startswith(prefix + " "):
            return cleaned[len(prefix):].strip(" ,.:;!?-")
    return cleaned


def _rewrite_query_for_retrieval(query: str) -> str:
    cleaned = _strip_leading_noise(query)
    cleaned = NON_WORD_RE.sub(" ", cleaned)
    cleaned = _normalize_query_spaces(cleaned)

    filtered = remove_stopwords_from_query(cleaned)
    filtered = _normalize_query_spaces(filtered)

    original_tokens = cleaned.split()
    filtered_tokens = filtered.split()

    # Keep the rewrite short and information-dense for lexical retrieval.
    chosen = filtered_tokens[:10] if filtered_tokens else original_tokens[:10]
    rewritten = " ".join(chosen).strip()
    return rewritten or _normalize_query_spaces(query)


def _rewrite_question_focus_query(query: str) -> str:
    cleaned = _strip_leading_noise(query)
    cleaned = NON_WORD_RE.sub(" ", cleaned)
    cleaned = _normalize_query_spaces(cleaned)

    filtered = remove_stopwords_from_query(cleaned)
    filtered = _normalize_query_spaces(filtered)
    tokens = [tok for tok in filtered.split() if tok.lower() not in QUESTION_GENERIC_TOKENS]

    if len(tokens) >= 3:
        return " ".join(tokens[:10])
    return filtered or cleaned


def _build_attempts_for_profile(
    *,
    normalized_query: str,
    rewritten_query: str,
    use_semantic: bool,
    query_mode: str,
) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    question_focus_query = _rewrite_question_focus_query(normalized_query) if query_mode == "question" else ""

    primary_query = rewritten_query if (not use_semantic and rewritten_query) else normalized_query
    attempts.append(
        {
            "label": "primary",
            "query_text": primary_query,
            "use_semantic": use_semantic,
        }
    )

    if query_mode in {"question", "long_descriptive"}:
        if query_mode == "question" and question_focus_query and question_focus_query.lower() != primary_query.lower():
            attempts.insert(
                0,
                {
                    "label": "question_focus_lexical",
                    "query_text": question_focus_query,
                    "use_semantic": False,
                }
            )
            attempts.append(
                {
                    "label": "question_focus_semantic",
                    "query_text": question_focus_query,
                    "use_semantic": True,
                }
            )
        if rewritten_query and rewritten_query.lower() != primary_query.lower():
            attempts.append(
                {
                    "label": "rewrite_semantic",
                    "query_text": rewritten_query,
                    "use_semantic": True,
                }
            )
        if normalized_query and normalized_query.lower() != primary_query.lower():
            attempts.append(
                {
                    "label": "original_lexical",
                    "query_text": normalized_query,
                    "use_semantic": False,
                }
            )
        attempts.append(
            {
                "label": "original_semantic",
                "query_text": normalized_query,
                "use_semantic": True,
            }
        )
    else:
        if rewritten_query and rewritten_query.lower() != primary_query.lower():
            attempts.append(
                {
                    "label": "rewrite_lexical",
                    "query_text": rewritten_query,
                    "use_semantic": False,
                }
            )
        if normalized_query and normalized_query.lower() != primary_query.lower():
            attempts.append(
                {
                    "label": "original_semantic",
                    "query_text": normalized_query,
                    "use_semantic": True,
                }
            )

    return attempts


def build_default_query_profile(query: str) -> Dict[str, Any]:
    normalized_query = _normalize_query_spaces(query)
    token_count = len(normalized_query.split())
    (
        base_use_semantic,
        looks_like_code_or_id,
        looks_like_acronym,
        _looks_like_quoted,
        very_short,
    ) = infer_query_intent(
        normalized_query,
        code_hint_env="QUERY_CODE_HINT_REGEX",
        acronym_min_len_env="QUERY_ACRONYM_MIN_LEN",
        acronym_max_len_env="QUERY_ACRONYM_MAX_LEN",
        acronym_min_caps_env="QUERY_ACRONYM_MIN_CAPS",
    )

    looks_like_url = bool(URL_RE.search(normalized_query))
    has_unit_shorthand = bool(UNIT_RE.search(normalized_query))
    has_codeish_token = bool(CODEISH_RE.search(normalized_query))
    looks_like_quoted = ('"' in normalized_query) or bool(EXPLICIT_SINGLE_QUOTE_RE.search(normalized_query))
    is_question = normalized_query.endswith("?") or bool(QUESTION_PREFIX_RE.search(normalized_query))
    long_query = token_count >= 8
    sentence_like = token_count >= 7 and not very_short
    rewritten_query = _rewrite_query_for_retrieval(normalized_query)
    rewrite_changed = rewritten_query.lower() != normalized_query.lower()

    if not normalized_query:
        query_mode = "match_all"
        use_semantic = False
        reason = "empty_query"
    elif has_unit_shorthand:
        query_mode = "unit_or_constraint"
        use_semantic = False
        reason = "contains_unit_or_numeric_constraint"
    elif looks_like_url or looks_like_code_or_id or looks_like_acronym or looks_like_quoted or has_codeish_token:
        query_mode = "identifier_or_title"
        use_semantic = False
        reason = "exact_identifier_or_title_like"
    elif is_question:
        query_mode = "question"
        use_semantic = False
        reason = "natural_language_question"
    elif long_query or sentence_like:
        query_mode = "long_descriptive"
        use_semantic = False
        reason = "long_descriptive_query"
    else:
        query_mode = "broad_topic"
        use_semantic = base_use_semantic
        reason = "short_broad_topic_query"

    attempts = _build_attempts_for_profile(
        normalized_query=normalized_query,
        rewritten_query=rewritten_query,
        use_semantic=use_semantic,
        query_mode=query_mode,
    )

    deduped_attempts: List[Dict[str, Any]] = []
    seen_attempts = set()
    for attempt in attempts:
        key = (attempt["query_text"].lower(), bool(attempt["use_semantic"]))
        if key in seen_attempts:
            continue
        seen_attempts.add(key)
        deduped_attempts.append(attempt)

    return {
        "query_mode": query_mode,
        "use_semantic": use_semantic,
        "reason": reason,
        "token_count": token_count,
        "is_question": is_question,
        "long_query": long_query,
        "very_short": very_short,
        "looks_like_code_or_id": looks_like_code_or_id,
        "looks_like_acronym": looks_like_acronym,
        "looks_like_quoted": looks_like_quoted,
        "looks_like_url": looks_like_url,
        "has_unit_shorthand": has_unit_shorthand,
        "rewrite_changed": rewrite_changed,
        "rewritten_query": rewritten_query,
        "attempts": deduped_attempts,
    }


def _build_query_part(query_text: str, model_id: str, use_semantic: bool, profile: Dict[str, Any]) -> Dict[str, Any]:
    q_len = len((query_text or "").split())

    k_content = 300 if q_len >= 3 else 200
    k_title = 100 if q_len >= 3 else 60
    k_subtitle = 120 if q_len >= 3 else 80
    k_description = 180 if q_len >= 3 else 120
    k_keywords = 140 if q_len >= 3 else 90
    bm25_msm = "70%" if q_len >= 5 else None

    if not query_text:
        return {"match_all": {}}

    lexical_and_query = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "title.en^12",
                "subtitle.en^10",
                "description.en^7",
                "keywords.en^8",
                "content_chunk.en^3",
            ],
            "operator": "and",
            "type": "best_fields",
            "boost": 1.8,
        }
    }
    lexical_or_query = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "title.en^10",
                "subtitle.en^9",
                "description.en^6",
                "keywords.en^5",
                "content_chunk.en^2",
            ],
            "operator": "or",
            **({"minimum_should_match": bm25_msm} if bm25_msm else {}),
            "type": "best_fields",
            "boost": 1.0,
        }
    }
    lexical_cross_fields_query = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "title.en^10",
                "subtitle.en^9",
                "description.en^7",
                "keywords.en^6",
                "content_chunk.en^3",
            ],
            "type": "cross_fields",
            "operator": "or",
            "minimum_should_match": "60%",
            "boost": 1.4,
        }
    }
    lexical_phrase_query = {
        "multi_match": {
            "query": query_text,
            "fields": [
                "title.en^14",
                "subtitle.en^12",
                "description.en^7",
            ],
            "type": "phrase",
            "boost": 2.4,
        }
    }
    project_acronym_boost = {"term": {"project_acronym": {"value": query_text, "boost": 6.0}}}

    if use_semantic:
        semantic_dismax = {
            "dis_max": {
                "tie_breaker": 0.1,
                "queries": [
                    {"neural": {"title_embedding": {"query_text": query_text, "model_id": model_id, "k": k_title, "boost": 1.4}}},
                    {"neural": {"subtitle_embedding": {"query_text": query_text, "model_id": model_id, "k": k_subtitle, "boost": 0.7}}},
                    {"neural": {"description_embedding": {"query_text": query_text, "model_id": model_id, "k": k_description, "boost": 1.1}}},
                    {"neural": {"keywords_embedding": {"query_text": query_text, "model_id": model_id, "k": k_keywords, "boost": 0.4}}},
                    {"neural": {"content_embedding": {"query_text": query_text, "model_id": model_id, "k": k_content, "boost": 1.0}}},
                ]
            }
        }
        return {
            "bool": {
                "must": [semantic_dismax],
                "should": [
                    lexical_or_query,
                    lexical_and_query,
                    lexical_phrase_query,
                    project_acronym_boost,
                ],
            }
        }

    lexical_should = [
        lexical_and_query,
        lexical_or_query,
        lexical_phrase_query,
        project_acronym_boost,
    ]
    if profile.get("query_mode") in {"question", "long_descriptive", "unit_or_constraint"}:
        lexical_should.insert(0, lexical_cross_fields_query)
        lexical_should.append(
            {
                "match_phrase_prefix": {
                    "title.en": {
                        "query": query_text,
                        "boost": 1.6,
                    }
                }
            }
        )
    if profile.get("query_mode") == "question":
        lexical_should.append(
            {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "title.en^11",
                        "subtitle.en^10",
                        "description.en^8",
                        "keywords.en^7",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": "50%",
                    "boost": 1.8,
                }
            }
        )
    return {
        "bool": {
            "should": lexical_should,
            "minimum_should_match": 1,
        }
    }


def _build_search_query(
    *,
    query_part: Dict[str, Any],
    filter_conditions: List[Dict[str, Any]],
    must_not: List[Dict[str, Any]],
    page_size: int,
    from_offset: int,
    sort_key: str,
    debug_profile: bool,
    debug_explain: bool,
    has_query: bool,
) -> Dict[str, Any]:
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
        search_query["profile"] = True

    if debug_explain:
        search_query["explain"] = True

    search_query["sort"] = build_sort(sort_key, has_query=has_query)
    return search_query


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
        query_profile: Optional[Dict[str, Any]] = None,
):
    # Allow caller (endpoint) to request "top-k parents" by fetching more per page
    page_size = int(size_override) if isinstance(size_override, int) and size_override > 0 else PAGE_SIZE

    # Pagination offset
    from_offset = (page - 1) * page_size
    profile = dict(query_profile or build_default_query_profile(query))
    use_semantic = bool(profile.get("use_semantic", use_semantic))

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

    if not query:
        must_not = []  # include meta docs so ALL KOs can show up
    else:
        must_not = [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_key = coerce_sort(raw_sort)

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

    attempt_summaries: List[Dict[str, Any]] = []
    response = None
    for attempt in (profile.get("attempts") or [{"label": "primary", "query_text": query, "use_semantic": use_semantic}]):
        attempt_query = str(attempt.get("query_text") or query).strip()
        attempt_use_semantic = bool(attempt.get("use_semantic"))
        query_part = _build_query_part(attempt_query, model_id, attempt_use_semantic, profile)
        search_query = _build_search_query(
            query_part=query_part,
            filter_conditions=filter_conditions,
            must_not=must_not,
            page_size=page_size,
            from_offset=from_offset,
            sort_key=sort_key,
            debug_profile=debug_profile,
            debug_explain=debug_explain,
            has_query=bool(attempt_query),
        )
        candidate_response = client.search(index=index_name, body=search_query)
        hits = ((candidate_response.get("hits") or {}).get("hits")) or []
        parent_count = len({((h.get("_source") or {}).get("parent_id") or h.get("_id")) for h in hits})
        attempt_summaries.append(
            {
                "label": attempt.get("label"),
                "query_text": attempt_query,
                "use_semantic": attempt_use_semantic,
                "hit_count": len(hits),
                "parent_count": parent_count,
            }
        )
        response = candidate_response
        if hits:
            profile["selected_attempt"] = dict(attempt_summaries[-1])
            break

    if response is None:
        response = {
            "took": 0,
            "timed_out": False,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "aggregations": {"unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}},
        }

    aggs = response.get("aggregations", {})
    total_parents_from_agg = (
        aggs.get("unique_parents_total", {}).get("value") if aggs else None
    )

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=page_size)

    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped
    response["_meta"] = response.get("_meta") or {}
    response["_meta"]["default_search"] = {
        "query_mode": profile.get("query_mode"),
        "reason": profile.get("reason"),
        "use_semantic": bool(profile.get("use_semantic")),
        "rewritten_query": profile.get("rewritten_query"),
        "selected_attempt": profile.get("selected_attempt"),
        "attempts": attempt_summaries,
    }

    return response
