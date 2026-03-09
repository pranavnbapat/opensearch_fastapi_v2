# services/neural_search_relevant_advanced.py

import re

from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple

from pydantic import BaseModel
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError

from services.utils import PAGE_SIZE, client, group_hits_by_parent, build_sort


class SortBy(str, Enum):
    score_desc = "score_desc"
    score_asc = "score_asc"

    ko_created_at_desc = "ko_created_at_desc"
    ko_created_at_asc = "ko_created_at_asc"
    ko_updated_at_desc = "ko_updated_at_desc"
    ko_updated_at_asc = "ko_updated_at_asc"

    proj_created_at_desc = "proj_created_at_desc"
    proj_created_at_asc = "proj_created_at_asc"
    proj_updated_at_desc = "proj_updated_at_desc"
    proj_updated_at_asc = "proj_updated_at_asc"


NUMERIC_SORT_MAP = {
    1: SortBy.score_desc,
    2: SortBy.score_asc,
    3: SortBy.ko_created_at_desc,
    4: SortBy.ko_created_at_asc,
    5: SortBy.ko_updated_at_desc,
    6: SortBy.ko_updated_at_asc,
    7: SortBy.proj_created_at_desc,
    8: SortBy.proj_created_at_asc,
    9: SortBy.proj_updated_at_desc,
    10: SortBy.proj_updated_at_asc,
}

TEXT_FIELDS_BROAD = [
    "title.en^10",
    "subtitle.en^9",
    "description.en^6",
    "keywords.en^4",
    "content_chunk.en^2",
]

TEXT_FIELDS_PRECISION = [
    "title.en^12",
    "subtitle.en^10",
    "description.en^7",
    "keywords.en^8",
]

TEXT_FIELDS_PHRASE = [
    "title.en^14",
    "subtitle.en^12",
    "description.en^7",
]

TOKEN_RE = re.compile(
    r'\+?[A-Za-z_]+:"[^"]+"|\+?[A-Za-z_]+:[^\s()]+|\+?[A-Za-z_]+:|"[^"]+"|--project|-project|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+',
)

TERM_PREFIX_RE = re.compile(r"^(\+?)([A-Za-z_]+):(.*)$")

FIELD_QUERY_MAP = {
    "title": {"text_fields": ["title.en^12"], "phrase_fields": ["title.en^16"], "embedding": "title_embedding"},
    "subtitle": {"text_fields": ["subtitle.en^10"], "phrase_fields": ["subtitle.en^14"], "embedding": "subtitle_embedding"},
    "description": {"text_fields": ["description.en^8"], "phrase_fields": ["description.en^10"], "embedding": "description_embedding"},
    "keywords": {"text_fields": ["keywords.en^10"], "phrase_fields": ["keywords.en^12"], "embedding": "keywords_embedding"},
    "content": {"text_fields": ["content_chunk.en^4"], "phrase_fields": ["content_chunk.en^6"], "embedding": "content_embedding"},
}

FIELD_FILTER_MAP = {
    "theme": "themes",
    "themes": "themes",
    "location": "locations",
    "locations": "locations",
    "language": "languages",
    "languages": "languages",
    "topic": "topics",
    "topics": "topics",
    "category": "category",
    "type": "project_type",
    "project_type": "project_type",
    "acronym": "project_acronym",
}

MODE_VALUES = {"strict", "broad", "semantic", "lexical"}


def coerce_sort(raw: Union[None, int, str, SortBy]) -> str:
    if raw is None:
        return SortBy.score_desc.value
    if isinstance(raw, SortBy):
        return raw.value
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
    advanced: Optional[bool] = True
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


def _tokenize_query(query: str) -> List[str]:
    return TOKEN_RE.findall(query or "")


def _normalize_term(token: str) -> str:
    token = (token or "").strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1].strip()
    return token


def _split_prefixed_token(token: str) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    m = TERM_PREFIX_RE.match(token)
    if not m:
        return None, None, None
    required, field, remainder = m.groups()
    return (required == "+"), field.lower(), _normalize_term(remainder)


def _parse_project_filters(tokens: List[str]) -> Tuple[List[str], List[str]]:
    project_filters: List[str] = []
    cleaned: List[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        lower = token.lower()
        if lower in {"-project", "--project"}:
            i += 1
            if i < len(tokens):
                value = _normalize_term(tokens[i])
                if value:
                    project_filters.append(value)
            i += 1
            continue
        cleaned.append(token)
        i += 1

    return cleaned, project_filters


def _parse_mode_controls(tokens: List[str]) -> Tuple[List[str], Optional[str]]:
    cleaned: List[str] = []
    mode: Optional[str] = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        required, field, remainder = _split_prefixed_token(token)
        if field == "mode":
            if remainder:
                candidate = remainder.lower()
            else:
                i += 1
                candidate = _normalize_term(tokens[i]).lower() if i < len(tokens) else ""
            if candidate in MODE_VALUES:
                mode = candidate
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, mode


def _consume_term_tokens(stream: "_TokenStream") -> Dict[str, Any]:
    parts: List[str] = []
    required_field = False
    scoped_field: Optional[str] = None

    first = stream.peek()
    req, field, remainder = _split_prefixed_token(first or "")
    if field and field != "mode":
        stream.next()
        required_field = bool(req)
        scoped_field = field
        if remainder:
            parts.append(remainder)
    elif first is not None:
        pass

    while True:
        token = stream.peek()
        if token is None or token in {"(", ")"} or token.upper() in {"AND", "OR", "NOT"}:
            break
        req, field, remainder = _split_prefixed_token(token)
        if parts and field is not None:
            break
        if field is not None and scoped_field is None and field != "mode":
            stream.next()
            required_field = bool(req)
            scoped_field = field
            if remainder:
                parts.append(remainder)
            continue
        parts.append(_normalize_term(stream.next() or ""))
    value = " ".join(part for part in parts if part).strip()
    node_type = "FILTER" if scoped_field in FIELD_FILTER_MAP or scoped_field == "project" else "TERM"
    if scoped_field == "mode":
        node_type = "TERM"
        scoped_field = None
    return {
        "type": node_type,
        "value": value,
        "field": scoped_field,
        "required": required_field,
    }


class _TokenStream:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def next(self) -> Optional[str]:
        token = self.peek()
        if token is not None:
            self.pos += 1
        return token

    def at_end(self) -> bool:
        return self.peek() is None


def _parse_primary(stream: _TokenStream) -> Dict[str, Any]:
    token = stream.peek()
    if token is None:
        raise ValueError("Unexpected end of query")

    if token == "(":
        stream.next()
        node = _parse_or(stream)
        if stream.next() != ")":
            raise ValueError("Unclosed parenthesis")
        return node

    if token.upper() in {"AND", "OR", ")"}:
        raise ValueError(f"Unexpected token: {token}")

    term = _consume_term_tokens(stream)
    if not term["value"]:
        raise ValueError("Empty term")
    return term


def _parse_not(stream: _TokenStream) -> Dict[str, Any]:
    token = stream.peek()
    if token and token.upper() == "NOT":
        stream.next()
        return {"type": "NOT", "child": _parse_not(stream)}
    return _parse_primary(stream)


def _parse_and(stream: _TokenStream) -> Dict[str, Any]:
    node = _parse_not(stream)
    children = [node]

    while True:
        token = stream.peek()
        if token is None or token == ")" or token.upper() == "OR":
            break

        if token.upper() == "AND":
            stream.next()
        # implicit AND is also allowed when two operands are adjacent
        children.append(_parse_not(stream))

    if len(children) == 1:
        return children[0]
    return {"type": "AND", "children": children}


def _parse_or(stream: _TokenStream) -> Dict[str, Any]:
    node = _parse_and(stream)
    children = [node]

    while True:
        token = stream.peek()
        if token is None or token == ")":
            break
        if token.upper() != "OR":
            break
        stream.next()
        children.append(_parse_and(stream))

    if len(children) == 1:
        return children[0]
    return {"type": "OR", "children": children}


def _collect_operators(tokens: List[str]) -> List[str]:
    operators: List[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            operators.append(upper)
    return operators


def _ast_to_debug(node: Dict[str, Any]) -> Dict[str, Any]:
    node_type = node["type"]
    if node_type in {"TERM", "FILTER"}:
        out = {"type": node_type, "value": node["value"]}
        if node.get("field"):
            out["field"] = node["field"]
        if node.get("required"):
            out["required"] = True
        return out
    if node_type == "NOT":
        return {"type": "NOT", "child": _ast_to_debug(node["child"])}
    return {"type": node_type, "children": [_ast_to_debug(child) for child in node["children"]]}


def _parse_advanced_query(query: str) -> Dict[str, Any]:
    raw_tokens = _tokenize_query(query)
    tokens, project_filters = _parse_project_filters(raw_tokens)
    tokens, mode = _parse_mode_controls(tokens)
    operators_used = _collect_operators(tokens)

    if not tokens:
        return {
            "raw_query": query,
            "tokens": [],
            "operators_used": [],
            "project_filters": project_filters,
            "mode": mode,
            "has_boolean_syntax": False,
            "ast": None,
        }

    stream = _TokenStream(tokens)
    ast = _parse_or(stream)
    if not stream.at_end():
        raise ValueError(f"Unexpected trailing token: {stream.peek()}")

    return {
        "raw_query": query,
        "tokens": tokens,
        "operators_used": operators_used,
        "project_filters": project_filters,
        "mode": mode,
        "has_boolean_syntax": bool(operators_used or "(" in tokens or ")" in tokens),
        "ast": _ast_to_debug(ast),
        "_ast_internal": ast,
    }


def _term_is_phrase(term: str) -> bool:
    return len(term.split()) > 1


def _dynamic_k(term: str) -> tuple[int, int, int, int, int]:
    q_len = len(term.split())
    k_content = 300 if q_len >= 3 else 200
    k_title = 100 if q_len >= 3 else 60
    k_subtitle = 120 if q_len >= 3 else 80
    k_description = 180 if q_len >= 3 else 120
    k_keywords = 140 if q_len >= 3 else 90
    return k_title, k_subtitle, k_description, k_keywords, k_content


def _semantic_dismax(term: str, model_id: str) -> Dict[str, Any]:
    k_title, k_subtitle, k_description, k_keywords, k_content = _dynamic_k(term)
    return {
        "dis_max": {
            "tie_breaker": 0.1,
            "queries": [
                {
                    "neural": {
                        "title_embedding": {
                            "query_text": term,
                            "model_id": model_id,
                            "k": k_title,
                            "boost": 1.4,
                        }
                    }
                },
                {
                    "neural": {
                        "subtitle_embedding": {
                            "query_text": term,
                            "model_id": model_id,
                            "k": k_subtitle,
                            "boost": 0.7,
                        }
                    }
                },
                {
                    "neural": {
                        "description_embedding": {
                            "query_text": term,
                            "model_id": model_id,
                            "k": k_description,
                            "boost": 1.1,
                        }
                    }
                },
                {
                    "neural": {
                        "keywords_embedding": {
                            "query_text": term,
                            "model_id": model_id,
                            "k": k_keywords,
                            "boost": 0.4,
                        }
                    }
                },
                {
                    "neural": {
                        "content_embedding": {
                            "query_text": term,
                            "model_id": model_id,
                            "k": k_content,
                            "boost": 1.0,
                        }
                    }
                },
            ],
        }
    }


def _field_semantic_query(term: str, model_id: str, embedding_field: str) -> Dict[str, Any]:
    k_title, k_subtitle, k_description, k_keywords, k_content = _dynamic_k(term)
    k_map = {
        "title_embedding": k_title,
        "subtitle_embedding": k_subtitle,
        "description_embedding": k_description,
        "keywords_embedding": k_keywords,
        "content_embedding": k_content,
    }
    return {
        "neural": {
            embedding_field: {
                "query_text": term,
                "model_id": model_id,
                "k": k_map.get(embedding_field, k_content),
                "boost": 1.0,
            }
        }
    }


def _lexical_clause(term: str, *, operator: str = "or", boost: float = 1.0) -> Dict[str, Any]:
    msm = "70%" if len(term.split()) >= 5 and operator == "or" else None
    return {
        "multi_match": {
            "query": term,
            "fields": TEXT_FIELDS_BROAD,
            "operator": operator,
            **({"minimum_should_match": msm} if msm else {}),
            "type": "best_fields",
            "boost": boost,
        }
    }


def _field_lexical_clause(
    term: str,
    *,
    text_fields: List[str],
    operator: str = "or",
    boost: float = 1.0,
    phrase: bool = False,
) -> Dict[str, Any]:
    query = {
        "query": term,
        "fields": text_fields,
        "boost": boost,
    }
    if phrase:
        query["type"] = "phrase"
    else:
        query["operator"] = operator
        query["type"] = "best_fields"
    return {"multi_match": query}


def _precision_boosts(term: str) -> List[Dict[str, Any]]:
    boosts: List[Dict[str, Any]] = [
        _lexical_clause(term, operator="and", boost=1.8),
        {"term": {"project_acronym": {"value": term, "boost": 6.0}}},
    ]
    if _term_is_phrase(term):
        boosts.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": TEXT_FIELDS_PHRASE,
                    "type": "phrase",
                    "boost": 2.2,
                }
            }
        )
    return boosts


def _field_filter_query(field: str, value: str) -> Dict[str, Any]:
    if field == "project":
        return {
            "bool": {
                "should": [
                    {"term": {"project_acronym": value}},
                    {"match_phrase": {"project_name": value}},
                    {"match": {"project_name": {"query": value, "operator": "and"}}},
                ],
                "minimum_should_match": 1,
            }
        }
    mapped = FIELD_FILTER_MAP[field]
    return {"term": {mapped: value}}


def _mode_semantic_enabled(mode: Optional[str], use_semantic: bool) -> bool:
    if mode == "lexical":
        return False
    if mode == "semantic":
        return True
    return use_semantic


def _positive_term_query(
    term: str,
    *,
    model_id: str,
    use_semantic: bool,
    field: Optional[str] = None,
    required: bool = False,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    effective_semantic = _mode_semantic_enabled(mode, use_semantic)

    if field == "project" or field in FIELD_FILTER_MAP:
        return _field_filter_query(field, term)

    if field in FIELD_QUERY_MAP:
        cfg = FIELD_QUERY_MAP[field]
        lexical_operator = "and" if required or mode == "strict" or _term_is_phrase(term) else "or"
        lexical = _field_lexical_clause(term, text_fields=cfg["text_fields"], operator=lexical_operator, boost=1.2)
        should_clauses: List[Dict[str, Any]] = [lexical]
        if _term_is_phrase(term):
            should_clauses.append(_field_lexical_clause(term, text_fields=cfg["phrase_fields"], phrase=True, boost=2.2))
        if effective_semantic and not required:
            should_clauses.insert(0, _field_semantic_query(term, model_id, cfg["embedding"]))
        return {"bool": {"should": should_clauses, "minimum_should_match": 1}}

    lexical_operator = "and" if mode == "strict" or _term_is_phrase(term) else "or"
    lexical = _lexical_clause(term, operator=lexical_operator, boost=1.1)
    if mode == "semantic":
        return _semantic_dismax(term, model_id)
    if not effective_semantic:
        return {
            "bool": {
                "should": [
                    lexical,
                    {"term": {"project_acronym": {"value": term, "boost": 6.0}}},
                ],
                "minimum_should_match": 1,
            }
        }

    should_clauses = [_semantic_dismax(term, model_id)]
    if mode == "broad":
        should_clauses.append(_lexical_clause(term, operator="or", boost=1.0))
    else:
        should_clauses.append(lexical)
        should_clauses.extend(_precision_boosts(term))
    return {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1,
        }
    }


def _negative_term_query(term: str, *, field: Optional[str] = None) -> Dict[str, Any]:
    if field == "project" or field in FIELD_FILTER_MAP:
        return _field_filter_query(field, term)

    if field in FIELD_QUERY_MAP:
        cfg = FIELD_QUERY_MAP[field]
        clauses: List[Dict[str, Any]] = [
            _field_lexical_clause(term, text_fields=cfg["text_fields"], operator="and" if _term_is_phrase(term) else "or", boost=1.0),
        ]
        if _term_is_phrase(term):
            clauses.append(_field_lexical_clause(term, text_fields=cfg["phrase_fields"], phrase=True, boost=1.0))
        return {"bool": {"should": clauses, "minimum_should_match": 1}}

    clauses: List[Dict[str, Any]] = [
        _lexical_clause(term, operator="and" if _term_is_phrase(term) else "or", boost=1.0),
        {"term": {"project_acronym": {"value": term}}},
    ]
    if _term_is_phrase(term):
        clauses.append(
            {
                "multi_match": {
                    "query": term,
                    "fields": TEXT_FIELDS_PHRASE,
                    "type": "phrase",
                }
            }
        )
    return {
        "bool": {
            "should": clauses,
            "minimum_should_match": 1,
        }
    }


def _ast_to_query(
    node: Dict[str, Any],
    *,
    model_id: str,
    use_semantic: bool,
    mode: Optional[str],
    negated: bool = False,
) -> Dict[str, Any]:
    node_type = node["type"]

    if node_type in {"TERM", "FILTER"}:
        return _negative_term_query(node["value"], field=node.get("field")) if negated else _positive_term_query(
            node["value"],
            model_id=model_id,
            use_semantic=use_semantic,
            field=node.get("field"),
            required=bool(node.get("required")),
            mode=mode,
        )

    if node_type == "NOT":
        return _ast_to_query(node["child"], model_id=model_id, use_semantic=use_semantic, mode=mode, negated=not negated)

    if node_type == "AND":
        key = "must_not" if negated else "must"
        return {
            "bool": {
                key: [
                    _ast_to_query(child, model_id=model_id, use_semantic=use_semantic, mode=mode, negated=negated)
                    for child in node["children"]
                ]
            }
        }

    if node_type == "OR":
        if negated:
            return {
                "bool": {
                    "must_not": [
                        _ast_to_query(child, model_id=model_id, use_semantic=use_semantic, mode=mode, negated=False)
                        for child in node["children"]
                    ]
                }
            }
        return {
            "bool": {
                "should": [
                    _ast_to_query(child, model_id=model_id, use_semantic=use_semantic, mode=mode, negated=False)
                    for child in node["children"]
                ],
                "minimum_should_match": 1,
            }
        }

    raise ValueError(f"Unsupported AST node type: {node_type}")


def _project_scope_filters(project_filters: List[str]) -> List[Dict[str, Any]]:
    scoped_filters: List[Dict[str, Any]] = []
    for project in project_filters:
        scoped_filters.append(
            {
                "bool": {
                    "should": [
                        {"term": {"project_acronym": project}},
                        {"match_phrase": {"project_name": project}},
                        {"match": {"project_name": {"query": project, "operator": "and"}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return scoped_filters


def _build_advanced_query_part(
    query: str,
    *,
    model_id: str,
    use_semantic: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    parsed = _parse_advanced_query(query)
    ast = parsed.pop("_ast_internal", None)
    mode = parsed.get("mode")

    if ast is None:
        return {"match_all": {}}, parsed, _project_scope_filters(parsed.get("project_filters", []))

    return (
        _ast_to_query(ast, model_id=model_id, use_semantic=use_semantic, mode=mode),
        parsed,
        _project_scope_filters(parsed.get("project_filters", [])),
    )


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
    page_size = int(size_override) if isinstance(size_override, int) and size_override > 0 else PAGE_SIZE
    from_offset = (page - 1) * page_size

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

    if not query:
        query_part = {"match_all": {}}
        parsed_query = {
            "raw_query": query,
            "tokens": [],
            "operators_used": [],
            "clauses": [{"must": [], "must_not": []}],
            "has_boolean_syntax": False,
        }
    else:
        try:
            query_part, parsed_query, advanced_filters = _build_advanced_query_part(
                query,
                model_id=model_id,
                use_semantic=use_semantic,
            )
            filter_conditions.extend(advanced_filters)
        except ValueError as e:
            return {
                "took": 0,
                "timed_out": False,
                "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                "aggregations": {"unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}},
                "grouped": {"total_parents": 0, "parents": []},
                "_meta": {
                    "error": f"Advanced query parse error: {e}",
                    "advanced_search": {
                        "enabled": True,
                        "query_parser": "boolean_v2",
                        "parse_error": str(e),
                    },
                },
            }

    if not query:
        must_not = []
    else:
        must_not = [{"term": {"chunk_index": -1}}]

    raw_sort = (filters or {}).get("sort_by")
    sort_key = coerce_sort(raw_sort)

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
                "must_not": must_not,
            }
        },
        "collapse": {
            "field": "parent_id",
            "inner_hits": {
                "name": "best_chunks",
                "size": 3,
                "_source": ["chunk_index", "content_chunk"],
            }
        },
        "aggs": {
            "unique_parents_total": {
                "cardinality": {
                    "field": "parent_id",
                    "precision_threshold": 40000,
                }
            },
            "top_projects": {
                "terms": {
                    "field": "project_id",
                    "size": 3,
                    "order": {"unique_parents": "desc"},
                },
                "aggs": {
                    "unique_parents": {
                        "cardinality": {
                            "field": "parent_id",
                            "precision_threshold": 40000,
                        }
                    }
                },
            },
        },
    }

    if debug_profile:
        search_query["profile"] = True

    if debug_explain:
        search_query["explain"] = True

    search_query["sort"] = build_sort(sort_key, has_query=bool(query))

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
        return {
            "took": 0,
            "timed_out": False,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "aggregations": {"unique_parents_total": {"value": 0}, "top_projects": {"buckets": []}},
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"OpenSearch error: {type(e).__name__}", "transport_error": True},
        }

    response = client.search(index=index_name, body=search_query)

    aggs = response.get("aggregations", {})
    total_parents_from_agg = aggs.get("unique_parents_total", {}).get("value") if aggs else None

    hits = response["hits"]["hits"]
    grouped = group_hits_by_parent(hits, parents_size=page_size)

    if total_parents_from_agg is not None:
        grouped["total_parents"] = int(total_parents_from_agg)

    response["grouped"] = grouped
    response["_meta"] = {
            "advanced_search": {
                "enabled": True,
                "query_parser": "boolean_v2",
                "use_semantic": use_semantic,
                "parsed_query": parsed_query,
                "not_behavior": "lexical_only_exclusion",
                "project_scope_behavior": "filter_project_acronym_or_project_name",
            }
        }

    return response
