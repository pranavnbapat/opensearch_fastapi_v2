# services/recommender_knn.py

import logging
from typing import Dict, Any, Optional, List, Tuple, Literal, cast

from fastapi import HTTPException
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError
from pydantic import BaseModel

from services.utils import client, group_hits_by_parent


logger = logging.getLogger(__name__)

# Prefer KO-level semantic signal for "related items".
# Order matters: first available vector will be used.
EMBEDDING_FIELDS: Dict[str, str] = {
    "content": "content_embedding",
    "description": "description_embedding",
    "title": "title_embedding",
    "keywords": "keywords_embedding",
}

class RecommendKNNRequest(BaseModel):
    parent_id: str
    k: Optional[int] = 6
    k_candidates: Optional[int] = 0
    dev: Optional[bool] = True
    model: Optional[str] = "msmarco"
    include_fulltext: Optional[bool] = False
    mode: Literal["exact", "ann"] = "exact"
    space: Literal["content", "title", "description", "keywords", "mix"] = "content"
    mix: Optional[Dict[str, float]] = None  # only used when space="mix"


def _fetch_seed_meta_doc(
    index_name: str,
    parent_id: str,
    fields: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Fetch the KO meta doc (chunk_index == -1) for the clicked KO.
    We only fetch embedding fields to keep it light.
    """
    body = {
        "size": 1,
        "_source": {"includes": ["parent_id", "chunk_index"] + fields},
        "query": {
            "bool": {
                "must": [
                    {"term": {"parent_id": parent_id}},
                    {"term": {"chunk_index": -1}},
                ]
            }
        }
    }

    try:
        resp = client.search(index=index_name, body=body)
    except TransportError as e:
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch error while fetching seed meta doc",
            "opensearch": detail,
        })

    hits = (resp.get("hits", {}) or {}).get("hits", []) or []
    if not hits:
        return None
    return hits[0].get("_source") or None


def _normalise_mix_weights(mix: Dict[str, float]) -> Dict[str, float]:
    # Keep only recognised keys and positive weights
    cleaned = {k: float(v) for k, v in (mix or {}).items() if k in EMBEDDING_FIELDS and float(v) > 0.0}
    s = sum(cleaned.values())
    if s <= 0.0:
        return {}
    return {k: v / s for k, v in cleaned.items()}


def _get_seed_vectors(
    seed_source: Dict[str, Any],
    space: str,
    mix: Optional[Dict[str, float]],
) -> Tuple[Dict[str, list], Dict[str, float]]:
    """
    Returns:
      vectors: { "content": [...], "title": [...] } for requested spaces
      weights: { "content": 0.7, "title": 0.3 } normalised (only for mix; else single 1.0)
    """
    if space != "mix":
        field = EMBEDDING_FIELDS.get(space)
        vec = seed_source.get(field) if field else None
        if not isinstance(vec, list) or not vec:
            return {}, {}
        return {space: vec}, {space: 1.0}

    weights = _normalise_mix_weights(mix or {})
    if not weights:
        return {}, {}

    vectors: Dict[str, list] = {}
    for sp in weights.keys():
        field = EMBEDDING_FIELDS[sp]
        vec = seed_source.get(field)
        if isinstance(vec, list) and vec:
            vectors[sp] = vec

    # If one of the requested vectors is missing, drop it + renormalise
    kept_weights = {sp: w for sp, w in weights.items() if sp in vectors}
    kept_weights = _normalise_mix_weights(kept_weights)

    return vectors, kept_weights



def _fetch_meta_docs_by_parent_ids(
    index_name: str,
    parent_ids: List[str],
    source_includes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch meta docs (chunk_index = -1) for given parent_ids.
    Returns: { parent_id: _source }
    """
    if not parent_ids:
        return {}

    body = {
        "size": len(parent_ids),
        "_source": {"includes": ["parent_id"] + source_includes},
        "query": {
            "bool": {
                "must": [
                    {"terms": {"parent_id": parent_ids}},
                    {"term": {"chunk_index": -1}},
                ]
            }
        }
    }

    try:
        resp = client.search(index=index_name, body=body)
    except TransportError as e:
        logger.warning("OpenSearch error while fetching meta docs for enrichment: %s", e)
        return {}

    hits = (resp.get("hits", {}) or {}).get("hits", []) or []

    out: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        src = h.get("_source", {}) or {}
        pid = src.get("parent_id")
        if pid:
            out[pid] = src
    return out


def _build_weighted_script(seed_vectors: Dict[str, list], seed_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Builds a script_score script that sums weighted cosine similarities.
    Missing vectors in docs return 0 contribution.
    """
    lines = ["double score = 0.0;"]
    params: Dict[str, Any] = {}

    for sp, vec in seed_vectors.items():
        w = float(seed_weights.get(sp, 0.0))
        if w <= 0.0:
            continue
        field = EMBEDDING_FIELDS[sp]

        # Give each space its own param names
        q_name = f"q_{sp}"
        f_name = f"f_{sp}"
        w_name = f"w_{sp}"

        params[q_name] = vec
        params[f_name] = field
        params[w_name] = w

        lines.append(
            f"if (params.{w_name} > 0 && doc[params.{f_name}].size() != 0) {{"
            f"  score += params.{w_name} * (cosineSimilarity(params.{q_name}, doc[params.{f_name}]) + 1.0);"
            f"}}"
        )

    lines.append("return score;")

    return {
        "source": "\n".join(lines),
        "params": params,
    }


def recommend_similar_knn(
    index_name: str,
    parent_id: str,
    k: int = 6,
    k_candidates: Optional[int] = None,
    include_fulltext: bool = False,
    mode: str = "exact",
    space: str = "content",
    mix: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:

    """
    Pure content-based recommender (no filters):
    - Seed: meta doc (chunk_index == -1) for parent_id
    - Retrieval: k-NN over best available embedding field (content > description > title...)
    - Output: top-k KOs collapsed by parent_id (like your search endpoints)
    """

    # Output size = how many unique parent KOs you want to return
    size = max(int(k), 1)

    use_ann = (mode == "ann") and (k_candidates is not None) and (int(k_candidates) > 0)
    ann_k = max(int(k_candidates), size * 30) if use_ann else None

    # ---- Fail fast if index doesn't exist ----
    try:
        exists = client.indices.exists(index=index_name, request_timeout=2)
        if not exists:
            return {
                "took": 0,
                "timed_out": False,
                "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                "grouped": {"total_parents": 0, "parents": []},
                "_meta": {"error": f"Index not found: {index_name}", "index_missing": True},
            }
    except (ConnectionTimeout, ConnectionError) as e:
        return {
            "took": 0,
            "timed_out": True,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"OpenSearch unreachable: {type(e).__name__}", "unreachable": True},
        }

    needed_spaces = [space] if space != "mix" else list((mix or {}).keys())
    needed_spaces = [sp for sp in needed_spaces if sp in EMBEDDING_FIELDS]
    seed_fields = [EMBEDDING_FIELDS[sp] for sp in needed_spaces]

    seed = _fetch_seed_meta_doc(index_name=index_name, parent_id=parent_id, fields=seed_fields)
    if not seed:
        return {
            "took": 0,
            "timed_out": False,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"Seed meta doc not found for parent_id={parent_id}", "seed_missing": True},
        }

    seed_vectors, seed_weights = _get_seed_vectors(seed_source=seed, space=space, mix=mix)
    if not seed_vectors:
        return {
            "took": 0,
            "timed_out": False,
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
            "grouped": {"total_parents": 0, "parents": []},
            "_meta": {"error": f"No seed vectors for space={space}", "no_seed_vector": True},
        }

    # Only return the fields you want
    source_includes = [
        "parent_id",
        "title",
        "subtitle",
        "description",
        "keywords",
        "category",
        "languages",
        "creators",
        "date_of_completion",
        "project_acronym",
        "project_id",
        "topics",
    ]

    # 2) Build query depending on mode/space
    collapse_block = {
        "field": "parent_id",
        "inner_hits": {
            "name": "best_chunks",
            "size": 1,
            "_source": ["chunk_index", "content_chunk"],
        },
    }

    if use_ann:
        # ANN can only retrieve using ONE embedding field.
        # If space=mix, choose the max-weight space for retrieval, then rerank precisely.
        if space != "mix":
            ann_space = space
        else:
            ann_space = max(seed_weights.keys(), key=lambda s: seed_weights.get(s, 0.0))

        ann_field = EMBEDDING_FIELDS[ann_space]
        ann_vector = seed_vectors[ann_space]

        # Step 1: ANN retrieval (pull ann_k chunk docs)
        ann_query = {
            "_source": {"includes": source_includes},
            "track_total_hits": False,
            "size": int(ann_k),
            "query": {
                "knn": {
                    ann_field: {
                        "vector": ann_vector,
                        "k": int(ann_k),
                        "filter": {
                            "bool": {
                                "must": [
                                    {"range": {"chunk_index": {"gte": 0}}},
                                    {"exists": {"field": ann_field}},
                                ],
                                "must_not": [
                                    {"term": {"parent_id": parent_id}},
                                ],
                            }
                        },
                    }
                }
            },
        }

        ann_resp = client.search(index=index_name, body=ann_query)
        ann_hits = (ann_resp.get("hits", {}) or {}).get("hits", []) or []

        if space != "mix":
            # For single-space ANN, you can collapse directly by doing a second query with ids + collapse.
            # (Because collapse + knn together is not always supported consistently depending on OpenSearch version/plugins.)
            candidate_ids = [h.get("_id") for h in ann_hits if h.get("_id")]
            if not candidate_ids:
                response = {"hits": {"hits": []}, "took": ann_resp.get("took", 0),
                            "timed_out": ann_resp.get("timed_out", False)}
            else:
                response = client.search(index=index_name, body={
                    "_source": {"includes": source_includes},
                    "track_total_hits": False,
                    "size": size,
                    "query": {"ids": {"values": candidate_ids}},
                    "collapse": collapse_block,
                })
        else:
            # Step 2: precise rerank using weighted script_score + collapse
            candidate_ids = [h.get("_id") for h in ann_hits if h.get("_id")]
            if not candidate_ids:
                response = {"hits": {"hits": []}, "took": ann_resp.get("took", 0),
                            "timed_out": ann_resp.get("timed_out", False)}
            else:
                script_obj = _build_weighted_script(seed_vectors, seed_weights)

                rerank_query = {
                    "_source": {"includes": source_includes},
                    "track_total_hits": False,
                    "size": size,
                    "query": {
                        "script_score": {
                            "query": {"ids": {"values": candidate_ids}},
                            "script": script_obj,
                        }
                    },
                    "collapse": collapse_block,
                }
                response = client.search(index=index_name, body=rerank_query)

    else:
        # Exact mode (single space or mix) using script_score directly
        script_obj = _build_weighted_script(seed_vectors, seed_weights)

        search_query = {
            "_source": {"includes": source_includes},
            "track_total_hits": False,
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"range": {"chunk_index": {"gte": 0}}},
                            ],
                            "must_not": [
                                {"term": {"parent_id": parent_id}},
                            ],
                        }
                    },
                    "script": script_obj,
                }
            },
            "collapse": collapse_block,
        }

        response = client.search(index=index_name, body=search_query)

    try:
        pass
    except TransportError as e:
        logger.exception("k-NN recommender search failed: %s", e)
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch k-NN recommender search failed",
            "opensearch": detail,
        })

    hits = response.get("hits", {}).get("hits", []) or []

    if include_fulltext:
        parent_ids_set = set()
        for h in hits:
            pid = (h.get("_source", {}) or {}).get("parent_id")
            if pid:
                parent_ids_set.add(pid)

        parent_ids = list(parent_ids_set)

        meta_map = _fetch_meta_docs_by_parent_ids(
            index_name=index_name,
            parent_ids=parent_ids,
            source_includes=["ko_content_flat"],
        )

        # Attach ko_content_flat onto each hit’s _source so formatter can emit it
        for h in hits:
            src = h.get("_source", {}) or {}
            pid = src.get("parent_id")
            if pid and pid in meta_map:
                src["ko_content_flat"] = meta_map[pid].get("ko_content_flat")
                h["_source"] = src

    grouped = group_hits_by_parent(hits, parents_size=size)

    response["grouped"] = grouped
    meta = cast(Dict[str, Any], response.setdefault("_meta", {}))
    meta.update({
        "seed_parent_id": parent_id,
        "mode": mode,
        "space": space,
        "mix": seed_weights if space == "mix" else None,  # normalised weights actually used
        "seed_embedding_fields": {sp: EMBEDDING_FIELDS[sp] for sp in seed_vectors.keys()},
        "k_candidates": int(k_candidates or 0),
        "ann_k_effective": int(ann_k or 0),
        "k_returned": int(size),
        "ann_enabled": bool(use_ann),
        "took_ms": int(response.get("took", 0) or 0),
        "timed_out": bool(response.get("timed_out", False)),
        "include_fulltext": bool(include_fulltext),
    })

    return _format_recommender_response(response)


def _format_recommender_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenSearch response into a clean:
      { "meta": {...}, "data": [...] }
    """
    hits = (response.get("hits", {}) or {}).get("hits", []) or []
    meta = response.get("_meta", {}) or {}

    data = []
    for h in hits:
        src = h.get("_source", {}) or {}

        # KO id is parent_id (same concept as your search endpoint _id)
        ko_id = src.get("parent_id")

        item = {
            "_id": ko_id,
            "_score": h.get("_score"),
            "title": src.get("title"),
            "subtitle": src.get("subtitle"),
            "description": src.get("description"),
            "category": src.get("category"),
            "languages": src.get("languages"),
            "creators": src.get("creators"),
            "date_of_completion": src.get("date_of_completion"),
            "keywords": src.get("keywords"),
            "project_acronym": src.get("project_acronym"),
            "project_id": src.get("project_id"),
            "topics": src.get("topics"),
        }

        # Only include full text when requested
        if meta.get("include_fulltext"):
            item["ko_content_flat"] = src.get("ko_content_flat")

        data.append(item)

    return {
        "meta": meta,
        "data": data,
    }

