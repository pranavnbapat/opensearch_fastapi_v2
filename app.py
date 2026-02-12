# app.py

import logging
import os
import re
# import time

# from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

# from services.clickhouse_logger import make_default_clickhouse_logger, build_search_event
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.neural_search_relevant_hybrid import neural_search_relevant_hybrid, RelevantSearchRequestHybrid
from services.neural_search_relevant_sparse import neural_search_relevant_sparse
from services.neural_search_relevant_new import (neural_search_relevant_new, split_query_into_fragments,
                                                 RelevantSearchRequestNew, score_chunk_for_fragments, )
from services.recommender_knn import recommend_similar_knn, RecommendKNNRequest
from services.search_endpoint_helpers import maybe_translate_query, resolve_auth_context, build_response_json
from services.summariser_hf import summarise_top5_hf
from services.utils import (BASIC_AUTH_PASS, BASIC_AUTH_USER, MODEL_CONFIG, MultiUserTimedAuthMiddleware,
                            fetch_chunks_for_parents, PAGE_SIZE, save_debug_dump)
from tools.debug_llm_summary import build_llm_summary_from_explain_top3
from tools.llm_explain_client import explain_debug_non_technical


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ch_logger = make_default_clickhouse_logger()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Start the background ClickHouse flush task
#     await ch_logger.start()
#     try:
#         yield
#     finally:
#         # Flush remaining events and close http client
#         await ch_logger.stop()


ALLOWED_USERS = {
    BASIC_AUTH_USER: {
        "password": BASIC_AUTH_PASS,
        "expires": None
    },
    "reviewer": {
        "password": "ItRWu8Y4jX1L",
        "expires": datetime(2025, 7, 31, 23, 59, 59)
    }
}

app = FastAPI(title="OpenSearch API", version="1.0")
app.add_middleware(MultiUserTimedAuthMiddleware, users=ALLOWED_USERS)

origins = [
    "http://127.0.0.1:8000",
    "https://api.opensearch.nexavion.com",
    "https://api.opensearchtest.nexavion.com",
    "https://backend-admin.dev.farmbook.ugent.be",
    "https://backend-admin.prd.farmbook.ugent.be",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SUMMARY_PROVIDER = (os.getenv("SUMMARY_PROVIDER") or "").lower()

@app.post("/neural_search_relevant", tags=["Search"],
          summary="Hybrid semantic search with explicit query control",
          description="""
          Performs a hybrid semantic search using neural vector similarity across multiple document fields 
          (title, subtitle, description, keywords, and content) combined with BM25 keyword matching.
          Semantic relevance is evaluated explicitly in the query layer using field-aware boosting and 
          disjunction-max scoring, while BM25 acts as a safety net for exact term and phrase matching. 
          Results are grouped at the parent document level.
          """)
async def neural_search_relevant_endpoint(request_temp: Request, request: RelevantSearchRequest):
    # --- DEBUG: raw inbound request body (includes access_token if sent) ---
    try:
        raw_payload = await request_temp.json()
    except Exception:
        raw_payload = None

    if isinstance(raw_payload, dict) and raw_payload.get("access_token"):
        tok = str(raw_payload["access_token"])
        logger.info("Raw access_token preview=%s...%s (len=%d)", tok[:6], tok[-4:], len(tok))

    page_number = max(request.page, 1)

    query = request.search_term.strip()

    # t0 = time.monotonic()

    access_token = request.access_token

    translation_allowed, include_summary, user_id = await resolve_auth_context(
        access_token=access_token,
        dev=bool(request.dev),
        include_summary_flag=bool(getattr(request, "include_summary", False)),
    )

    filters = {
        "topics": request.topics,
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "project_acronym": request.project_acronym,
        "locations": request.locations,
        "sort_by": getattr(request, "sort_by", None)
    }

    #------------------------------------ DeepL Translation ------------------------------------#
    query = maybe_translate_query(query, translation_allowed=translation_allowed)

    # Smart fallback to BM25 if query is short
    q = query.strip()
    q_tokens = q.split()

    looks_like_code_or_id = bool(re.search(r"\d|[_:/]|cve-|doi|isbn", q.lower()))

    # Acronym-like: single token, short, mostly letters, with multiple capitals (handles "PCFruit", "EIPAGRI", etc.)
    if len(q_tokens) == 1 and 2 <= len(q) <= 12 and re.fullmatch(r"[A-Za-z]+", q):
        cap_count = sum(1 for ch in q if ch.isupper())
        looks_like_acronym = cap_count >= 2
    else:
        looks_like_acronym = False

    looks_like_quoted = ('"' in q) or ("'" in q)
    very_short = len(q_tokens) <= 2

    # Default: hybrid-style semantic ON, but keep BM25 strong for exact lookups
    use_semantic = not (looks_like_code_or_id or looks_like_quoted or looks_like_acronym)

    # If it’s *very short* but not an acronym/id, semantic is still useful
    if very_short and not (looks_like_code_or_id or looks_like_acronym):
        use_semantic = True

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    # If k is requested: we want top-k parents (no pagination), so fetch that many
    # size_override = int(request.k) if (request.k is not None and request.k > 0) else None

    response = neural_search_relevant(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
        use_semantic=use_semantic,
        size_override=None,
        debug_profile=bool(getattr(request, "debug_profile", False)),
        debug_explain=bool(getattr(request, "debug_explain", False)),
    )

    include_fulltext = bool(getattr(request, "include_fulltext", False))

    response_json = await build_response_json(
        response=response,
        index_name=index_name,
        page_number=page_number,
        include_fulltext=include_fulltext,
        include_summary=include_summary,
        query=query,
        summary_provider=SUMMARY_PROVIDER,
        summarise_top5_hf_fn=summarise_top5_hf,
    )

    logger.info("Search Query: '%s', Semantic: %s, Index: %s, Page: %d",
                query, use_semantic, index_name, page_number)

    if bool(getattr(request, "debug_explain", False)):
        raw_hits = (response or {}).get("hits", {}).get("hits", [])

        # Keep it small; explain payloads are huge
        response_json["_debug"] = response_json.get("_debug", {})
        response_json["_debug"]["explain_top3"] = [
            {
                "_id": h.get("_id"),
                "_score": h.get("_score"),
                "parent_id": (h.get("_source") or {}).get("parent_id"),
                "_explanation": h.get("_explanation"),
            }
            for h in raw_hits[:3]
        ]

    # Optional: show whether OpenSearch even received explain/profile flags
    response_json["_debug"] = response_json.get("_debug", {})
    response_json["_debug"]["request_flags"] = {
        "debug_explain": bool(getattr(request, "debug_explain", False)),
        "debug_profile": bool(getattr(request, "debug_profile", False)),
    }

    return response_json

    # latency_ms = int((time.monotonic() - t0) * 1000)

    # IP + headers (Traefik/proxies)
    # xff = request_temp.headers.get("x-forwarded-for", "")
    # ip = (xff.split(",")[0].strip() if xff else (request_temp.client.host if request_temp.client else ""))
    #
    # user_agent = request_temp.headers.get("user-agent", "")
    # accept_language = request_temp.headers.get("accept-language", "")
    # referer = request_temp.headers.get("referer", "")

    # request_id = request_temp.headers.get("x-request-id") or request_temp.headers.get("x-correlation-id")

    # MAX_LOG_RESULTS = 10
    # result_orig_ids = [
    #     item.get("_id")
    #     for item in formatted_results
    #     if item.get("_id")
    # ][:MAX_LOG_RESULTS]

    # logger.info("Logging %d result_orig_ids; first=%s", len(result_orig_ids), result_orig_ids[0] if result_orig_ids else None)

    # event = build_search_event(
    #     endpoint="/neural_search_relevant",
    #     original_query=original_query,
    #     final_query=query,
    #     ip=ip,
    #     user_agent=user_agent,
    #     accept_language=accept_language,
    #     referer=referer,
    #     user_id=user_id,
    #     result_orig_ids=result_orig_ids,
    #     page=page_number,
    #     k=(request.k if (getattr(request, "k", None) is not None) else None),
    #     model_key=model_key,
    #     index_name=index_name,
    #     use_semantic=use_semantic,
    #     translation_allowed=translation_allowed,
    #     detected_lang=detected_lang,
    #     translated=translated,
    #     filters=filters,
    #     status_code=200,
    #     latency_ms=latency_ms,
    #     results_count=len(formatted_results),
    #     total_records=response_json.get("pagination", {}).get("total_records"),
    #     request_id=request_id,
    # )

    # Never break search if analytics logging fails
    # try:
    #     ch_logger.log_event(event)
    # except Exception as e:
    #     logger.warning("ClickHouse logging failed: %s", e)


@app.post("/neural_search_relevant_hybrid", tags=["Search"],
          summary="Hybrid search with BM25–neural score normalisation",
          description="""
          Executes a hybrid search combining BM25 keyword matching and neural vector similarity, then applies a search 
          pipeline to normalise and merge scores from both retrieval methods.
          This endpoint is optimised for stable, comparable ranking across heterogeneous queries by letting OpenSearch 
          handle score blending and normalisation at the pipeline level.
          """)
async def neural_search_relevant_hybrid_endpoint(request_temp: Request, request: RelevantSearchRequestHybrid):
    # --- DEBUG: raw inbound request body (includes access_token if sent) ---
    try:
        raw_payload = await request_temp.json()
    except Exception:
        raw_payload = None

    if isinstance(raw_payload, dict) and raw_payload.get("access_token"):
        tok = str(raw_payload["access_token"])
        logger.info("Raw access_token preview=%s...%s (len=%d)", tok[:6], tok[-4:], len(tok))

    page_number = max(request.page, 1)
    query = request.search_term.strip()

    access_token = request.access_token

    translation_allowed, include_summary, user_id = await resolve_auth_context(
        access_token=access_token,
        dev=bool(request.dev),
        include_summary_flag=bool(getattr(request, "include_summary", False)),
    )

    # ------------------------------------ DeepL Translation ------------------------------------#
    query = maybe_translate_query(query, translation_allowed=translation_allowed)

    filters = {
        "topics": request.topics,
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "project_acronym": request.project_acronym,
        "locations": request.locations,
        "sort_by": getattr(request, "sort_by", None)
    }

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    # ---- Compute debug flags ----
    debug_explain = bool(getattr(request, "debug_explain", False))
    debug_profile = bool(getattr(request, "debug_profile", False))
    debug_analyze = bool(getattr(request, "debug_analyze", False))
    debug_field = getattr(request, "debug_field", None)
    debug_enabled = debug_explain or debug_profile

    response = neural_search_relevant_hybrid(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
        search_pipeline="eufb-hybrid-v1",
        size_override=None,
        debug_profile=debug_profile,
        debug_explain=debug_explain,
    )

    include_fulltext = bool(getattr(request, "include_fulltext", False))

    response_json = await build_response_json(
        response=response,
        index_name=index_name,
        page_number=page_number,
        include_fulltext=include_fulltext,
        include_summary=include_summary,
        query=query,
        summary_provider=SUMMARY_PROVIDER,
        summarise_top5_hf_fn=summarise_top5_hf,
    )

    logger.info("[HYBRID] Search Query: '%s', Index: %s, Page: %d, Pipeline: %s",
                query, index_name, page_number, "eufb-hybrid-v1")

    if debug_enabled:
        debug_obj = response_json.setdefault("_debug", {})

        # Single canonical request_flags block (don’t overwrite later)
        debug_obj["request_flags"] = {
            "debug_explain": debug_explain,
            "debug_profile": debug_profile,
            "debug_analyze": debug_analyze,
            "debug_field": debug_field,
        }

        # Store OpenSearch request
        debug_obj["opensearch_request"] = (response.get("_debug_bundle", {}) or {}).get("opensearch_request")

        # Store OpenSearch debug bits
        raw_hits = (response.get("hits", {}) or {}).get("hits", [])
        debug_obj["opensearch_debug"] = {
            "hits": [
                {
                    "_id": h.get("_id"),
                    "_score": h.get("_score"),
                    "_explanation": h.get("_explanation"),
                    "_source_parent_id": (h.get("_source") or {}).get("parent_id"),
                }
                for h in raw_hits
                if h.get("_explanation") is not None
            ],
            "profile": response.get("profile"),
        }

        # Optional convenience view: explain_top3 (small, human-scannable)
        if debug_explain:
            debug_obj["explain_top3"] = [
                {
                    "_id": h.get("_id"),
                    "_score": h.get("_score"),
                    "parent_id": (h.get("_source") or {}).get("parent_id"),
                    "_explanation": h.get("_explanation"),
                }
                for h in raw_hits[:3]
                if h.get("_explanation") is not None
            ]

            # Optional: ask LLM to explain the debug in non-technical language
            if bool(getattr(request, "debug_llm_explain", False)):
                try:
                    llm_input = build_llm_summary_from_explain_top3(debug_obj["explain_top3"])
                    llm_out = explain_debug_non_technical(llm_input, timeout_s=30)

                    if llm_out:
                        # Keep raw text for copy/paste
                        # debug_obj["llm_explanation"] = llm_out

                        # Add a JSON-friendly version (no ugly \n when inspected)
                        debug_obj["llm_explanation_lines"] = llm_out.splitlines()

                        # Optional: a single-line version (handy for logs/headers)
                        # debug_obj["llm_explanation_one_line"] = " ".join(llm_out.split())

                    else:
                        debug_obj["llm_explanation_error"] = "LLM not configured (set LLM_URL and LLM_MODEL)."
                except Exception as e:
                    # Don't break search if LLM fails
                    debug_obj["llm_explanation_error"] = str(e)

        # Optional: persist debug to disk
        if bool(getattr(request, "debug_save", False)):
            # tools/ folder sits alongside app.py (same directory level)
            tools_dir = Path(__file__).resolve().parent / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)  # create if missing

            debug_path = save_debug_dump(
                debug_obj=debug_obj,
                base_dir=str(tools_dir),
                index_name=index_name,
                user_id=str(user_id) if user_id is not None else None,
                query=query,
            )
            debug_obj["saved_to"] = debug_path

    return response_json


@app.post("/neural_search_relevant_sparse", tags=["Search"],
          summary="Neural sparse search (separate from hybrid)",
          description="""
          Executes a neural sparse search against the `content_sparse` rank_features field.
          Uses doc-only sparse mode: query-time uses the compatible analyzer (bert-uncased).
          Results are grouped at the parent document level.
          """)
async def neural_search_relevant_sparse_endpoint(request_temp: Request, request: RelevantSearchRequest):
    # --- DEBUG: raw inbound request body (includes access_token if sent) ---
    try:
        raw_payload = await request_temp.json()
    except Exception:
        raw_payload = None

    if isinstance(raw_payload, dict) and raw_payload.get("access_token"):
        tok = str(raw_payload["access_token"])
        logger.info("Raw access_token preview=%s...%s (len=%d)", tok[:6], tok[-4:], len(tok))

    page_number = max(request.page, 1)
    query = request.search_term.strip()

    access_token = request.access_token

    translation_allowed, include_summary, user_id = await resolve_auth_context(
        access_token=access_token,
        dev=bool(request.dev),
        include_summary_flag=bool(getattr(request, "include_summary", False)),
    )

    # ---------------- DeepL Translation ----------------
    query = maybe_translate_query(query, translation_allowed=translation_allowed)

    filters = {
        "topics": request.topics,
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "project_acronym": request.project_acronym,
        "locations": request.locations,
        "sort_by": getattr(request, "sort_by", None)
    }

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    response = neural_search_relevant_sparse(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        analyzer="bert-uncased",
        size_override=None,
    )

    include_fulltext = bool(getattr(request, "include_fulltext", False))

    response_json = await build_response_json(
        response=response,
        index_name=index_name,
        page_number=page_number,
        include_fulltext=include_fulltext,
        include_summary=include_summary,
        query=query,
        summary_provider=SUMMARY_PROVIDER,
        summarise_top5_hf_fn=summarise_top5_hf,
    )

    logger.info("[SPARSE] Search Query: '%s', Index: %s, Page: %d, Analyzer: %s",
                query, index_name, page_number, "bert-uncased")

    return response_json


@app.post("/recommend_similar_knn", tags=["Recommend"],
          summary="Semantic k-NN recommendations for a clicked KO",
          description="""
          Given a clicked knowledge object (parent_id), returns top-k similar KOs using k-NN vector similarity.
          No filters are applied: this is pure nearest-neighbour recommendation.
          """)
async def recommend_similar_knn_endpoint(request: RecommendKNNRequest):
    parent_id = request.parent_id.strip()

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    k = max(int(request.k or 6), 1)

    # If 0/None => auto inside service
    raw_kc = request.k_candidates
    k_candidates = None if (raw_kc is None or int(raw_kc) <= 0) else int(raw_kc)

    response = recommend_similar_knn(
        index_name=index_name,
        parent_id=parent_id,
        k=k,
        k_candidates=k_candidates,
        include_fulltext=bool(request.include_fulltext),
        mode=request.mode,
        space=request.space,
        mix=request.mix,
    )

    logger.info("[RECOMMENDER KNN] parent_id=%s index=%s k=%d k_candidates=%s",
                parent_id, index_name, k, str(k_candidates) if k_candidates is not None else "auto")

    return response



@app.post("/neural_search_relevant_new", tags=["Search"],
          summary="Context-aware neural search with smart fallback",
          description="""Performs semantic relevance-based search using one of the supported models and retrieves 
          contextually matched documents. Supported semantic models:
          - `msmarco` (default)
          - `mpnetv2`
          - `minilml12v2`
          You can optionally pass `model` (default is `msmarco`) and `k` to get only the top-k ranked results 
          (no pagination).""")
async def neural_search_relevant_endpoint_new(request_temp: Request, request: RelevantSearchRequestNew):
    page_number = max(request.page, 1)

    query = request.search_term.strip()

    query_fragments = split_query_into_fragments(query)

    filters = {
        "topics": request.topics,
        "themes": request.themes,
        "languages": request.languages,
        "category": request.category,
        "project_type": request.project_type,
        "project_acronym": request.project_acronym,
        "locations": request.locations,
        "sort_by": getattr(request, "sort_by", None)
    }

    # Smart fallback to BM25 if query is short
    if len(query.split()) <= 5:
        logger.info("Short query detected, switching to BM25")
        use_semantic = False
    else:
        use_semantic = True

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    response = neural_search_relevant_new(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
        use_semantic=use_semantic
    )

    grouped = response.get("grouped", {})
    parents = grouped.get("parents", [])
    total_parents = grouped.get("total_parents", len(parents))

    # Ask for full text? (optional; defaults to False)
    include_fulltext = bool(getattr(request, "include_fulltext", False))

    # Only fetch chunks if caller explicitly wants full text AND we have parents
    parent_ids = [p["parent_id"] for p in parents]
    chunks_map = fetch_chunks_for_parents(index_name, parent_ids) if (include_fulltext and parent_ids) else {}

    # Parent-level “formatted” result objects
    formatted_results = []
    for p in parents:
        pid = p.get("parent_id")

        # --- NEW: rerank chunks for this parent based on query fragments ---
        if include_fulltext:
            raw_chunks = chunks_map.get(pid, [])  # whatever fetch_chunks_for_parents returns
            ko_chunks = []

            if raw_chunks:
                scored_chunks = []
                for ch in raw_chunks:
                    chunk_text = ch.get("content") or ch.get("content_chunk") or ""
                    stats = score_chunk_for_fragments(chunk_text, query_fragments)

                    # Simple weighted score; feel free to tune these weights
                    final_chunk_score = (
                            0.5 * stats["coverage"] +
                            0.3 * stats["avg_score"] +
                            0.2 * stats["max_score"]
                    )

                    final_chunk_score_pct = final_chunk_score * 100.0

                    scored_chunks.append({
                        "text": chunk_text,
                        "score": final_chunk_score,
                        "score_pct": final_chunk_score_pct,
                        "chunk_index": ch.get("chunk_index"),
                        "stats": stats,
                    })

                # sort by our custom score, highest first
                scored_chunks.sort(key=lambda x: x["score"], reverse=True)

                # we only expose the text list for now, like before
                ko_chunks = [s["text"] for s in scored_chunks]

                ko_chunks_scored = [
                    {
                        "text": s["text"],
                        "score": s["score"],
                        "score_pct": s["score"] * 100,
                        "chunk_index": s.get("chunk_index"),
                        "coverage": s["stats"]["coverage"],
                        "avg_score": s["stats"]["avg_score"],
                        "max_score": s["stats"]["max_score"],
                        "phrase_hits": s["stats"].get("phrase_hits"),
                    }
                    for s in scored_chunks
                ]
            else:
                ko_chunks = None
                ko_chunks_scored = None
        else:
            ko_chunks = None
            ko_chunks_scored = None
        # --- END NEW CHUNK RERANKING BLOCK ---

        # ko_chunks = [c["content"] for c in chunks_map.get(pid, [])] if include_fulltext else None

        doc_date = p.get("date_of_completion")
        if isinstance(doc_date, str) and len(doc_date) >= 10:
            try:
                y, m, d = doc_date[:10].split("-")
                date_created = f"{d}-{m}-{y}"
            except Exception:
                date_created = doc_date
        else:
            date_created = None

        item = {
            "_id": pid,
            "_score": p.get("max_score"),
            "title": p.get("title"),
            "subtitle": p.get("subtitle") or "",
            "description": p.get("description"),
            "projectAcronym": p.get("project_acronym"),
            "projectName": p.get("project_name"),
            "project_type": p.get("project_type"),
            "project_id": p.get("project_id"),
            "topics": p.get("topics") or [],
            "themes": p.get("themes") or [],
            "keywords": p.get("keywords") or [],
            "languages": p.get("languages") or [],
            "locations": p.get("locations") or [],
            "category": p.get("category"),
            "subcategories": p.get("subcategories") or [],
            "creators": p.get("creators") or [],
            "dateCreated": date_created,
            "@id": p.get("@id"),
            "_orig_id": p.get("_orig_id"),
            "_tags": p.get("keywords") or []
        }

        # Attach full text only if requested
        if include_fulltext:
            item["ko_content_flat"] = ko_chunks or []
            item["ko_content_scored"] = ko_chunks_scored or []

        formatted_results.append(item)

    # --- Normalise parent scores 0–100 (per response) ---
    scores = [item["_score"] for item in formatted_results if item.get("_score") is not None]

    if scores:
        s_min = min(scores)
        s_max = max(scores)
        span = s_max - s_min or 1.0  # avoid division by zero

        for item in formatted_results:
            raw = item.get("_score")
            if raw is None:
                item["score_norm_0_100"] = None
            else:
                item["score_norm_0_100"] = 100.0 * (raw - s_min) / span

    # k override still applies (now to parent results)
    if request.k is not None and request.k > 0:
        formatted_results = formatted_results[:request.k]
        pagination = {
            "total_records": len(formatted_results),
            "current_page": 1,
            "total_pages": 1,
            "next_page": None,
            "prev_page": None
        }
    else:
        total_pages = (total_parents + PAGE_SIZE - 1) // PAGE_SIZE
        pagination = {
            "total_records": total_parents,
            "current_page": page_number,
            "total_pages": total_pages,
            "next_page": page_number + 1 if page_number < total_pages else None,
            "prev_page": page_number - 1 if page_number > 1 else None
        }

    page_counts = {}
    for item in formatted_results:
        pid = item.get("project_id")
        if pid:
            page_counts[pid] = page_counts.get(pid, 0) + 1

    related_projects_from_this_page = [
        {"project_id": k, "count": v}
        for k, v in sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:3]  # drop [:3] to return all
    ]

    aggs = response.get("aggregations", {})
    buckets = aggs.get("top_projects", {}).get("buckets", [])
    related_projects_all = [
        {
            "project_id": b.get("key"),
            "count": b.get("unique_parents", {}).get("value", 0)
        }
        for b in buckets
    ]

    response_json = {
        "data": formatted_results,
        "related_projects_from_this_page": related_projects_from_this_page,
        "related_projects_from_entire_resultset": related_projects_all,
        "pagination": pagination
    }

    logger.info(f"Search Query: '{query}', Semantic: {use_semantic}, Index: {index_name}, Page: {page_number}")

    return response_json
