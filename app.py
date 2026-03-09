# app.py

import logging
import os
import time

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from services.clickhouse_logger import make_default_clickhouse_logger
from services.neural_search_relevant import neural_search_relevant, RelevantSearchRequest
from services.neural_search_relevant_advanced import (
    neural_search_relevant as neural_search_relevant_advanced,
    RelevantSearchRequest as RelevantSearchRequestAdvanced,
    build_advanced_clause_debug_for_hits,
)
from services.neural_search_relevant_hybrid import neural_search_relevant_hybrid, RelevantSearchRequestHybrid
from services.neural_search_relevant_sparse import neural_search_relevant_sparse
from services.neural_search_relevant_new import (neural_search_relevant_new, split_query_into_fragments,
                                                 RelevantSearchRequestNew, score_chunk_for_fragments, )
from services.recommender_knn import recommend_similar_knn, RecommendKNNRequest
from services.search_endpoint_helpers import maybe_translate_query, resolve_auth_context, build_response_json, log_search_event
from services.summariser_hf import summarise_top5_hf
from services.utils import (BASIC_AUTH_PASS, BASIC_AUTH_USER, MODEL_CONFIG, MultiUserTimedAuthMiddleware,
                            fetch_chunks_for_parents, PAGE_SIZE, save_debug_dump, infer_query_intent)
from tools.debug_llm_summary import build_llm_summary_from_explain_top3, build_llm_summary_from_advanced_matches
from tools.llm_explain_client import explain_debug_non_technical


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Temporarily disable ClickHouse analytics logging at the app level.
# Endpoints still call log_search_event(), but it will no-op when ch_logger is None.
ch_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        pass


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

app = FastAPI(title="OpenSearch API", version="1.0", lifespan=lifespan)
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
HYBRID_SEARCH_PIPELINE = os.getenv("HYBRID_SEARCH_PIPELINE", "eufb-hybrid-v1")
HYBRID_ENABLE_PIPELINE = (os.getenv("HYBRID_ENABLE_PIPELINE", "true").strip().lower() in {"1", "true", "yes", "on"})

@app.post("/neural_search_relevant", tags=["Search"],
          summary="Semantic-first search with lexical boosting",
          response_description="Parent-grouped search results with pagination and optional debug metadata.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Performs semantic-first search using neural vector similarity across multiple document fields
          (title, subtitle, description, keywords, and content), with lexical boosting for exact term and
          phrase evidence. For code-like or acronym-style queries, the endpoint falls back to a stricter
          lexical-first mode. Results are grouped at the parent document level.
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

    original_query = request.search_term.strip()
    query = original_query

    t0 = time.monotonic()

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
    # Track translation status for analytics
    detected_lang = "en"
    translated = False
    
    if translation_allowed:
        from services.language_detect import detect_language
        try:
            detected_lang = detect_language(query).lower()
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
    
    translated_query = maybe_translate_query(query, translation_allowed=translation_allowed)
    if translated_query != query:
        translated = True
    query = translated_query

    use_semantic, _, _, _, _ = infer_query_intent(
        query,
        code_hint_env="QUERY_CODE_HINT_REGEX",
        acronym_min_len_env="QUERY_ACRONYM_MIN_LEN",
        acronym_max_len_env="QUERY_ACRONYM_MAX_LEN",
        acronym_min_caps_env="QUERY_ACRONYM_MIN_CAPS",
    )

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

        try:
            llm_input = build_llm_summary_from_explain_top3(response_json["_debug"]["explain_top3"])
            llm_out = explain_debug_non_technical(llm_input, timeout_s=30)
            if llm_out:
                response_json["_debug"]["llm_explanation_lines"] = llm_out.splitlines()
        except Exception:
            pass

    # Optional: show whether OpenSearch even received explain/profile flags
    response_json["_debug"] = response_json.get("_debug", {})
    response_json["_debug"]["request_flags"] = {
        "debug_explain": bool(getattr(request, "debug_explain", False)),
        "debug_profile": bool(getattr(request, "debug_profile", False)),
    }

    # Log to ClickHouse (non-blocking)
    await log_search_event(
        ch_logger=ch_logger,
        request_temp=request_temp,
        endpoint="/neural_search_relevant",
        original_query=original_query,
        query=query,
        detected_lang=detected_lang,
        translated=translated,
        translation_allowed=translation_allowed,
        user_id=user_id,
        model_key=model_key,
        index_name=index_name,
        use_semantic=use_semantic,
        filters=filters,
        response_json=response_json,
        request_k=(request.k if (getattr(request, "k", None) is not None) else None),
        t0=t0,
    )

    return response_json


@app.post("/neural_search_relevant_advanced", tags=["Search"],
          summary="Experimental advanced boolean-aware search",
          response_description="Parent-grouped advanced search results with pagination and optional debug metadata.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Experimental search endpoint for advanced query syntax. When `advanced=true`, it supports explicit
          uppercase boolean operators (`AND`, `OR`, `NOT`), parentheses, quoted phrases, field-scoped clauses,
          project scoping, and mode controls. Positive clauses use semantic and lexical evidence; negative clauses
          use lexical exclusion for predictable behavior. When `advanced=false`, this endpoint falls back to the
          same retrieval behavior as `/neural_search_relevant`.
          """)
async def neural_search_relevant_advanced_endpoint(request_temp: Request, request: RelevantSearchRequestAdvanced):
    try:
        raw_payload = await request_temp.json()
    except Exception:
        raw_payload = None

    if isinstance(raw_payload, dict) and raw_payload.get("access_token"):
        tok = str(raw_payload["access_token"])
        logger.info("Raw access_token preview=%s...%s (len=%d)", tok[:6], tok[-4:], len(tok))

    page_number = max(request.page, 1)

    original_query = request.search_term.strip()
    query = original_query

    t0 = time.monotonic()

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

    detected_lang = "en"
    translated = False

    if translation_allowed:
        from services.language_detect import detect_language
        try:
            detected_lang = detect_language(query).lower()
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

    translated_query = maybe_translate_query(query, translation_allowed=translation_allowed)
    if translated_query != query:
        translated = True
    query = translated_query

    use_semantic, _, _, _, _ = infer_query_intent(
        query,
        code_hint_env="QUERY_CODE_HINT_REGEX",
        acronym_min_len_env="QUERY_ACRONYM_MIN_LEN",
        acronym_max_len_env="QUERY_ACRONYM_MAX_LEN",
        acronym_min_caps_env="QUERY_ACRONYM_MIN_CAPS",
    )

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    advanced_enabled = bool(getattr(request, "advanced", True))

    if advanced_enabled:
        response = neural_search_relevant_advanced(
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
    else:
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

    logger.info("Advanced Search Query: '%s', Advanced: %s, Semantic: %s, Index: %s, Page: %d",
                query, advanced_enabled, use_semantic, index_name, page_number)

    if bool(getattr(request, "debug_explain", False)):
        raw_hits = (response or {}).get("hits", {}).get("hits", [])
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

        if advanced_enabled:
            parsed_query = ((response_json.get("_meta") or {}).get("advanced_search") or {}).get("parsed_query")
            clause_debug_by_parent = build_advanced_clause_debug_for_hits(parsed_query or {}, raw_hits)
            if clause_debug_by_parent:
                for item in response_json.get("data", []):
                    parent_id = item.get("_id")
                    if parent_id in clause_debug_by_parent:
                        item["_debug"] = item.get("_debug", {})
                        item["_debug"]["matched_clauses"] = clause_debug_by_parent[parent_id]

                try:
                    llm_input = build_llm_summary_from_advanced_matches(
                        query=query,
                        parsed_query=parsed_query or {},
                        results=response_json.get("data", []),
                    )
                    llm_out = explain_debug_non_technical(llm_input, timeout_s=30)
                    if llm_out:
                        response_json["_debug"]["llm_explanation_lines"] = llm_out.splitlines()
                except Exception:
                    pass

    response_json["_debug"] = response_json.get("_debug", {})
    response_json["_debug"]["request_flags"] = {
        "debug_explain": bool(getattr(request, "debug_explain", False)),
        "debug_profile": bool(getattr(request, "debug_profile", False)),
    }

    await log_search_event(
        ch_logger=ch_logger,
        request_temp=request_temp,
        endpoint="/neural_search_relevant_advanced",
        original_query=original_query,
        query=query,
        detected_lang=detected_lang,
        translated=translated,
        translation_allowed=translation_allowed,
        user_id=user_id,
        model_key=model_key,
        index_name=index_name,
        use_semantic=use_semantic,
        filters=filters,
        response_json=response_json,
        request_k=(request.k if (getattr(request, "k", None) is not None) else None),
        t0=t0,
    )

    return response_json


@app.post("/neural_search_relevant_hybrid", tags=["Search"],
          summary="Hybrid search with BM25–neural score normalisation",
          response_description="Parent-grouped hybrid search results with pagination and optional debug metadata.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Executes an OpenSearch hybrid query that combines a BM25 branch and a neural branch, then optionally applies
          a configured search pipeline to normalize and merge the scores. Query-intent routing is still applied before
          the hybrid query is built. Results are grouped at the parent document level. This endpoint is experimental
          relative to `/neural_search_relevant` and is mainly useful when you want OpenSearch-side score fusion.
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
    original_query = request.search_term.strip()
    query = original_query

    access_token = request.access_token

    translation_allowed, include_summary, user_id = await resolve_auth_context(
        access_token=access_token,
        dev=bool(request.dev),
        include_summary_flag=bool(getattr(request, "include_summary", False)),
    )

    # ------------------------------------ DeepL Translation ------------------------------------#
    # Track translation status for analytics
    detected_lang = "en"
    translated = False
    
    if translation_allowed:
        from services.language_detect import detect_language
        try:
            detected_lang = detect_language(query).lower()
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
    
    translated_query = maybe_translate_query(query, translation_allowed=translation_allowed)
    if translated_query != query:
        translated = True
    query = translated_query

    # Query-intent routing (same as /neural_search_relevant, configurable via env)
    use_semantic_hybrid, _, _, _, _ = infer_query_intent(
        query,
        code_hint_env="HYBRID_QUERY_CODE_HINT_REGEX",
        code_hint_fallback_env="QUERY_CODE_HINT_REGEX",
        acronym_min_len_env="HYBRID_QUERY_ACRONYM_MIN_LEN",
        acronym_min_len_fallback_env="QUERY_ACRONYM_MIN_LEN",
        acronym_max_len_env="HYBRID_QUERY_ACRONYM_MAX_LEN",
        acronym_max_len_fallback_env="QUERY_ACRONYM_MAX_LEN",
        acronym_min_caps_env="HYBRID_QUERY_ACRONYM_MIN_CAPS",
        acronym_min_caps_fallback_env="QUERY_ACRONYM_MIN_CAPS",
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

    model_key = request.model.lower().strip() if request.model else "msmarco"
    model_config = MODEL_CONFIG.get(model_key, MODEL_CONFIG["msmarco"])
    index_name = model_config["index"]

    if request.dev:
        index_name += "_dev"

    model_id = model_config["model_id"]

    # ---- Timing & debug flags ----
    t0 = time.monotonic()
    
    debug_explain = bool(getattr(request, "debug_explain", False))
    debug_profile = bool(getattr(request, "debug_profile", False))
    debug_enabled = debug_explain or debug_profile

    response = neural_search_relevant_hybrid(
        index_name=index_name,
        query=query,
        filters=filters,
        page=page_number,
        model_id=model_id,
        use_semantic=use_semantic_hybrid,
        search_pipeline=(HYBRID_SEARCH_PIPELINE if HYBRID_ENABLE_PIPELINE else None),
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

    logger.info(
        "[HYBRID] Search Query: '%s', Semantic: %s, Index: %s, Page: %d, Pipeline enabled: %s, Requested pipeline: %s",
        query,
        use_semantic_hybrid,
        index_name,
        page_number,
        HYBRID_ENABLE_PIPELINE,
        HYBRID_SEARCH_PIPELINE,
    )

    if debug_enabled:
        debug_obj = response_json.setdefault("_debug", {})

        # Single canonical request_flags block (don’t overwrite later)
        debug_obj["request_flags"] = {
            "debug_explain": debug_explain,
            "debug_profile": debug_profile,
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
            try:
                llm_input = build_llm_summary_from_explain_top3(debug_obj["explain_top3"])
                llm_out = explain_debug_non_technical(llm_input, timeout_s=30)
                if llm_out:
                    debug_obj["llm_explanation_lines"] = llm_out.splitlines()
            except Exception:
                pass

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

    # Log to ClickHouse (non-blocking)
    await log_search_event(
        ch_logger=ch_logger,
        request_temp=request_temp,
        endpoint="/neural_search_relevant_hybrid",
        original_query=original_query,
        query=query,
        detected_lang=detected_lang,
        translated=translated,
        translation_allowed=translation_allowed,
        user_id=user_id,
        model_key=model_key,
        index_name=index_name,
        use_semantic=use_semantic_hybrid,
        filters=filters,
        response_json=response_json,
        request_k=(request.k if (getattr(request, "k", None) is not None) else None),
        t0=t0,
    )

    return response_json


@app.post("/neural_search_relevant_sparse", tags=["Search"],
          summary="Neural sparse search (separate from hybrid)",
          response_description="Parent-grouped sparse semantic search results with pagination.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Executes sparse neural retrieval against the `content_sparse` field using the query-time analyzer
          `bert-uncased`. This endpoint is separate from the hybrid pipeline and is intended for sparse semantic
          retrieval experiments. Results are grouped at the parent document level.
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
    original_query = request.search_term.strip()
    query = original_query

    access_token = request.access_token

    translation_allowed, include_summary, user_id = await resolve_auth_context(
        access_token=access_token,
        dev=bool(request.dev),
        include_summary_flag=bool(getattr(request, "include_summary", False)),
    )

    # ---------------- DeepL Translation ----------------
    # Track translation status for analytics
    detected_lang = "en"
    translated = False
    
    if translation_allowed:
        from services.language_detect import detect_language
        try:
            detected_lang = detect_language(query).lower()
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
    
    translated_query = maybe_translate_query(query, translation_allowed=translation_allowed)
    if translated_query != query:
        translated = True
    query = translated_query

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

    t0 = time.monotonic()

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

    # Log to ClickHouse (non-blocking)
    await log_search_event(
        ch_logger=ch_logger,
        request_temp=request_temp,
        endpoint="/neural_search_relevant_sparse",
        original_query=original_query,
        query=query,
        detected_lang=detected_lang,
        translated=translated,
        translation_allowed=translation_allowed,
        user_id=user_id,
        model_key=model_key,
        index_name=index_name,
        use_semantic=True,  # Sparse search is semantic
        filters=filters,
        response_json=response_json,
        request_k=(request.k if (getattr(request, "k", None) is not None) else None),
        t0=t0,
    )

    return response_json


@app.post("/recommend_similar_knn", tags=["Recommend"],
          summary="Semantic similar-item recommendations for a KO",
          response_description="Similar parent knowledge objects for the given seed KO.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Given a seed knowledge object (`parent_id`), returns semantically similar parent KOs using vector
          similarity from the seed meta document. Supports exact or ANN retrieval and can use content, title,
          description, keywords, or a weighted mix of embedding spaces. No request-time filters are applied.
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
          summary="Experimental fragment-aware neural search",
          response_description="Parent-grouped fragment-aware search results with pagination.",
          responses={
              401: {"description": "Authentication required or invalid credentials."},
              422: {"description": "Invalid request payload."},
              502: {"description": "Upstream OpenSearch request failed."},
          },
          description="""
          Experimental search endpoint that splits the query into fragments, retrieves candidate parent documents,
          and can re-score returned chunks against those fragments when full text is requested. It uses a simple
          short-query fallback heuristic and is separate from the current default `/neural_search_relevant`
          endpoint.
          """)
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
