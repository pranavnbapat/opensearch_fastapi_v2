# services/search_endpoint_helpers.py

import logging
import time

from typing import Any, Dict, List, Optional

from services.clickhouse_logger import build_search_event
from services.language_detect import detect_language, translate_text_with_backoff, DEEPL_SUPPORTED_LANGUAGES
from services.utils import is_translation_allowed, jwt_claim, PAGE_SIZE, fetch_chunks_for_parents, fetch_meta_fields_for_parents


logger = logging.getLogger(__name__)


def maybe_translate_query(query: str, translation_allowed: bool) -> str:
    """
    Translates query to EN if allowed and detected language is supported.
    Returns possibly modified query.
    """
    if not translation_allowed:
        logger.info(
            "Translation not allowed; skipping translation (missing token, invalid token, wrong dev/prd target, or token validation failure)."
        )
        return query

    try:
        detected_lang = detect_language(query).lower()
    except Exception as e:
        logger.error("Failed to detect language for query '%s': %s", query, e)
        detected_lang = "en"

    if detected_lang != "en" and detected_lang.upper() in DEEPL_SUPPORTED_LANGUAGES:
        try:
            translated = translate_text_with_backoff(query, target_language="EN")
            logger.info("Translated query to English (detected: %s): %s", detected_lang, translated)
            return translated
        except Exception as e:
            logger.error("Failed to translate non-English query: %s", e)
            return query

    logger.info("Skipping translation: detected_lang=%s, supported=%s",
                detected_lang, detected_lang.upper() in DEEPL_SUPPORTED_LANGUAGES)
    return query


async def resolve_auth_context(access_token: Optional[str], dev: bool, include_summary_flag: bool):
    """
    Returns: (translation_allowed, include_summary, user_id)
    """
    user_id = None

    if access_token:
        uid = jwt_claim(access_token, "user_id")
        user_id = int(uid) if uid and str(uid).isdigit() else None
        translation_allowed = await is_translation_allowed(access_token, bool(dev))
    else:
        translation_allowed = False

    include_summary = bool(include_summary_flag) and bool(access_token) and bool(translation_allowed) and (user_id is not None)

    logger.info(
        "Summary auth context: include_summary_flag=%s access_token_present=%s translation_allowed=%s user_id=%s include_summary=%s dev=%s",
        bool(include_summary_flag),
        bool(access_token),
        bool(translation_allowed),
        user_id,
        bool(include_summary),
        bool(dev),
    )

    return translation_allowed, include_summary, user_id


def _format_date_created(doc_date: Any) -> Optional[str]:
    if isinstance(doc_date, str) and len(doc_date) >= 10:
        try:
            y, m, d = doc_date[:10].split("-")
            return f"{d}-{m}-{y}"
        except Exception:
            return doc_date
    return None


def format_parent_result_item(
    p: Dict[str, Any],
    *,
    fulltext_pages: Optional[list[str]] = None,
    fulltext_chunks: Optional[list[Dict[str, Any]]] = None,
    include_fulltext: bool = False,
) -> Dict[str, Any]:
    item = {
        "_id": p.get("parent_id"),
        "_score": p.get("max_score"),
        "title": p.get("title"),
        "title_original": p.get("title_original"),
        "subtitle": p.get("subtitle") or "",
        "subtitle_original": p.get("subtitle_original"),
        "description": p.get("description"),
        "description_original": p.get("description_original"),
        "keywords": p.get("keywords") or [],
        "keywords_original": p.get("keywords_original") or [],
        "projectAcronym": p.get("project_acronym"),
        "projectName": p.get("project_name"),
        "projectDisplayName": p.get("project_display_name"),
        "project_type": p.get("project_type"),
        "project_id": p.get("project_id"),
        "projectUrl": p.get("project_url"),
        "topics": p.get("topics") or [],
        "themes": p.get("themes") or [],
        "languages": p.get("languages") or [],
        "locations": p.get("locations") or [],
        "category": p.get("category"),
        "subcategories": p.get("subcategories") or [],
        "creators": p.get("creators") or [],
        "intended_purposes": p.get("intended_purposes") or [],
        "dateCreated": _format_date_created(p.get("date_of_completion")),
        "date_of_completion": p.get("date_of_completion"),
        "@id": p.get("@id"),
        "_orig_id": p.get("_orig_id"),
        "ko_id": p.get("ko_id"),
        "ko_created_at": p.get("ko_created_at"),
        "ko_updated_at": p.get("ko_updated_at"),
        "proj_created_at": p.get("proj_created_at"),
        "proj_updated_at": p.get("proj_updated_at"),
        "_tags": p.get("keywords") or [],
    }

    if include_fulltext:
        item["ko_content_flat"] = fulltext_pages or []
        item["fulltext_chunks"] = fulltext_chunks or []

    if p.get("ko_content_flat_summarised") is not None:
        item["ko_content_flat_summarised"] = p.get("ko_content_flat_summarised")

    return item


async def build_response_json(
    response: Dict[str, Any],
    index_name: str,
    page_number: int,
    include_fulltext: bool,
    include_summary: bool,
    query: str,
    summary_provider: str,
    summarise_top5_hf_fn,  # pass the function in to avoid circular imports
    summarise_topk_llm_fn=None,
) -> Dict[str, Any]:
    grouped = response.get("grouped", {}) or {}
    parents = grouped.get("parents", []) or []

    total_parents = grouped.get("total_parents")
    if total_parents is None:
        total_parents = len(parents)

    parent_ids = [p.get("parent_id") for p in parents if p.get("parent_id")]
    chunks_map = fetch_chunks_for_parents(index_name, parent_ids) if (include_fulltext and parent_ids) else {}
    meta_map = fetch_meta_fields_for_parents(
        index_name,
        parent_ids,
        ["ko_content_flat_summarised"],
    ) if parent_ids else {}

    formatted_results = []
    for p in parents:
        pid = p.get("parent_id")
        raw_chunks = chunks_map.get(pid, []) if include_fulltext else []
        fulltext_pages = [c.get("content", "") for c in raw_chunks if c.get("content")] if include_fulltext else None
        meta_fields = meta_map.get(pid) or {}
        if meta_fields.get("ko_content_flat_summarised") is not None:
            p = dict(p)
            p["ko_content_flat_summarised"] = meta_fields.get("ko_content_flat_summarised")

        item = format_parent_result_item(
            p,
            fulltext_pages=fulltext_pages,
            fulltext_chunks=raw_chunks if include_fulltext else None,
            include_fulltext=include_fulltext,
        )
        formatted_results.append(item)

    total_pages = (int(total_parents) + PAGE_SIZE - 1) // PAGE_SIZE
    pagination = {
        "total_records": int(total_parents),
        "current_page": page_number,
        "total_pages": total_pages,
        "next_page": page_number + 1 if page_number < total_pages else None,
        "prev_page": page_number - 1 if page_number > 1 else None,
    }

    page_counts: Dict[str, int] = {}
    for item in formatted_results:
        pid = item.get("project_id")
        if pid:
            page_counts[pid] = page_counts.get(pid, 0) + 1

    related_projects_from_this_page = [
        {"project_id": k, "count": v}
        for k, v in sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    ]

    aggs = response.get("aggregations", {}) or {}
    buckets = (aggs.get("top_projects", {}) or {}).get("buckets", []) or []
    related_projects_all = [
        {"project_id": b.get("key"), "count": (b.get("unique_parents", {}) or {}).get("value", 0)}
        for b in buckets
    ]

    summary = None
    if include_summary and summary_provider == "hf":
        # Caller ensures auth eligibility; here we just compute.
        logger.info(
            "Summary generation starting: provider=hf query=%r results=%d index=%s page=%d",
            query,
            len(formatted_results),
            index_name,
            page_number,
        )
        summary = summarise_top5_hf_fn(query=query, hits=formatted_results)
    elif include_summary and summary_provider == "llm" and summarise_topk_llm_fn is not None:
        logger.info(
            "Summary generation starting: provider=llm query=%r results=%d index=%s page=%d",
            query,
            len(formatted_results),
            index_name,
            page_number,
        )
        try:
            summary = await summarise_topk_llm_fn(query=query, hits=formatted_results)
        except Exception:
            logger.exception("Summary generation crashed: provider=llm query=%r", query)
            summary = None
    else:
        logger.info(
            "Summary generation skipped: include_summary=%s provider=%s has_llm_fn=%s query=%r results=%d",
            bool(include_summary),
            summary_provider,
            summarise_topk_llm_fn is not None,
            query,
            len(formatted_results),
        )

    logger.info(
        "Summary generation finished: provider=%s query=%r summary_present=%s summary_preview=%r",
        summary_provider,
        query,
        bool(summary),
        (summary[:160] if isinstance(summary, str) else summary),
    )

    response_json = {
        "summary": summary,
        "data": formatted_results,
        "related_projects_from_this_page": related_projects_from_this_page,
        "related_projects_from_entire_resultset": related_projects_all,
        "pagination": pagination,
    }

    # Preserve service-level metadata so evaluation and debugging can see
    # whether a hybrid search pipeline was requested, used, or bypassed.
    meta = response.get("_meta")
    if isinstance(meta, dict) and meta:
        response_json["_meta"] = meta

    return response_json


def _compact_project_display(src: Dict[str, Any]) -> Optional[str]:
    project_name = src.get("project_name") or src.get("projectName") or src.get("project_display_name")
    acronym = src.get("project_acronym") or src.get("projectAcronym")
    project_type = src.get("project_type")

    cap = (project_name or acronym or "").strip()
    if not cap:
        return None
    if project_type:
        return f"{cap} ({project_type})"
    return cap


def _build_llm_context(src: Dict[str, Any]) -> str:
    parts: List[str] = []

    title = (src.get("title") or "").strip()
    subtitle = (src.get("subtitle") or "").strip()
    description = (src.get("description") or "").strip()
    content = (src.get("content_chunk") or src.get("content") or "").strip()
    summarised_pages = src.get("ko_content_flat_summarised") or []
    summarised_text = ""
    if isinstance(summarised_pages, list):
        summarised_text = " ".join(str(x) for x in summarised_pages if str(x).strip()).strip()
    elif isinstance(summarised_pages, str):
        summarised_text = summarised_pages.strip()

    # Prefer chunk evidence, but fall back to summarised document text if the chunk
    # is too short to ground an answer usefully.
    evidence_text = content
    if len(evidence_text) < 140 and summarised_text:
        evidence_text = summarised_text[:1200]
    project_display = _compact_project_display(src)
    keywords = src.get("keywords") or []
    topics = src.get("topics") or []

    if title:
        parts.append(f"Title: {title}")
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")
    if project_display:
        parts.append(f"Project: {project_display}")
    if description:
        parts.append(f"Description: {description[:500]}")
    if keywords:
        parts.append("Keywords: " + ", ".join(map(str, keywords[:8])))
    if topics:
        parts.append("Topics: " + ", ".join(map(str, topics[:6])))
    if evidence_text:
        parts.append(f"Evidence: {evidence_text[:1200]}")

    return "\n".join(parts).strip()


def _format_llm_result_item_from_chunk(
    parent_hit: Dict[str, Any],
    chunk_hit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    parent_src = (parent_hit or {}).get("_source") or {}
    chunk_src = (chunk_hit or {}).get("_source") or {}

    parent_id = parent_src.get("parent_id") or parent_src.get("_orig_id")
    chunk_text = (chunk_src.get("content_chunk") or parent_src.get("content_chunk") or "").strip()
    chunk_index = chunk_src.get("chunk_index", parent_src.get("chunk_index"))

    if not parent_id or not chunk_text:
        return None

    merged_src = dict(parent_src)
    merged_src["content_chunk"] = chunk_text
    merged_src["chunk_index"] = chunk_index
    project_display = _compact_project_display(parent_src)

    return {
        "_id": chunk_hit.get("_id") or parent_hit.get("_id"),
        "_score": chunk_hit.get("_score", parent_hit.get("_score")),
        "parent_id": parent_id,
        "chunk_index": chunk_index,
        "title": parent_src.get("title"),
        "subtitle": parent_src.get("subtitle") or "",
        "description": parent_src.get("description"),
        "keywords": parent_src.get("keywords") or [],
        "topics": parent_src.get("topics") or [],
        "themes": parent_src.get("themes") or [],
        "languages": parent_src.get("languages") or [],
        "locations": parent_src.get("locations") or [],
        "category": parent_src.get("category"),
        "projectAcronym": parent_src.get("project_acronym"),
        "projectName": parent_src.get("project_name"),
        "projectDisplayName": project_display,
        "project_type": parent_src.get("project_type"),
        "project_id": parent_src.get("project_id"),
        "projectUrl": parent_src.get("project_url"),
        "@id": parent_src.get("@id"),
        "_orig_id": parent_src.get("_orig_id"),
        "date_of_completion": parent_src.get("date_of_completion"),
        "content_chunk": chunk_text,
        "ko_content_flat_summarised": parent_src.get("ko_content_flat_summarised"),
        "llm_context": _build_llm_context(merged_src),
        "citation_label": f"{parent_id}#c{chunk_index}",
    }


def expand_llm_result_items(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    inner_hits = ((((hit or {}).get("inner_hits") or {}).get("best_chunks") or {}).get("hits") or {}).get("hits") or []
    if inner_hits:
        items = []
        for chunk_hit in inner_hits:
            item = _format_llm_result_item_from_chunk(hit, chunk_hit)
            if item:
                items.append(item)
        return items

    fallback = _format_llm_result_item_from_chunk(hit, hit)
    return [fallback] if fallback else []


async def build_llm_retrieval_response(
    response: Dict[str, Any],
    index_name: str,
    top_k: int,
    max_chunks_per_parent: int,
) -> Dict[str, Any]:
    hits = ((response.get("hits") or {}).get("hits")) or []
    parent_ids = []
    for hit in hits:
        src = (hit.get("_source") or {})
        pid = src.get("parent_id") or src.get("_orig_id")
        if pid:
            parent_ids.append(pid)

    meta_map = fetch_meta_fields_for_parents(
        index_name,
        list(dict.fromkeys(parent_ids)),
        ["ko_content_flat_summarised"],
    ) if parent_ids else {}

    enriched_hits = []
    for hit in hits:
        src = (hit.get("_source") or {})
        pid = src.get("parent_id") or src.get("_orig_id")
        if pid and pid in meta_map:
            hit = dict(hit)
            merged_src = dict(src)
            merged_src.update(meta_map[pid])
            hit["_source"] = merged_src
        enriched_hits.append(hit)

    formatted_results = []
    seen_ids = set()
    per_parent_counts: Dict[str, int] = {}

    target_k = max(1, int(top_k))
    parent_limit = max(1, int(max_chunks_per_parent))

    for hit in enriched_hits:
        for item in expand_llm_result_items(hit):
            item_id = item.get("_id")
            parent_id = item.get("parent_id")
            chunk_index = item.get("chunk_index")
            dedupe_key = f"{item_id}:{chunk_index}"
            if dedupe_key in seen_ids or not parent_id:
                continue
            if per_parent_counts.get(parent_id, 0) >= parent_limit:
                continue

            seen_ids.add(dedupe_key)
            per_parent_counts[parent_id] = per_parent_counts.get(parent_id, 0) + 1
            formatted_results.append(item)

            if len(formatted_results) >= target_k:
                break
        if len(formatted_results) >= target_k:
            break

    response_json = {
        "data": formatted_results,
        "pagination": {
            "total_records": len(formatted_results),
            "current_page": 1,
            "total_pages": 1,
            "next_page": None,
            "prev_page": None,
        },
    }

    meta = response.get("_meta")
    response_json["_meta"] = dict(meta) if isinstance(meta, dict) else {}
    response_json["_meta"]["retrieval_mode"] = "llm"
    response_json["_meta"]["returned_chunks"] = len(formatted_results)
    response_json["_meta"]["max_chunks_per_parent"] = parent_limit

    return response_json


async def log_search_event(
    ch_logger,
    request_temp,
    endpoint: str,
    original_query: str,
    query: str,
    detected_lang: str,
    translated: bool,
    translation_allowed: bool,
    user_id: Optional[int],
    model_key: str,
    index_name: str,
    use_semantic: bool,
    filters: Dict[str, Any],
    response_json: Dict[str, Any],
    request_k: Optional[int],
    t0: float,
) -> None:
    """
    Helper to log search events to ClickHouse.
    Non-blocking: failures are caught and logged but not raised.
    """
    if ch_logger is None:
        return

    try:
        latency_ms = int((time.monotonic() - t0) * 1000)

        # IP + headers (Traefik/proxies)
        xff = request_temp.headers.get("x-forwarded-for", "")
        ip = (xff.split(",")[0].strip() if xff else (request_temp.client.host if request_temp.client else ""))

        user_agent = request_temp.headers.get("user-agent", "")
        accept_language = request_temp.headers.get("accept-language", "")
        referer = request_temp.headers.get("referer", "")

        request_id = request_temp.headers.get("x-request-id") or request_temp.headers.get("x-correlation-id")

        MAX_LOG_RESULTS = 10
        results_data = response_json.get("data", [])
        result_orig_ids = [
            item.get("_id")
            for item in results_data
            if item.get("_id")
        ][:MAX_LOG_RESULTS]

        logger.info("Logging %d result_orig_ids; first=%s", len(result_orig_ids), result_orig_ids[0] if result_orig_ids else None)

        event = build_search_event(
            endpoint=endpoint,
            original_query=original_query,
            final_query=query,
            ip=ip,
            user_agent=user_agent,
            accept_language=accept_language,
            referer=referer,
            user_id=user_id,
            result_orig_ids=result_orig_ids,
            page=response_json.get("pagination", {}).get("current_page", 1),
            k=request_k,
            model_key=model_key,
            index_name=index_name,
            use_semantic=use_semantic,
            translation_allowed=translation_allowed,
            detected_lang=detected_lang,
            translated=translated,
            filters=filters,
            status_code=200,
            latency_ms=latency_ms,
            results_count=len(results_data),
            total_records=response_json.get("pagination", {}).get("total_records"),
            request_id=request_id,
        )

        # Never break search if analytics logging fails
        ch_logger.log_event(event)
    except Exception as e:
        logger.warning("ClickHouse logging failed: %s", e)
