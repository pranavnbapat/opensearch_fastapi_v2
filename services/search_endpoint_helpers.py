# services/search_endpoint_helpers.py

import logging
import time

from typing import Any, Dict, Optional

from services.clickhouse_logger import build_search_event
from services.language_detect import detect_language, translate_text_with_backoff, DEEPL_SUPPORTED_LANGUAGES
from services.utils import is_translation_allowed, jwt_claim, PAGE_SIZE, fetch_chunks_for_parents


logger = logging.getLogger(__name__)


def maybe_translate_query(query: str, translation_allowed: bool) -> str:
    """
    Translates query to EN if allowed and detected language is supported.
    Returns possibly modified query.
    """
    if not translation_allowed:
        logger.info("No access token provided; skipping translation.")
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

    return translation_allowed, include_summary, user_id


async def build_response_json(
    response: Dict[str, Any],
    index_name: str,
    page_number: int,
    include_fulltext: bool,
    include_summary: bool,
    query: str,
    summary_provider: str,
    summarise_top5_hf_fn,  # pass the function in to avoid circular imports
) -> Dict[str, Any]:
    grouped = response.get("grouped", {}) or {}
    parents = grouped.get("parents", []) or []

    total_parents = grouped.get("total_parents")
    if total_parents is None:
        total_parents = len(parents)

    parent_ids = [p.get("parent_id") for p in parents if p.get("parent_id")]
    chunks_map = fetch_chunks_for_parents(index_name, parent_ids) if (include_fulltext and parent_ids) else {}

    formatted_results = []
    for p in parents:
        pid = p.get("parent_id")

        ko_chunks = [c["content"] for c in chunks_map.get(pid, [])] if include_fulltext else None

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
            "title_original": p.get("title_original"),

            "subtitle": p.get("subtitle") or "",
            "subtitle_original": p.get("subtitle_original"),

            "description": p.get("description"),
            "description_original": p.get("description_original"),

            "keywords": p.get("keywords") or [],
            "keywords_original": p.get("keywords_original") or [],

            "projectAcronym": p.get("project_acronym"),
            "projectName": p.get("project_name"),
            "project_type": p.get("project_type"),
            "project_id": p.get("project_id"),
            "topics": p.get("topics") or [],
            "themes": p.get("themes") or [],

            "languages": p.get("languages") or [],
            "locations": p.get("locations") or [],
            "category": p.get("category"),
            "subcategories": p.get("subcategories") or [],
            "creators": p.get("creators") or [],
            "dateCreated": date_created,
            "@id": p.get("@id"),
            "_orig_id": p.get("_orig_id"),
            "_tags": p.get("keywords") or [],
        }

        if include_fulltext:
            item["ko_content_flat"] = ko_chunks or []

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
        summary = summarise_top5_hf_fn(query=query, hits=formatted_results)

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
