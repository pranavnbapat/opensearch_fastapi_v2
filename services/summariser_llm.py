import asyncio
import logging
import os
import re

from typing import Any, Dict, List, Optional

import httpx


LLM_URL = (os.getenv("LLM_URL") or "").strip()
LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip()
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("LLM_KEY") or "").strip()

SUMMARY_TIMEOUT_S = float(os.getenv("SUMMARY_TIMEOUT_S", "10.0"))
SUMMARY_MAX_DESC_CHARS = int(os.getenv("SUMMARY_MAX_DESC_CHARS", "250"))
SUMMARY_MAX_CONCURRENCY = int(os.getenv("SUMMARY_MAX_CONCURRENCY", "4"))
SUMMARY_TOP_K = int(os.getenv("SUMMARY_TOP_K", "10"))
SUMMARY_MIN_MAX_TOKENS = int(os.getenv("SUMMARY_MIN_MAX_TOKENS", "100"))
SUMMARY_MAX_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_MAX_TOKENS", "300"))

_sem = asyncio.Semaphore(SUMMARY_MAX_CONCURRENCY)


def _clip(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _hit_relevance_score(query: str, h: Dict[str, Any]) -> int:
    q = _tokens(query)
    text = " ".join([
        h.get("title") or "",
        h.get("subtitle") or "",
        h.get("description") or "",
        " ".join(h.get("keywords") or []),
        " ".join(h.get("topics") or []),
        " ".join(h.get("themes") or []),
    ])
    return len(q & _tokens(text))


def _build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    lines = [
        f"Query: {query}",
        "",
        "Task: Write one short neutral summary about the query topic using only the evidence below.",
        "",
        "Rules:",
        "- Return plain text only.",
        "- Use at most 120 words.",
        "- Do not mention the number of results.",
        "- Do not mention that you used search results or documents.",
        "- Do not add bullets, markdown, or explanations.",
        "- If the evidence is insufficient, return exactly: null",
        "",
        "Evidence:",
    ]
    for idx, hit in enumerate(hits[:SUMMARY_TOP_K], start=1):
        lines.append(
            f"{idx}) {hit.get('title', '').strip()} — {_clip(hit.get('description'), SUMMARY_MAX_DESC_CHARS)}"
        )
    return "\n".join(lines)


def _summary_max_tokens(query: str, hits: List[Dict[str, Any]]) -> int:
    prompt_chars = len(_build_prompt(query, hits))
    dynamic_budget = 100 + (prompt_chars // 20)
    return max(SUMMARY_MIN_MAX_TOKENS, min(SUMMARY_MAX_MAX_TOKENS, dynamic_budget))


async def summarise_topk_llm(query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
    if not hits or not LLM_URL or not LLM_MODEL:
        logging.getLogger(__name__).info(
            "LLM summary skipped: hits=%d llm_url_present=%s llm_model_present=%s query=%r",
            len(hits) if hits else 0,
            bool(LLM_URL),
            bool(LLM_MODEL),
            query,
        )
        return None

    scored = sorted(
        ((h, _hit_relevance_score(query, h)) for h in hits),
        key=lambda item: item[1],
        reverse=True,
    )
    filtered = [hit for hit, score in scored if score > 0][:SUMMARY_TOP_K]
    max_tokens = _summary_max_tokens(query, filtered) if filtered else SUMMARY_MIN_MAX_TOKENS
    logging.getLogger(__name__).info(
        "LLM summary relevance filter: query=%r hits=%d filtered=%d top_scores=%s max_tokens=%d",
        query,
        len(hits),
        len(filtered),
        [score for _, score in scored[:5]],
        max_tokens,
    )
    if not filtered:
        return None

    prompt = _build_prompt(query, filtered)

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You write concise factual search summaries."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    endpoint = LLM_URL.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = f"{endpoint}/v1/chat/completions"

    timeout = httpx.Timeout(connect=5.0, read=SUMMARY_TIMEOUT_S, write=10.0, pool=5.0)

    try:
        async with _sem:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                logging.getLogger(__name__).info(
                    "LLM summary HTTP response: endpoint=%s status=%s query=%r",
                    endpoint,
                    response.status_code,
                    query,
                )
                response.raise_for_status()
                data = response.json()
    except Exception as exc:
        logging.getLogger(__name__).exception("LLM summary failed: %s", exc)
        return None

    try:
        content = (data.get("choices") or [])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return None

    if not content or content.lower() == "null":
        logging.getLogger(__name__).info(
            "LLM summary empty/null response: query=%r raw_content=%r",
            query,
            content,
        )
        return None

    logging.getLogger(__name__).info(
        "LLM summary success: query=%r content_preview=%r",
        query,
        content[:160],
    )
    return content
