import asyncio
import ast
import json
import logging
import os
import re

from typing import Any, Dict, List, Optional, TypedDict

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


class SummarySegment(TypedDict):
    text: str
    highlight: bool


class SummaryPayload(TypedDict):
    summary: str
    summary_segments: List[SummarySegment]


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
        "- Use at most 120 words.",
        "- Do not mention the number of results.",
        "- Do not mention that you used search results or documents.",
        "- Do not add bullets, markdown, or explanations.",
        "- Highlight only the most relevant topic phrases, not every keyword.",
        "- If the evidence is insufficient, return exactly: null",
        "- Return valid JSON only with this shape:",
        '  {"summary":"...","summary_segments":[{"text":"...","highlight":false}]}',
        '- The concatenated "text" values in summary_segments must equal summary exactly.',
        '- Use boolean true/false for "highlight".',
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


def _extract_json_object(content: str) -> Optional[str]:
    content = (content or "").strip()
    if not content:
        return None

    code_fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if code_fence_match:
        return code_fence_match.group(1)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return content[start:end + 1]


def _coerce_payload(candidate: Any) -> Optional[SummaryPayload]:
    if not isinstance(candidate, dict):
        return None

    summary = candidate.get("summary")
    raw_segments = candidate.get("summary_segments")
    if not isinstance(summary, str):
        return None
    summary = summary.strip()
    if not summary:
        return None

    segments: List[SummarySegment] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text:
                continue
            segments.append({
                "text": text,
                "highlight": bool(item.get("highlight", False)),
            })

    if segments:
        joined = "".join(segment["text"] for segment in segments)
        if joined != summary:
            segments = []

    return {
        "summary": summary,
        "summary_segments": segments,
    }


def _extract_summary_and_segments_by_regex(content: str) -> Optional[SummaryPayload]:
    summary_match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"])*)"', content, re.DOTALL)
    if not summary_match:
        return None

    try:
        summary = json.loads(f'"{summary_match.group(1)}"')
    except Exception:
        return None

    segments_match = re.search(r'"summary_segments"\s*:\s*(\[[\s\S]*\])', content, re.DOTALL)
    if not segments_match:
        return {"summary": summary, "summary_segments": []}

    try:
        raw_segments = json.loads(segments_match.group(1))
    except Exception:
        return {"summary": summary, "summary_segments": []}

    return _coerce_payload({
        "summary": summary,
        "summary_segments": raw_segments,
    })


def _build_fallback_segments(summary: str, query: str) -> List[SummarySegment]:
    summary = (summary or "").strip()
    if not summary:
        return []

    query_terms = [
        term.strip()
        for term in re.split(r"\s+", query.strip())
        if term.strip()
    ]
    phrases = sorted(
        {query.strip(), *query_terms},
        key=len,
        reverse=True,
    )
    phrases = [phrase for phrase in phrases if len(phrase) >= 2]
    if not phrases:
        return [{"text": summary, "highlight": False}]

    pattern = re.compile("(" + "|".join(re.escape(phrase) for phrase in phrases) + ")", re.IGNORECASE)
    parts = pattern.split(summary)
    segments: List[SummarySegment] = []

    for part in parts:
        if not part:
            continue
        is_match = any(part.lower() == phrase.lower() for phrase in phrases)
        segments.append({"text": part, "highlight": is_match})

    return segments or [{"text": summary, "highlight": False}]


def _normalise_summary_payload(content: str, query: str) -> Optional[SummaryPayload]:
    extracted_json = _extract_json_object(content)
    if extracted_json:
        for loader in (json.loads, ast.literal_eval):
            try:
                payload = loader(extracted_json)
            except Exception:
                payload = None
            normalized = _coerce_payload(payload)
            if normalized:
                if not normalized["summary_segments"]:
                    normalized["summary_segments"] = _build_fallback_segments(normalized["summary"], query)
                return normalized

    regex_payload = _extract_summary_and_segments_by_regex(content)
    if regex_payload:
        if not regex_payload["summary_segments"]:
            regex_payload["summary_segments"] = _build_fallback_segments(regex_payload["summary"], query)
        return regex_payload

    plain_summary = (content or "").strip()
    if not plain_summary or plain_summary.lower() == "null":
        return None

    return {
        "summary": plain_summary,
        "summary_segments": _build_fallback_segments(plain_summary, query),
    }


async def summarise_topk_llm(query: str, hits: List[Dict[str, Any]]) -> Optional[SummaryPayload]:
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

    payload = _normalise_summary_payload(content, query)
    if not payload:
        logging.getLogger(__name__).info(
            "LLM summary empty/null response: query=%r raw_content=%r",
            query,
            content,
        )
        return None

    logging.getLogger(__name__).info(
        "LLM summary success: query=%r content_preview=%r segments=%d",
        query,
        payload["summary"][:160],
        len(payload["summary_segments"]),
    )
    return payload
