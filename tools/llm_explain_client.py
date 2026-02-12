# tools/llm_explain_client.py

from __future__ import annotations

import os
from typing import Optional

import requests


def explain_debug_non_technical(text: str, timeout_s: int = 30) -> Optional[str]:
    llm_url = (os.getenv("LLM_URL") or "").strip()
    llm_model = (os.getenv("LLM_MODEL") or "").strip()
    llm_key = (os.getenv("LLM_KEY") or "").strip() or None

    if not llm_url or not llm_model:
        return None  # not configured

    headers = {"Content-Type": "application/json"}
    if llm_key:
        headers["Authorization"] = f"Bearer {llm_key}"

    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a search relevance engineer. Explain the scoring/debug summary "
                    "in plain English, focusing on which fields/terms dominate and why."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }

    base = llm_url.rstrip("/")
    endpoint = base if base.endswith("/v1/chat/completions") else f"{base}/v1/chat/completions"

    r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
