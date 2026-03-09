# services/utils.py

import base64
import httpx
import json
import logging
import nltk
import os
import re

from collections import defaultdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

from nltk.corpus import stopwords
from opensearchpy import OpenSearch, RequestsHttpConnection
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stopwords(lang="english"):
    try:
        return set(stopwords.words(lang))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words(lang))

STOPWORDS = get_stopwords()

PAGE_SIZE = 10

MODEL_CONFIG = {
    "msmarco": {
        "index": "neural_search_index_msmarco_distilbert",
        "model_id": "ovClkJsB_qkkFA9OzmRk"
    }
}

model_id = os.getenv("OPENSEARCH_MSMARCO_MODEL_ID", "ovClkJsB_qkkFA9OzmRk")  # Fallback to default if not set

# Fetch OpenSearch credentials
OPENSEARCH_API = os.getenv("OPENSEARCH_API")
OPENSEARCH_USR = os.getenv("OPENSEARCH_USR")
OPENSEARCH_PWD = os.getenv("OPENSEARCH_PWD")

BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER")
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS")

if not all([OPENSEARCH_API, OPENSEARCH_USR, OPENSEARCH_PWD]):
    raise EnvironmentError("Missing OpenSearch environment variables!")

# OpenSearch Client
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_API, "port": 443}],
    http_auth=(OPENSEARCH_USR, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=True,
    http_compress=True,
    connection_class=RequestsHttpConnection,
    timeout=10,
    max_retries=2,
    retry_on_timeout=False
)


class MultiUserTimedAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, users: dict):
        super().__init__(app)
        self.users = users

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Basic "):
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Auth required")

        try:
            encoded = auth.split(" ")[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Invalid auth")

        user = self.users.get(username)
        if not user or user["password"] != password:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Invalid credentials")

        # Time-based access restriction
        expires = user.get("expires")
        if expires and datetime.now() > expires:
            return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"},
                            content="Access expired")

        return await call_next(request)


def remove_stopwords_from_query(query: str) -> str:
    tokens = query.lower().split()
    return ' '.join([word for word in tokens if word not in STOPWORDS])


def group_hits_by_parent(hits, parents_size=PAGE_SIZE):
    """
    Groups collapsed hits by parent_id while PRESERVING the incoming hit order
    (which already reflects the OpenSearch sort clause).
    """
    grouped = {}
    order = []  # keep first-seen parent order

    for h in hits:
        src = h.get("_source", {})
        pid = src.get("parent_id") or src.get("_orig_id")
        if not pid:
            continue

        if pid not in grouped:
            grouped[pid] = {
                "parent_id": pid,
                "project_name": src.get("project_name"),
                "project_acronym": src.get("project_acronym"),

                "title": src.get("title"),
                "subtitle": src.get("subtitle"),
                "description": src.get("description"),
                "keywords": src.get("keywords"),

                "title_original": src.get("title_original"),

                "subtitle_original": src.get("subtitle_original"),

                "description_original": src.get("description_original"),

                "keywords_original": src.get("keywords_original"),

                "topics": src.get("topics"),
                "themes": src.get("themes"),
                "locations": src.get("locations"),
                "languages": src.get("languages"),
                "category": src.get("category"),
                "subcategories": src.get("subcategories"),
                "date_of_completion": src.get("date_of_completion"),
                "creators": src.get("creators"),
                "intended_purposes": src.get("intended_purposes"),
                "project_id": src.get("project_id"),
                "project_type": src.get("project_type"),
                "project_url": src.get("project_url"),
                "@id": src.get("@id"),
                "_orig_id": src.get("_orig_id"),

                # include the new date fields so you can see what you sorted by
                "ko_created_at": src.get("ko_created_at"),
                "ko_updated_at": src.get("ko_updated_at"),
                "proj_created_at": src.get("proj_created_at"),
                "proj_updated_at": src.get("proj_updated_at"),

                "max_score": 0.0,
            }
            order.append(pid)

        # track max score per parent (for display/analytics only)
        score_raw = h.get("_score")
        score = score_raw if isinstance(score_raw, (int, float)) else 0.0
        if score > grouped[pid]["max_score"]:
            grouped[pid]["max_score"] = score

    # PRESERVE OS order: no re-sorting here
    parents = [grouped[pid] for pid in order][:parents_size]
    return {"total_parents": len(order), "parents": parents}

def fetch_chunks_for_parents(index_name: str, parent_ids: list[str]) -> dict[str, list[dict]]:
    """
    Return all chunks for each parent_id, ordered by chunk_index.
    Shape per chunk: {"chunk_index": int, "content": str, "_id": str}
    """
    if not parent_ids:
        return {}

    body = {
        "_source": ["parent_id", "content_chunk", "chunk_index"],
        "size": 10000,
        "query": {
            "bool": {
                "filter": [{"terms": {"parent_id": parent_ids}}]
            }
        },
        "sort": [
            {"parent_id": "asc"},
            {"chunk_index": "asc"}
        ]
    }
    resp = client.search(index=index_name, body=body)

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        # skip meta docs or empties
        if src.get("chunk_index", -1) < 0:
            continue
        txt = src.get("content_chunk", "")
        if not isinstance(txt, str) or not txt.strip():
            continue

        by_parent[src["parent_id"]].append({
            "chunk_index": src.get("chunk_index"),
            "content": txt,
            "_id": h.get("_id"),
        })

    return by_parent

def build_sort(sort_by: str, has_query: bool):
    """
    Returns an OpenSearch sort clause for the top-level request.
    `sort_by` is a canonical string like 'ko_updated_at_desc' or 'score_desc'.
    """
    mapping = {
        "score_desc": [{"_score": "desc"}, {"chunk_index": "asc"}],
        "score_asc":  [{"_score": "asc"},  {"chunk_index": "asc"}],

        "ko_created_at_desc":  [
            {"ko_created_at":  {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "ko_created_at_asc":   [
            {"ko_created_at":  {"order": "asc",  "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "ko_updated_at_desc":  [
            {"ko_updated_at":  {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "ko_updated_at_asc":   [
            {"ko_updated_at":  {"order": "asc",  "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "proj_created_at_desc": [
            {"proj_created_at":{"order": "desc", "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "proj_created_at_asc":  [
            {"proj_created_at":{"order": "asc",  "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "proj_updated_at_desc": [
            {"proj_updated_at":{"order": "desc", "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
        "proj_updated_at_asc":  [
            {"proj_updated_at":{"order": "asc",  "unmapped_type": "date", "missing": "_last"}},
            {"_score": "desc"}
        ],
    }
    if not has_query and sort_by in ("score_desc", "score_asc"):
        return [{"chunk_index": "asc"}, {"_id": "asc"}]
    return mapping.get(sort_by, mapping["score_desc"])


def build_sort_hybrid(sort_by: str, has_query: bool):
    """
    Hybrid queries: OpenSearch forbids combining _score sort with any other sort criteria.
    So:
      - score_* => ONLY _score
      - date sorts => date field only (optionally + a non-_score tiebreaker)
      - no query + score sort => stable deterministic order
    """
    if not has_query and sort_by in ("score_desc", "score_asc"):
        return [{"chunk_index": "asc"}, {"_id": "asc"}]

    if sort_by == "score_desc":
        return [{"_score": {"order": "desc"}}]
    if sort_by == "score_asc":
        return [{"_score": {"order": "asc"}}]

    mapping = {
        "ko_created_at_desc": [{"ko_created_at": {"order": "desc", "unmapped_type": "date", "missing": "_last"}}],
        "ko_created_at_asc":  [{"ko_created_at": {"order": "asc",  "unmapped_type": "date", "missing": "_last"}}],
        "ko_updated_at_desc": [{"ko_updated_at": {"order": "desc", "unmapped_type": "date", "missing": "_last"}}],
        "ko_updated_at_asc":  [{"ko_updated_at": {"order": "asc",  "unmapped_type": "date", "missing": "_last"}}],
        "proj_created_at_desc": [{"proj_created_at": {"order": "desc", "unmapped_type": "date", "missing": "_last"}}],
        "proj_created_at_asc":  [{"proj_created_at": {"order": "asc",  "unmapped_type": "date", "missing": "_last"}}],
        "proj_updated_at_desc": [{"proj_updated_at": {"order": "desc", "unmapped_type": "date", "missing": "_last"}}],
        "proj_updated_at_asc":  [{"proj_updated_at": {"order": "asc",  "unmapped_type": "date", "missing": "_last"}}],
    }
    # Optional tiebreaker that is NOT _score:
    base = mapping.get(sort_by, [{"_score": "desc"}])
    if base and list(base[0].keys())[0] != "_score":
        base = base + [{"parent_id": "asc"}]
    return base


VALIDATE_ACCESS_TOKEN_DEV_URL = os.getenv(
    "VALIDATE_ACCESS_TOKEN_DEV_URL",
    "https://backend-admin.dev.farmbook.ugent.be/fastapi/validate_access_token/",
)

VALIDATE_ACCESS_TOKEN_PRD_URL = os.getenv(
    "VALIDATE_ACCESS_TOKEN_PRD_URL",
    "https://backend-admin.prd.farmbook.ugent.be/fastapi/validate_access_token/",
)

async def is_translation_allowed(access_token: str | None, is_dev: bool) -> bool:
    """
    Check with the Django backend whether the given access_token is valid.
    """
    if not access_token:
        return False

    url = VALIDATE_ACCESS_TOKEN_DEV_URL if is_dev else VALIDATE_ACCESS_TOKEN_PRD_URL

    try:
        # Use a short timeout so search is not blocked too long by auth issues
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(url, json={"access_token": access_token})

        if resp.status_code != 200:
            logger.warning(
                f"Access token validation failed with HTTP {resp.status_code} "
                f"for URL {url}"
            )
            return False

        data = resp.json()
        # Django validate_access_token_api returns: {'status': 'success', ...} on success
        status_value = data.get("status")
        if status_value == "success":
            logger.info("Access token valid; enabling DeepL translation.")
            return True

        logger.warning(
            f"Access token not accepted by backend: {data!r}"
        )
        return False

    except Exception as e:
        # On any error (network, JSON, etc.), fail closed: no translation
        logger.error(f"Error while validating access token for translation: {e}")
        return False

def jwt_claim(token: str, key: str) -> Optional[str]:
    """Extract a claim from JWT payload without verifying signature (analytics only)."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None

        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # base64url padding

        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        )
        val = payload.get(key)
        return str(val) if val is not None else None
    except Exception:
        return None


def safe_filename(s: str, max_len: int = 60) -> str:
    # Keep filenames portable and not insane
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    cleaned = "".join(ch if ch in allowed else "_" for ch in (s or ""))
    return cleaned[:max_len] or "query"


def save_debug_dump(*, debug_obj: dict, base_dir: str, index_name: str, user_id: str | None, query: str) -> str:
    """
    Saves debug JSON to disk and returns the absolute filepath.
    NOTE: use only in dev/test unless you have a retention/PII policy.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    q_part = safe_filename(query)
    u_part = safe_filename(user_id or "anon", max_len=24)
    i_part = safe_filename(index_name, max_len=48)

    filename = f"hybrid_debug__{ts}__{i_part}__{u_part}__{q_part}.json"
    path = Path(base_dir) / filename

    with path.open("w", encoding="utf-8") as f:
        json.dump(debug_obj, f, ensure_ascii=False, indent=2)

    return str(path.resolve())


def _env_int(name: str, default: int, fallback_env: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and fallback_env:
        raw = os.getenv(fallback_env)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid int env %s=%r. Using default=%d", name, raw, default)
        return default


def _compile_hint_regex(env_name: str, default_pattern: str, fallback_env: str | None = None) -> re.Pattern:
    raw = os.getenv(env_name)
    if raw is None and fallback_env:
        raw = os.getenv(fallback_env)
    if raw is None:
        raw = default_pattern
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error:
        logger.warning("Invalid regex env %s=%r. Falling back to default.", env_name, raw)
        return re.compile(default_pattern, re.IGNORECASE)


def infer_query_intent(
    q: str,
    *,
    code_hint_env: str = "QUERY_CODE_HINT_REGEX",
    code_hint_fallback_env: str | None = None,
    acronym_min_len_env: str = "QUERY_ACRONYM_MIN_LEN",
    acronym_min_len_fallback_env: str | None = None,
    acronym_max_len_env: str = "QUERY_ACRONYM_MAX_LEN",
    acronym_max_len_fallback_env: str | None = None,
    acronym_min_caps_env: str = "QUERY_ACRONYM_MIN_CAPS",
    acronym_min_caps_fallback_env: str | None = None,
) -> tuple[bool, bool, bool, bool, bool]:
    """
    Returns:
      (use_semantic, looks_like_code_or_id, looks_like_acronym, looks_like_quoted, very_short)

    Env names are parameterized so endpoints can tune intent rules independently.
    """
    query = (q or "").strip()
    q_tokens = query.split()

    code_hint_re = _compile_hint_regex(
        code_hint_env,
        r"\d|[_:/]|cve-|doi|isbn",
        fallback_env=code_hint_fallback_env,
    )
    looks_like_code_or_id = bool(code_hint_re.search(query))

    acronym_min_len = _env_int(acronym_min_len_env, 2, fallback_env=acronym_min_len_fallback_env)
    acronym_max_len = _env_int(acronym_max_len_env, 12, fallback_env=acronym_max_len_fallback_env)
    acronym_min_caps = _env_int(acronym_min_caps_env, 2, fallback_env=acronym_min_caps_fallback_env)

    if (
        len(q_tokens) == 1
        and acronym_min_len <= len(query) <= acronym_max_len
        and re.fullmatch(r"[A-Za-z]+", query)
    ):
        cap_count = sum(1 for ch in query if ch.isupper())
        looks_like_acronym = cap_count >= acronym_min_caps
    else:
        looks_like_acronym = False

    looks_like_quoted = ('"' in query) or ("'" in query)
    very_short = len(q_tokens) <= 2

    use_semantic = not (looks_like_code_or_id or looks_like_quoted or looks_like_acronym)
    if very_short and not (looks_like_code_or_id or looks_like_acronym):
        use_semantic = True

    return use_semantic, looks_like_code_or_id, looks_like_acronym, looks_like_quoted, very_short
