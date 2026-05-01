import logging
from typing import Any, Dict, Optional, Literal

from fastapi import HTTPException
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError, TransportError
from pydantic import BaseModel, Field, model_validator

from services.search_endpoint_helpers import format_parent_result_item
from services.utils import client, fetch_chunks_for_parents


logger = logging.getLogger(__name__)


class RecordDetailsRequest(BaseModel):
    record_id: Optional[str] = Field(default=None, alias="_id")
    at_id: Optional[str] = Field(default=None, alias="@id")
    dev: Optional[bool] = False
    model: Optional[str] = "msmarco"
    include_fulltext: Literal[True] = True

    class Config:
        allow_population_by_field_name = True

    @model_validator(mode="after")
    def validate_exactly_one_identifier(self):
        has_record_id = bool((self.record_id or "").strip())
        has_at_id = bool((self.at_id or "").strip())
        if has_record_id == has_at_id:
            raise ValueError("Provide exactly one of: _id or @id")
        return self


SOURCE_INCLUDES = [
    "parent_id", "ko_id", "chunk_index", "@id", "_orig_id",
    "title", "title_original", "subtitle", "subtitle_original",
    "description", "description_original", "keywords", "keywords_original",
    "topics", "themes", "locations", "languages", "category", "subcategories",
    "creators", "intended_purposes", "date_of_completion",
    "project_name", "project_acronym", "project_display_name",
    "project_id", "project_type", "project_url",
    "ko_created_at", "ko_updated_at", "proj_created_at", "proj_updated_at",
    "ko_content_flat", "ko_content_flat_summarised",
]


def _build_lookup_query(*, record_id: Optional[str], at_id: Optional[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if record_id:
        return ({
            "bool": {
                "must": [
                    {"term": {"chunk_index": -1}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"parent_id": record_id}},
                                {"term": {"ko_id": record_id}},
                                {"term": {"_orig_id": record_id}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        }, {"match_field": "_id", "match_value": record_id})

    if at_id:
        return ({
            "bool": {
                "must": [
                    {"term": {"chunk_index": -1}},
                    {"term": {"@id": at_id}},
                ]
            }
        }, {"match_field": "@id", "match_value": at_id})

    raise HTTPException(status_code=422, detail="Provide one of: _id or @id")


def get_record_details(index_name: str, request: RecordDetailsRequest) -> Dict[str, Any]:
    record_id = (request.record_id or "").strip() or None
    at_id = (request.at_id or "").strip() or None

    query, lookup_meta = _build_lookup_query(record_id=record_id, at_id=at_id)

    body = {
        "size": 1,
        "_source": {"includes": SOURCE_INCLUDES},
        "query": query,
        "sort": [
            {"_score": {"order": "desc"}},
            {"ko_updated_at": {"order": "desc", "unmapped_type": "date", "missing": "_last"}},
        ],
    }

    try:
        exists = client.indices.exists(index=index_name, request_timeout=2)
        if not exists:
            return {
                "data": [],
                "_meta": {"error": f"Index not found: {index_name}", "index_missing": True, **lookup_meta},
            }

        resp = client.search(index=index_name, body=body)
    except (ConnectionTimeout, ConnectionError) as e:
        return {
            "data": [],
            "_meta": {"error": f"OpenSearch unreachable: {type(e).__name__}", "unreachable": True, **lookup_meta},
        }
    except TransportError as e:
        detail = getattr(e, "info", None) or {"error": str(e)}
        raise HTTPException(status_code=502, detail={
            "message": "OpenSearch error while fetching record details",
            "opensearch": detail,
        })

    hits = (resp.get("hits", {}) or {}).get("hits", []) or []
    if not hits:
        return {
            "data": [],
            "_meta": {"found": False, **lookup_meta},
        }

    hit = hits[0]
    src = (hit.get("_source") or {}).copy()
    parent_id = src.get("parent_id")

    raw_chunks = fetch_chunks_for_parents(index_name, [parent_id]).get(parent_id, []) if (request.include_fulltext and parent_id) else []
    fulltext_pages = src.get("ko_content_flat")
    if not isinstance(fulltext_pages, list):
        fulltext_pages = [c.get("content") for c in raw_chunks if c.get("content")]

    src["max_score"] = hit.get("_score")
    item = format_parent_result_item(
        src,
        fulltext_pages=fulltext_pages,
        fulltext_chunks=raw_chunks if request.include_fulltext else None,
        include_fulltext=bool(request.include_fulltext),
    )

    return {
        "data": [item],
        "_meta": {
            "found": True,
            "include_fulltext": bool(request.include_fulltext),
            **lookup_meta,
        },
    }


def get_record_details_by_identifier(
    *,
    index_name: str,
    record_id: Optional[str] = None,
    at_id: Optional[str] = None,
    include_fulltext: bool = True,
) -> Dict[str, Any]:
    request = RecordDetailsRequest(
        **{
            "_id": record_id,
            "@id": at_id,
            "include_fulltext": True if include_fulltext else True,
        }
    )
    return get_record_details(index_name=index_name, request=request)
