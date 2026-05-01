import os
from typing import Any, Dict, Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

from services.record_details import get_record_details_by_identifier

MONGO_DB_NAME = "logical_layer"
MONGO_COLLECTION_NAME = "ko_metadata"
SYNC_FIELDS = [
    "title",
    "title_original",
    "subtitle",
    "subtitle_original",
    "keywords",
    "keywords_original",
    "description",
    "description_original",
    "ko_content_flat_summarised",
]


class MongoKOMetadataSyncRequest(BaseModel):
    mode: Literal["DEV", "PRD"] = "DEV"
    dry_run: bool = True
    model: Optional[str] = "msmarco"
    limit: Optional[int] = None
    include_fulltext: Literal[True] = True


class MongoConfigError(RuntimeError):
    pass


def _resolve_mongo_uri(mode: str) -> str:
    mode_upper = mode.upper()
    key = f"MONGO_DB_URI_{mode_upper}"
    value = os.getenv(key)
    if value and value.strip():
        return value.strip()
    raise MongoConfigError(
        f"Missing MongoDB URI for mode={mode_upper}. Set {key}"
    )


def _extract_sync_payload(record_item: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in SYNC_FIELDS:
        if field in record_item:
            payload[field] = record_item.get(field)
    return payload


def sync_ko_metadata_from_opensearch(*, request: MongoKOMetadataSyncRequest, index_name: str) -> Dict[str, Any]:
    try:
        mongo_uri = _resolve_mongo_uri(request.mode)
    except MongoConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))

    client = MongoClient(mongo_uri)
    collection = client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

    query: Dict[str, Any] = {}
    projection = {"_id": 1, "@id": 1}
    cursor = collection.find(query, projection=projection).sort("_id", 1)
    if request.limit is not None and int(request.limit) > 0:
        cursor = cursor.limit(int(request.limit))

    stats: Dict[str, Any] = {
        "mode": request.mode,
        "dry_run": bool(request.dry_run),
        "db": MONGO_DB_NAME,
        "collection": MONGO_COLLECTION_NAME,
        "scanned": 0,
        "matched": 0,
        "updated": 0,
        "skipped_missing_lookup": 0,
        "skipped_missing_at_id": 0,
        "errors": 0,
        "samples": [],
    }

    try:
        for doc in cursor:
            stats["scanned"] += 1
            mongo_id = doc.get("_id")
            mongo_id_str = str(mongo_id) if mongo_id is not None else ""
            at_id = (doc.get("@id") or "").strip()
            lookup_used = None

            if not at_id:
                stats["skipped_missing_at_id"] += 1
                if len(stats["samples"]) < 20:
                    stats["samples"].append({
                        "mongo_id": mongo_id_str,
                        "at_id": at_id,
                        "status": "missing_at_id",
                    })
                continue

            try:
                details = get_record_details_by_identifier(
                    index_name=index_name,
                    at_id=at_id,
                    include_fulltext=True,
                )
                if details.get("data"):
                    lookup_used = "@id"

            except Exception as e:
                stats["errors"] += 1
                if len(stats["samples"]) < 20:
                    stats["samples"].append({
                        "mongo_id": mongo_id_str,
                        "at_id": at_id,
                        "status": "error",
                        "error": str(e),
                    })
                continue

            items = details.get("data") or []
            if not items:
                stats["skipped_missing_lookup"] += 1
                if len(stats["samples"]) < 20:
                    stats["samples"].append({
                        "mongo_id": mongo_id_str,
                        "at_id": at_id,
                        "status": "not_found",
                    })
                continue

            payload = _extract_sync_payload(items[0])
            if not payload:
                stats["skipped_missing_lookup"] += 1
                continue

            stats["matched"] += 1
            if len(stats["samples"]) < 20:
                stats["samples"].append({
                    "mongo_id": mongo_id_str,
                    "at_id": at_id,
                    "status": "dry_run" if request.dry_run else "updated",
                    "lookup_used": lookup_used,
                    "fields": sorted(payload.keys()),
                })

            if request.dry_run:
                continue

            result = collection.update_one(
                {"_id": mongo_id},
                {"$set": payload},
            )
            if result.modified_count > 0 or result.matched_count > 0:
                stats["updated"] += 1
    finally:
        client.close()

    return stats
