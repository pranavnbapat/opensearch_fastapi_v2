import os
from typing import Any, Dict, Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

from services.record_details import get_record_details_by_identifier

LOGICAL_MONGO_DB_NAME = "logical_layer"
PHYSICAL_MONGO_DB_NAME = "physical_layer"
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


def _extract_physical_sync_payload(record_item: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "ko_metadata.title": record_item.get("title"),
        "ko_metadata.subtitle": record_item.get("subtitle"),
        "ko_metadata.description": record_item.get("description"),
        "ko_metadata.keywords": record_item.get("keywords") or [],
    }
    return {k: v for k, v in payload.items() if v is not None}


def sync_ko_metadata_from_opensearch(*, request: MongoKOMetadataSyncRequest, index_name: str) -> Dict[str, Any]:
    try:
        mongo_uri = _resolve_mongo_uri(request.mode)
    except MongoConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))

    client = MongoClient(mongo_uri)
    logical_collection = client[LOGICAL_MONGO_DB_NAME][MONGO_COLLECTION_NAME]
    physical_collection = client[PHYSICAL_MONGO_DB_NAME][MONGO_COLLECTION_NAME]

    query: Dict[str, Any] = {}
    projection = {"_id": 1, "@id": 1, "physical_layer_ko_metadata_id": 1}
    cursor = logical_collection.find(query, projection=projection).sort("_id", 1)
    if request.limit is not None and int(request.limit) > 0:
        cursor = cursor.limit(int(request.limit))

    stats: Dict[str, Any] = {
        "mode": request.mode,
        "dry_run": bool(request.dry_run),
        "db": LOGICAL_MONGO_DB_NAME,
        "collection": MONGO_COLLECTION_NAME,
        "scanned": 0,
        "matched": 0,
        "updated": 0,
        "physical_updated": 0,
        "skipped_missing_lookup": 0,
        "skipped_missing_at_id": 0,
        "skipped_missing_physical_id": 0,
        "errors": 0,
        "samples": [],
    }

    try:
        for doc in cursor:
            stats["scanned"] += 1
            mongo_id = doc.get("_id")
            mongo_id_str = str(mongo_id) if mongo_id is not None else ""
            at_id = (doc.get("@id") or "").strip()
            physical_id = doc.get("physical_layer_ko_metadata_id")
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
            physical_payload = _extract_physical_sync_payload(items[0])

            stats["matched"] += 1
            if len(stats["samples"]) < 20:
                sample = {
                    "mongo_id": mongo_id_str,
                    "at_id": at_id,
                    "status": "dry_run" if request.dry_run else "updated",
                    "lookup_used": lookup_used,
                    "fields": sorted(payload.keys()),
                }
                if physical_id is not None:
                    sample["physical_mongo_id"] = str(physical_id)
                    sample["physical_fields"] = sorted(physical_payload.keys())
                else:
                    sample["physical_status"] = "missing_physical_layer_ko_metadata_id"
                stats["samples"].append(sample)

            if request.dry_run:
                continue

            result = logical_collection.update_one(
                {"_id": mongo_id},
                {"$set": payload},
            )
            if result.modified_count > 0 or result.matched_count > 0:
                stats["updated"] += 1
            if physical_id is None:
                stats["skipped_missing_physical_id"] += 1
                continue
            if not physical_payload:
                continue

            physical_result = physical_collection.update_one(
                {"_id": physical_id},
                {"$set": physical_payload},
            )
            if physical_result.modified_count > 0 or physical_result.matched_count > 0:
                stats["physical_updated"] += 1
    finally:
        client.close()

    return stats
