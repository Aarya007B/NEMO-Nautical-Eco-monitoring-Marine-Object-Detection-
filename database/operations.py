"""
database/operations.py — NEMO MongoDB CRUD operations.

Provides typed storage and retrieval for missions, detections, and reports.
All operations gracefully degrade when MongoDB is not configured.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from database.connection import get_collection, is_connected

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------

def store_mission(
    mission_id: str,
    mission_dir: str,
    total_frames: int,
    total_detections: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Store a processed mission record.

    Returns:
        Inserted document ID as string, or None if MongoDB is not available.
    """
    col = get_collection("missions")
    if col is None:
        return None

    doc = {
        "mission_id": mission_id,
        "mission_dir": mission_dir,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "processed_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
    }

    try:
        result = col.insert_one(doc)
        logger.info("MongoDB: stored mission %s", mission_id)
        return str(result.inserted_id)
    except Exception as e:
        logger.warning("MongoDB: failed to store mission %s: %s", mission_id, e)
        return None


def get_mission(mission_id: str) -> Optional[Dict]:
    """Retrieve a mission by ID."""
    col = get_collection("missions")
    if col is None:
        return None

    try:
        doc = col.find_one({"mission_id": mission_id}, {"_id": 0})
        return doc
    except Exception as e:
        logger.warning("MongoDB: failed to get mission %s: %s", mission_id, e)
        return None


def list_missions(limit: int = 50) -> List[Dict]:
    """List recent missions, most recent first."""
    col = get_collection("missions")
    if col is None:
        return []

    try:
        cursor = col.find({}, {"_id": 0}).sort("processed_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.warning("MongoDB: failed to list missions: %s", e)
        return []


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def store_detections(
    mission_id: str,
    frame_id: str,
    detections: List[Dict[str, Any]],
) -> int:
    """
    Store detections for a single frame.

    Args:
        mission_id: Parent mission identifier.
        frame_id: Frame identifier.
        detections: List of detection dicts.

    Returns:
        Number of detections inserted, or 0 if MongoDB is not available.
    """
    col = get_collection("detections")
    if col is None:
        return 0

    docs = []
    for det in detections:
        doc = {
            "mission_id": mission_id,
            "frame_id": frame_id,
            "stored_at": datetime.utcnow().isoformat(),
            **det,
        }
        docs.append(doc)

    if not docs:
        return 0

    try:
        result = col.insert_many(docs)
        logger.debug("MongoDB: stored %d detections for frame %s", len(result.inserted_ids), frame_id)
        return len(result.inserted_ids)
    except Exception as e:
        logger.warning("MongoDB: failed to store detections: %s", e)
        return 0


def get_detections(
    mission_id: str,
    frame_id: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    """Retrieve detections for a mission (optionally filtered by frame)."""
    col = get_collection("detections")
    if col is None:
        return []

    query = {"mission_id": mission_id}
    if frame_id:
        query["frame_id"] = frame_id

    try:
        cursor = col.find(query, {"_id": 0}).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.warning("MongoDB: failed to get detections: %s", e)
        return []


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def store_report(
    mission_id: str,
    report_type: str,
    report_path: str,
    summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Store a report record.

    Args:
        mission_id: Parent mission identifier.
        report_type: 'json' or 'csv'.
        report_path: Local file path to the report.
        summary: Optional summary metadata.

    Returns:
        Inserted document ID as string, or None.
    """
    col = get_collection("reports")
    if col is None:
        return None

    doc = {
        "mission_id": mission_id,
        "report_type": report_type,
        "report_path": report_path,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary or {},
    }

    try:
        result = col.insert_one(doc)
        logger.info("MongoDB: stored %s report for mission %s", report_type, mission_id)
        return str(result.inserted_id)
    except Exception as e:
        logger.warning("MongoDB: failed to store report: %s", e)
        return None


def get_reports(mission_id: Optional[str] = None) -> List[Dict]:
    """Retrieve reports, optionally filtered by mission."""
    col = get_collection("reports")
    if col is None:
        return []

    query = {"mission_id": mission_id} if mission_id else {}

    try:
        cursor = col.find(query, {"_id": 0}).sort("generated_at", -1)
        return list(cursor)
    except Exception as e:
        logger.warning("MongoDB: failed to get reports: %s", e)
        return []
