"""
database/connection.py — MongoDB Atlas connection manager for NEMO.

Connects to MongoDB Atlas using credentials from environment variables.
The frontend/browser must NEVER connect directly to MongoDB.

Environment Variables:
    NEMO_MONGO_URI       — MongoDB Atlas connection string
                           (e.g. mongodb+srv://user:pass@cluster.mongodb.net/)
    NEMO_MONGO_DB        — Database name (default: nemo)

Usage:
    from database.connection import get_database, get_collection

    db = get_database()
    missions = get_collection("missions")
    missions.insert_one({...})
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None
_db = None


def get_mongo_client():
    """
    Get or create a singleton MongoDB client.

    Reads NEMO_MONGO_URI from environment. Returns None if
    MongoDB is not configured (allows graceful degradation).
    """
    global _client

    if _client is not None:
        return _client

    mongo_uri = os.environ.get("NEMO_MONGO_URI")
    if not mongo_uri:
        logger.info(
            "NEMO_MONGO_URI not set. MongoDB integration disabled. "
            "Results will be stored locally via JSON/CSV export only."
        )
        return None

    try:
        from pymongo import MongoClient
        _client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        # Verify connectivity
        _client.admin.command("ping")
        logger.info("MongoDB Atlas: connected successfully")
        return _client
    except ImportError:
        logger.warning(
            "pymongo not installed. MongoDB integration disabled. "
            "Install with: pip install pymongo[srv]"
        )
        return None
    except Exception as e:
        logger.warning("MongoDB Atlas: connection failed: %s", e)
        return None


def get_database(db_name: Optional[str] = None):
    """
    Get the NEMO MongoDB database instance.

    Args:
        db_name: Database name override (default from NEMO_MONGO_DB env var or 'nemo').

    Returns:
        pymongo Database instance, or None if MongoDB is not configured.
    """
    global _db

    if _db is not None and db_name is None:
        return _db

    client = get_mongo_client()
    if client is None:
        return None

    name = db_name or os.environ.get("NEMO_MONGO_DB", "nemo")
    _db = client[name]
    logger.info("MongoDB Atlas: using database '%s'", name)
    return _db


def get_collection(collection_name: str):
    """
    Get a MongoDB collection by name.

    Args:
        collection_name: One of 'missions', 'detections', 'reports'.

    Returns:
        pymongo Collection instance, or None if MongoDB is not configured.
    """
    db = get_database()
    if db is None:
        return None
    return db[collection_name]


def is_connected() -> bool:
    """Check if MongoDB Atlas is connected."""
    client = get_mongo_client()
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False


def close():
    """Close the MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB Atlas: connection closed")
