"""dashboard — NEMO REST API server and UI components.

The FastAPI server (app.py) provides REST endpoints for the frontend team.
Legacy Streamlit components are preserved as reference for the frontend team.

API Endpoints:
    GET  /api/health
    GET  /api/status
    GET  /api/missions
    POST /api/mission/process
    GET  /api/mission/{mission_id}
    GET  /api/mission/{mission_id}/detections
    POST /api/detect
    GET  /api/crop/{detection_id}
    GET  /api/export/json
    GET  /api/export/csv
    GET  /api/demo
    GET  /api/config
"""
__all__ = []
