"""
dashboard/app.py — NEMO REST API Server.

FastAPI backend for the NEMO system.
The frontend team can call these REST endpoints to build the UI.

Endpoints:
    GET  /api/health          — Health check
    GET  /api/missions        — List available missions
    POST /api/mission/process — Process a mission directory
    GET  /api/mission/{id}    — Get mission results
    GET  /api/mission/{id}/detections — Get detections for a mission
    POST /api/detect          — Run detection on a single image
    GET  /api/crop/{id}       — Get detection crop image
    GET  /api/export/json     — Export mission results as JSON
    GET  /api/export/csv      — Export mission results as CSV
    GET  /api/status          — System status and model info

Usage:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
    # or
    python dashboard/app.py
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NEMO API",
    description="AI-powered side-scan sonar debris detection REST API",
    version="0.1.0-mvp",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

_CONFIG = load_config()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "outputs"


def get_pipeline():
    """Lazily initialize and return the MissionPipeline."""
    from inference.pipeline import MissionPipeline
    from inference.detector import CandidateDetector
    from inference.verifier import CandidateVerifier
    from inference.crop_extractor import CropExtractor
    from inference.evidence_fusion import EvidenceFusion
    from metadata.alignment import MetadataAligner

    runtime = _CONFIG.get("runtime", {}).get("device", "auto")
    detector_cfg = _CONFIG.get("detector", {})
    verifier_cfg = _CONFIG.get("verifier", {})
    preprocessing_cfg = _CONFIG.get("preprocessing", {})
    scoring_cfg = _CONFIG.get("scoring", {})

    detector = CandidateDetector(
        detector_config=detector_cfg,
        preprocessing_config=preprocessing_cfg,
        device=runtime,
    )
    verifier = CandidateVerifier(
        verifier_config=verifier_cfg,
        device=runtime,
    )
    crop_extractor = CropExtractor(
        crop_size=verifier_cfg.get("crop_size", 128),
        bbox_padding=verifier_cfg.get("bbox_padding", 0.15),
        save_dir=str(OUTPUT_DIR / "crops"),
    )
    evidence_fusion = EvidenceFusion(scoring_config=scoring_cfg)
    metadata_aligner = MetadataAligner()

    return MissionPipeline(
        detector=detector,
        verifier=verifier,
        crop_extractor=crop_extractor,
        evidence_fusion=evidence_fusion,
        metadata_aligner=metadata_aligner,
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    detection_id: str
    bbox: dict
    detector_confidence: float
    detector_class: str
    verifier_class: Optional[str] = None
    artificial_probability: Optional[float] = None
    natural_probability: Optional[float] = None
    artificialness_score: float
    priority_score: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[str] = None
    status: str = "candidate"


class MissionResult(BaseModel):
    mission_id: str
    frame_id: str
    detections: List[DetectionResult] = []
    processing_time_ms: float = 0.0


class ProcessMissionRequest(BaseModel):
    mission_dir: str
    navigation_file: Optional[str] = None


class DetectImageRequest(BaseModel):
    image_path: str
    conf_threshold: Optional[float] = 0.25


# ---------------------------------------------------------------------------
# In-memory mission cache
# ---------------------------------------------------------------------------

_mission_cache: dict = {}


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    from database.connection import is_connected as mongo_connected
    return {
        "status": "healthy",
        "service": "NEMO",
        "version": "0.1.0-mvp",
        "device": _CONFIG.get("runtime", {}).get("device", "auto"),
        "torch_version": torch.__version__,
        "mongodb": "connected" if mongo_connected() else "not configured",
    }


@app.get("/api/status")
def system_status():
    """Get system status and model information."""
    from database.connection import is_connected as mongo_connected

    detector_cfg = _CONFIG.get("detector", {})
    verifier_cfg = _CONFIG.get("verifier", {})

    return {
        "detector_architecture": detector_cfg.get("architecture", "yolo11n_1c"),
        "verifier_architecture": verifier_cfg.get("architecture", "mobilenet_v3_small"),
        "device": _CONFIG.get("runtime", {}).get("device", "auto"),
        "confidence_threshold": detector_cfg.get("confidence_threshold", 0.25),
        "num_classes": detector_cfg.get("num_classes", 1),
        "image_size": _CONFIG.get("input", {}).get("image_size", 640),
        "scoring_weights": _CONFIG.get("scoring", {}),
        "preprocessing": {
            "normalize": _CONFIG.get("preprocessing", {}).get("normalize", False),
            "denoise": _CONFIG.get("preprocessing", {}).get("denoise", {}).get("enabled", False),
            "contrast": _CONFIG.get("preprocessing", {}).get("contrast", {}).get("enabled", False),
        },
        "mongodb": "connected" if mongo_connected() else "not configured",
    }


@app.get("/api/missions")
def list_missions():
    """List available mission reports in outputs/reports/."""
    reports_dir = OUTPUT_DIR / "reports"
    if not reports_dir.exists():
        return {"missions": [], "count": 0}

    reports = sorted(reports_dir.glob("mission_*.json"), reverse=True)
    missions = []
    for r in reports:
        try:
            data = json.loads(r.read_text())
            missions.append({
                "mission_id": data.get("mission_id", r.stem),
                "total_frames": data.get("total_frames", 0),
                "total_detections": data.get("total_detections", 0),
                "timestamp": data.get("export_timestamp", ""),
                "path": str(r),
            })
        except Exception:
            continue

    # Also list raw mission directories
    sample_mission = _PROJECT_ROOT / "data" / "sample_mission"
    if sample_mission.exists():
        missions.append({
            "mission_id": "SAMPLE_MISSION_001",
            "total_frames": 3,
            "total_detections": 0,
            "timestamp": "sample",
            "path": str(sample_mission),
        })

    return {"missions": missions, "count": len(missions)}


@app.post("/api/mission/process")
def process_mission(request: ProcessMissionRequest):
    """Process a recorded mission directory through the full pipeline."""
    from mission.recorded import RecordedMissionSource
    from inference.pipeline import MissionPipeline
    from reporting.report_generator import ReportGenerator

    mission_dir = request.mission_dir
    navigation_file = request.navigation_file

    source = RecordedMissionSource(mission_dir)
    if len(source) == 0:
        raise HTTPException(status_code=404, detail=f"No frames found in {mission_dir}")

    pipeline = MissionPipeline.from_config(
        _CONFIG,
        navigation_file=navigation_file or source.navigation_file,
    )

    all_results = pipeline.process_mission(source)

    # Generate reports
    reporter = ReportGenerator(_CONFIG.get("output", {}))
    outputs = reporter.generate(all_results, mission_id=source.mission_id)

    # Store to MongoDB Atlas (graceful — skips if not configured)
    from database import operations as db_ops
    total_dets = sum(len(r.detections) for r in all_results)
    db_ops.store_mission(
        mission_id=source.mission_id,
        mission_dir=mission_dir,
        total_frames=len(all_results),
        total_detections=total_dets,
    )
    for result in all_results:
        det_dicts = []
        for d in result.detections:
            det_dicts.append({
                "detection_id": d.detection_id,
                "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
                "detector_confidence": d.detector_confidence,
                "detector_class": d.detector_class,
                "verifier_class": d.verifier_class,
                "artificial_probability": d.artificial_probability,
                "artificialness_score": d.artificialness_score,
                "priority_score": d.priority_score,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
            })
        if det_dicts:
            db_ops.store_detections(source.mission_id, result.frame_id, det_dicts)
    for fmt, path in outputs.items():
        db_ops.store_report(source.mission_id, fmt, path)

    # Cache results
    _mission_cache[source.mission_id] = {
        "results": all_results,
        "outputs": outputs,
        "frame_count": len(all_results),
    }

    return {
        "mission_id": source.mission_id,
        "status": "processed",
        "frames_processed": len(all_results),
        "total_detections": total_dets,
        "outputs": outputs,
    }


@app.get("/api/mission/{mission_id}")
def get_mission(mission_id: str):
    """Get mission results by mission ID."""
    if mission_id in _mission_cache:
        cached = _mission_cache[mission_id]
        return {
            "mission_id": mission_id,
            "frame_count": cached["frame_count"],
            "outputs": cached["outputs"],
        }

    # Check output files
    reports_dir = OUTPUT_DIR / "reports"
    for r in reports_dir.glob(f"mission_{mission_id}*.json"):
        data = json.loads(r.read_text())
        return {"mission_id": mission_id, "data": data}

    raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")


@app.get("/api/mission/{mission_id}/detections")
def get_detections(mission_id: str):
    """Get all detections for a mission."""
    if mission_id in _mission_cache:
        results = _mission_cache[mission_id]["results"]
    else:
        reports_dir = OUTPUT_DIR / "reports"
        for r in reports_dir.glob(f"mission_{mission_id}*.json"):
            data = json.loads(r.read_text())
            return JSONResponse(data)
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")

    detections = []
    for result in results:
        for det in result.detections:
            detections.append({
                "detection_id": det.detection_id,
                "frame_id": result.frame_id,
                "mission_id": result.mission_id,
                "bbox": {
                    "x1": det.bbox.x1, "y1": det.bbox.y1,
                    "x2": det.bbox.x2, "y2": det.bbox.y2,
                },
                "detector_confidence": det.detector_confidence,
                "detector_class": det.detector_class,
                "verifier_class": det.verifier_class,
                "artificial_probability": det.artificial_probability,
                "natural_probability": det.natural_probability,
                "artificialness_score": det.artificialness_score,
                "priority_score": det.priority_score,
                "latitude": det.latitude,
                "longitude": det.longitude,
                "timestamp": det.timestamp,
                "status": det.status.value if hasattr(det.status, "value") else str(det.status),
                "crop_path": det.crop_path,
            })

    return {"mission_id": mission_id, "detections": detections, "count": len(detections)}


@app.post("/api/detect")
def run_detection(request: DetectImageRequest):
    """Run detection on a single image."""
    from inference.detector import CandidateDetector
    from inference.crop_extractor import CropExtractor
    from inference.verifier import CandidateVerifier
    from inference.evidence_fusion import EvidenceFusion
    from metadata.alignment import MetadataAligner
    from inference.types import DetectionResult, DetectionStatus
    from scoring.features import AcousticFeatureExtractor

    detector = CandidateDetector(
        detector_config=_CONFIG.get("detector", {}),
        preprocessing_config=_CONFIG.get("preprocessing", {}),
    )
    verifier = CandidateVerifier(
        verifier_config=_CONFIG.get("verifier", {}),
    )
    crop_extractor = CropExtractor(crop_size=128, bbox_padding=0.15)
    fusion = EvidenceFusion(scoring_config=_CONFIG.get("scoring", {}))
    aligner = MetadataAligner()

    image = cv2.imread(request.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not read image: {request.image_path}")

    detections, _, _, _ = detector.detect(image, frame_id="single")

    results = []
    for i, det in enumerate(detections):
        det_id = f"single_{i}"
        crop_tensor, crop_path = crop_extractor.extract(image, det.bbox, detection_id=det_id)
        verification = verifier.verify(crop_tensor)

        from inference.types import AcousticFeatures, FrameMetadata
        features = AcousticFeatureExtractor().extract(image, det.bbox)
        meta = FrameMetadata(frame_id=det_id)

        artificialness = fusion.artificialness_scorer.score(
            detector_confidence=det.detector_confidence,
            artificial_probability=verification.artificial_probability,
            features=features,
        )
        priority = fusion.priority_scorer.score(
            artificialness_score=artificialness,
            detector_confidence=det.detector_confidence,
            has_gps=meta.has_gps,
            has_timestamp=meta.has_timestamp,
        )

        results.append({
            "detection_id": det_id,
            "bbox": {"x1": det.bbox.x1, "y1": det.bbox.y1, "x2": det.bbox.x2, "y2": det.bbox.y2},
            "detector_confidence": det.detector_confidence,
            "detector_class": det.class_name,
            "verifier_class": verification.predicted_class,
            "artificial_probability": verification.artificial_probability,
            "natural_probability": verification.natural_probability,
            "artificialness_score": artificialness,
            "priority_score": priority,
            "status": "verified" if verification.predicted_class == "artificial" else "rejected",
            "crop_path": crop_path,
        })

    return {"image_path": request.image_path, "detections": results, "count": len(results)}


@app.get("/api/crop/{detection_id}")
def get_crop(detection_id: str):
    """Get a detection crop image by detection ID."""
    crop_dir = OUTPUT_DIR / "crops"
    # Search for the crop file
    for ext in [".png", ".jpg"]:
        crop_path = crop_dir / f"{detection_id}{ext}"
        if crop_path.exists():
            return FileResponse(str(crop_path), media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Crop {detection_id} not found")


@app.get("/api/export/json")
def export_json(mission_id: Optional[str] = None):
    """Export mission results as JSON."""
    from reporting.json_export import JSONExporter

    exporter = JSONExporter(str(OUTPUT_DIR / "reports"))

    if mission_id and mission_id in _mission_cache:
        results = _mission_cache[mission_id]["results"]
        path = exporter.export_mission(results, mission_id)
        return {"format": "json", "path": path, "mission_id": mission_id}

    # Try to find the latest mission
    reports_dir = OUTPUT_DIR / "reports"
    reports = sorted(reports_dir.glob("mission_*.json"))
    if reports:
        data = json.loads(reports[-1].read_text())
        return JSONResponse(data)

    raise HTTPException(status_code=404, detail="No missions to export")


@app.get("/api/export/csv")
def export_csv(mission_id: Optional[str] = None):
    """Export mission results as CSV."""
    from reporting.csv_export import CSVExporter

    exporter = CSVExporter(str(OUTPUT_DIR / "reports"))

    if mission_id and mission_id in _mission_cache:
        results = _mission_cache[mission_id]["results"]
        path = exporter.export_mission(results, mission_id)
        return {"format": "csv", "path": path, "mission_id": mission_id}

    reports_dir = OUTPUT_DIR / "reports"
    reports = sorted(reports_dir.glob("mission_*.csv"))
    if reports:
        return FileResponse(str(reports[-1]), media_type="text/csv")

    raise HTTPException(status_code=404, detail="No missions to export")


@app.get("/api/demo")
def get_demo_mission():
    """Get demo mission data for testing."""
    from dashboard.components.demo_data import get_demo_mission
    return get_demo_mission("NEMO-DEMO-MISSION")


@app.get("/api/config")
def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "detector": _CONFIG.get("detector", {}),
        "verifier": _CONFIG.get("verifier", {}),
        "scoring": _CONFIG.get("scoring", {}),
        "preprocessing": _CONFIG.get("preprocessing", {}),
        "input": _CONFIG.get("input", {}),
        "runtime": _CONFIG.get("runtime", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting NEMO REST API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
