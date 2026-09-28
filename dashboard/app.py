"""
dashboard/app.py — NEMO REST API Server.

FastAPI backend for the NEMO system.
The frontend team can call these REST endpoints to build the UI.

Endpoints:
    GET  /api/health                       — Health check
    GET  /api/missions                     — List available missions
    POST /api/mission/process              — Process a mission directory (images)
    POST /api/mission/upload-video         — Upload a video, extract frames, run pipeline
    GET  /api/mission/video-status/{id}    — Poll video processing job status
    GET  /api/mission/{id}                 — Get mission results
    GET  /api/mission/{id}/detections      — Get detections for a mission
    POST /api/detect                       — Run detection on a single image
    GET  /api/crop/{id}                    — Get detection crop image
    GET  /api/export/json                  — Export mission results as JSON
    GET  /api/export/csv                   — Export mission results as CSV
    GET  /api/status                       — System status and model info

Usage:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
    # or
    python dashboard/app.py
"""
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request, BackgroundTasks
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
_DASHBOARD_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Static frontend routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_landing():
    """Serve the NEMO landing page (problem story + console entry)."""
    landing_path = _DASHBOARD_DIR / "landing.html"
    if landing_path.exists():
        return FileResponse(str(landing_path), headers={"Cache-Control": "no-store"})
    return await serve_console()


@app.get("/console", response_class=FileResponse)
async def serve_console():
    """Serve the NEMO Mission Dashboard interface."""
    index_path = _DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path), headers={"Cache-Control": "no-store"})


@app.get("/assets/{filename}")
async def serve_dashboard_asset(filename: str):
    """Serve bundled dashboard artwork (real sonar imagery, no external deps)."""
    safe = Path(filename).name
    if not safe or safe.startswith("."):
        raise HTTPException(status_code=404, detail="Asset not found")
    asset_path = _DASHBOARD_DIR / "assets" / safe
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Asset {safe} not found")
    return FileResponse(str(asset_path))


@app.get("/about", response_class=FileResponse)
async def serve_about():
    """Serve the NEMO Project Documentation / About page."""
    about_path = _DASHBOARD_DIR / "about.html"
    if not about_path.exists():
        raise HTTPException(status_code=404, detail="about.html not found")
    return FileResponse(str(about_path), headers={"Cache-Control": "no-store"})


@app.get("/index.html", response_class=FileResponse)
async def serve_index_alias():
    return await serve_console()


@app.get("/api/uploads/{filename}")
async def serve_uploaded_file(filename: str):
    """Serve uploaded video or image file."""
    file_path = OUTPUT_DIR / "uploads" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(str(file_path))


@app.get("/api/sample-image")
async def get_sample_sonar_image(index: int = 0):
    """Serve a real side-scan sonar image from the validation dataset for testing."""
    valid_dir = _PROJECT_ROOT / "data" / "raw" / "seabedobjects" / "valid" / "images"
    if valid_dir.exists():
        images = sorted(valid_dir.glob("*.jpg"))
        if images:
            img = images[index % len(images)]
            return FileResponse(
                str(img),
                media_type="image/jpeg",
                headers={"X-Image-Name": img.name, "Access-Control-Expose-Headers": "X-Image-Name"},
            )
    raise HTTPException(status_code=404, detail="No validation images found")


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


class VideoUploadResponse(BaseModel):
    job_id: str
    mission_id: str
    status: str                 # queued | processing | done | error
    frames_total: int = 0
    frames_done: int = 0
    total_detections: int = 0
    outputs: dict = {}
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

_mission_cache: dict = {}

# Video processing jobs: job_id -> VideoUploadResponse dict
_video_jobs: dict = {}


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
    missions = []
    seen = set()

    if reports_dir.exists():
        reports = sorted(reports_dir.glob("mission_*.json"), reverse=True)
        for r in reports:
            try:
                data = json.loads(r.read_text())
                m_id = data.get("mission_id", r.stem)
                if m_id in seen:
                    continue
                seen.add(m_id)
                missions.append({
                    "mission_id": m_id,
                    "total_frames": data.get("total_frames", 0),
                    "total_detections": data.get("total_detections", 0),
                    "timestamp": data.get("export_timestamp", ""),
                    "path": str(r),
                })
            except Exception:
                continue

    # Also list raw sample mission if exists
    sample_mission = _PROJECT_ROOT / "data" / "sample_mission"
    if sample_mission.exists() and "SAMPLE_MISSION_001" not in seen:
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


# ---------------------------------------------------------------------------
# Mission creation from browser uploads
# ---------------------------------------------------------------------------

MISSION_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MISSION_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_MISSION_FILES = 60
MAX_MISSION_FILE_MB = 40


def _sanitize_mission_id(raw: Optional[str]) -> str:
    """Keep only safe characters; fall back to a generated ID."""
    cleaned = "".join(c for c in (raw or "") if c.isalnum() or c in ("-", "_")).strip("_-")[:48]
    return cleaned or f"MISSION_{uuid.uuid4().hex[:8].upper()}"


def _mission_exists(mission_id: str) -> bool:
    if mission_id in _mission_cache:
        return True
    reports_dir = OUTPUT_DIR / "reports"
    if reports_dir.exists() and list(reports_dir.glob(f"mission_{mission_id}_*.json")):
        return True
    if (OUTPUT_DIR / "missions" / mission_id).exists():
        return True
    return False


def _finalize_mission(mission_id: str, all_results, mission_dir: str) -> dict:
    """Generate reports, store to DB (graceful), cache, and summarize."""
    from reporting.report_generator import ReportGenerator

    reporter = ReportGenerator(_CONFIG.get("output", {}))
    outputs = reporter.generate(all_results, mission_id=mission_id)

    from database import operations as db_ops
    total_dets = sum(len(r.detections) for r in all_results)
    db_ops.store_mission(
        mission_id=mission_id,
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
            db_ops.store_detections(mission_id, result.frame_id, det_dicts)
    for fmt, path in outputs.items():
        db_ops.store_report(mission_id, fmt, path)

    _mission_cache[mission_id] = {
        "results": all_results,
        "outputs": outputs,
        "frame_count": len(all_results),
    }

    verified = sum(
        1 for r in all_results for d in r.detections
        if getattr(d, "verifier_class", None) == "artificial"
    )
    geotagged = sum(
        1 for r in all_results for d in r.detections
        if getattr(d, "latitude", None) is not None
        and getattr(d, "longitude", None) is not None
    )
    return {
        "mission_id": mission_id,
        "status": "processed",
        "frames_processed": len(all_results),
        "total_detections": total_dets,
        "verified_detections": verified,
        "geotagged_detections": geotagged,
        "outputs": outputs,
    }


@app.post("/api/mission/create")
async def create_mission(
    mission_id: str = Form(...),
    files: List[UploadFile] = File(...),
    navigation: Optional[UploadFile] = File(None),
):
    """
    Create a mission from multiple uploaded sonar images.

    Saves the images as a mission directory, runs the full NEMO pipeline
    (YOLO11n + MobileNetV3 + reports), and registers the mission so it
    appears in ``GET /api/missions`` and ``GET /api/mission/{id}/detections``.

    Geotagging is honest by construction: detections receive coordinates
    only when an optional navigation CSV (timestamp,latitude,longitude)
    accompanies the upload. Frames pair to nav rows in capture order, and
    the pairing assumption is reported back in ``geotag_note``. Without
    navigation data, coordinates stay empty — never invented.
    """
    from mission.recorded import RecordedMissionSource
    from inference.pipeline import MissionPipeline
    from metadata.gps import GPSParser

    mid = _sanitize_mission_id(mission_id)
    if _mission_exists(mid):
        raise HTTPException(
            status_code=409,
            detail=f"Mission '{mid}' already exists. Choose a different name.",
        )
    if not files or len(files) > MAX_MISSION_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload between 1 and {MAX_MISSION_FILES} images per mission.",
        )

    sonar_dir = OUTPUT_DIR / "missions" / mid / "sonar"
    sonar_dir.mkdir(parents=True, exist_ok=True)
    saved_stems: List[str] = []
    try:
        for i, upload in enumerate(files):
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in MISSION_IMAGE_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file '{upload.filename}'. "
                           f"Mission images must be: {', '.join(sorted(MISSION_IMAGE_EXTS))}",
                )
            contents = await upload.read()
            if len(contents) > MAX_MISSION_FILE_MB * 1_048_576:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{upload.filename}' exceeds the {MAX_MISSION_FILE_MB} MB limit.",
                )
            image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{upload.filename}' is not a readable image.",
                )
            stem = Path(upload.filename or f"frame_{i:04d}").stem
            safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:40] or f"frame_{i:04d}"
            cv2.imwrite(str(sonar_dir / f"{safe_stem}.png"), image)
            saved_stems.append(safe_stem)

        mission_dir = OUTPUT_DIR / "missions" / mid
        geotag_note = "No navigation data — detections carry no coordinates."
        timestamps: dict = {}
        if navigation is not None and navigation.filename:
            nav_suffix = Path(navigation.filename).suffix.lower()
            if nav_suffix != ".csv":
                raise HTTPException(
                    status_code=400,
                    detail=f"Navigation file must be CSV, got '{navigation.filename}'.",
                )
            nav_path = mission_dir / "navigation.csv"
            nav_path.write_bytes(await navigation.read())
            records = GPSParser().parse(str(nav_path))
            usable = [r for r in records
                      if r.latitude is not None and r.longitude is not None]
            if not usable:
                nav_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="Navigation CSV has no rows with latitude and longitude. "
                           "Expected columns: timestamp, latitude, longitude.",
                )
            # Pair frames to nav rows in capture order (documented assumption).
            for i, stem in enumerate(saved_stems):
                ref = usable[min(i, len(usable) - 1)]
                timestamps[stem] = ref.timestamp
            if len(usable) < len(saved_stems):
                geotag_note = (
                    f"Navigation covers {len(usable)} of {len(saved_stems)} frames "
                    f"(paired in capture order); remaining frames carry no coordinates."
                )
            else:
                geotag_note = (
                    f"Frames paired to {len(usable)} navigation rows in capture order."
                )

        (mission_dir / "metadata.json").write_text(json.dumps({
            "mission_id": mid,
            "frames": len(saved_stems),
            "created_via": "dashboard-upload",
            "timestamps": timestamps,
        }))

        source = RecordedMissionSource(str(mission_dir))
        if len(source) == 0:
            raise HTTPException(status_code=400, detail="No readable frames found in upload.")

        pipeline = MissionPipeline.from_config(
            _CONFIG,
            navigation_file=source.navigation_file,
        )
        all_results = pipeline.process_mission(source)
        summary = _finalize_mission(mid, all_results, str(mission_dir))
        summary["geotag_note"] = geotag_note
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Mission creation failed for %s: %s", mid, exc)
        raise HTTPException(status_code=500, detail=f"Mission pipeline failed: {exc}")


# ---------------------------------------------------------------------------
# Video upload & processing
# ---------------------------------------------------------------------------

def _run_video_pipeline(
    job_id: str,
    video_path: str,
    mission_id: str,
    frame_interval: int,
    target_fps: Optional[float],
) -> None:
    """
    Background task: extract frames from a video and run the pipeline.
    Results are stored in _mission_cache and _video_jobs.
    """
    from mission.video import VideoMissionSource
    from inference.pipeline import MissionPipeline
    from reporting.report_generator import ReportGenerator
    from database import operations as db_ops

    job = _video_jobs[job_id]
    job["status"] = "processing"

    try:
        frames_dir = str(OUTPUT_DIR / "video_frames" / mission_id)

        source = VideoMissionSource(
            video_path=video_path,
            output_dir=frames_dir,
            frame_interval=frame_interval,
            target_fps=target_fps,
            mission_id=mission_id,
        )

        job["frames_total"] = len(source)

        pipeline = MissionPipeline.from_config(_CONFIG)

        all_results = []
        for frame in source:
            result = pipeline.process(frame)
            all_results.append(result)
            job["frames_done"] += 1

        # Reports
        reporter = ReportGenerator(_CONFIG.get("output", {}))
        outputs = reporter.generate(all_results, mission_id=mission_id)

        # MongoDB (graceful)
        total_dets = sum(len(r.detections) for r in all_results)
        db_ops.store_mission(
            mission_id=mission_id,
            mission_dir=video_path,
            total_frames=len(all_results),
            total_detections=total_dets,
        )
        for result in all_results:
            det_dicts = [
                {
                    "detection_id": d.detection_id,
                    "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1,
                             "x2": d.bbox.x2, "y2": d.bbox.y2},
                    "detector_confidence": d.detector_confidence,
                    "artificialness_score": d.artificialness_score,
                    "priority_score": d.priority_score,
                    "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                }
                for d in result.detections
            ]
            if det_dicts:
                db_ops.store_detections(mission_id, result.frame_id, det_dicts)
        for fmt, path in outputs.items():
            db_ops.store_report(mission_id, fmt, path)

        # Cache
        _mission_cache[mission_id] = {
            "results": all_results,
            "outputs": outputs,
            "frame_count": len(all_results),
        }

        job["status"] = "done"
        job["total_detections"] = total_dets
        job["outputs"] = outputs

        logger.info(
            "Video job %s complete: %d frames, %d detections",
            job_id, len(all_results), total_dets,
        )

    except Exception as exc:
        logger.exception("Video job %s failed: %s", job_id, exc)
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/mission/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Mission video file (MP4, AVI, MOV, MKV…)"),
    mission_id: Optional[str] = Query(
        default=None,
        description="Custom mission ID. Defaults to the uploaded filename stem.",
    ),
    frame_interval: int = Query(
        default=5,
        ge=1,
        le=300,
        description="Extract every N-th frame. Lower = more frames = slower.",
    ),
    target_fps: Optional[float] = Query(
        default=None,
        description="Alternatively, target frames-per-second to extract. Overrides frame_interval.",
    ),
):
    """
    Upload a mission video file, extract frames, and run the full NEMO pipeline.

    The endpoint returns immediately with a *job_id*. Poll
    ``GET /api/mission/video-status/{job_id}`` to track progress.
    When status == 'done', results are available at
    ``GET /api/mission/{mission_id}/detections``.
    """
    import shutil

    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
    if suffix not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}",
        )

    # Resolve mission ID
    stem = Path(file.filename or f"video_{uuid.uuid4().hex[:8]}").stem
    resolved_mission_id = mission_id or stem

    # Save the upload to disk
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{resolved_mission_id}{suffix}"

    with open(video_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    logger.info(
        "Video uploaded: %s → %s (%.1f MB)",
        file.filename, video_path,
        video_path.stat().st_size / 1_048_576,
    )

    # Create job record
    job_id = uuid.uuid4().hex[:12]
    _video_jobs[job_id] = {
        "job_id": job_id,
        "mission_id": resolved_mission_id,
        "status": "queued",
        "frames_total": 0,
        "frames_done": 0,
        "total_detections": 0,
        "outputs": {},
        "error": None,
    }

    # Kick off background processing
    background_tasks.add_task(
        _run_video_pipeline,
        job_id=job_id,
        video_path=str(video_path),
        mission_id=resolved_mission_id,
        frame_interval=frame_interval,
        target_fps=target_fps,
    )

    return VideoUploadResponse(**_video_jobs[job_id])


@app.get("/api/mission/video-status/{job_id}", response_model=VideoUploadResponse)
def video_status(job_id: str):
    """Poll the status of a video processing job."""
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return VideoUploadResponse(**_video_jobs[job_id])


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


@app.get("/api/mission/{mission_id}/frame/{frame_id}")
def get_mission_frame(mission_id: str, frame_id: str):
    """
    Serve one frame image of a mission.

    Looks in created-mission sonar directories first, then in extracted
    video frames. Recorded missions processed from arbitrary server paths
    are not served (their frames live outside the outputs tree).
    """
    mid = _sanitize_mission_id(mission_id)
    stem = "".join(c for c in Path(frame_id).stem if c.isalnum() or c in ("-", "_"))
    if not stem:
        raise HTTPException(status_code=404, detail="Frame not found")
    for candidate in (
        OUTPUT_DIR / "missions" / mid / "sonar" / f"{stem}.png",
        OUTPUT_DIR / "video_frames" / mid / f"{stem}.png",
    ):
        if candidate.is_file():
            return FileResponse(str(candidate), media_type="image/png")
    raise HTTPException(
        status_code=404,
        detail=f"Frame '{frame_id}' is not served for mission '{mid}'.",
    )


@app.get("/api/mission/{mission_id}/detections")
def get_detections(mission_id: str):
    """Get all detections for a mission."""
    if mission_id in _mission_cache:
        results = _mission_cache[mission_id]["results"]
        detections = []
        for result in results:
            for det in result.detections:
                w = round(det.bbox.x2 - det.bbox.x1, 1) if det.bbox else 0.0
                h = round(det.bbox.y2 - det.bbox.y1, 1) if det.bbox else 0.0
                is_accepted = det.verifier_class == "artificial"
                detections.append({
                    "detection_id": det.detection_id,
                    "frame_id": result.frame_id,
                    "mission_id": result.mission_id,
                    "bbox": {
                        "x1": round(det.bbox.x1, 1), "y1": round(det.bbox.y1, 1),
                        "x2": round(det.bbox.x2, 1), "y2": round(det.bbox.y2, 1),
                        "w": w, "h": h,
                    },
                    "detector_confidence": round(det.detector_confidence, 3),
                    "detector_class": det.detector_class,
                    "classification": det.detector_class,
                    "verifier_class": det.verifier_class,
                    "verifier_status": "accepted" if is_accepted else "rejected",
                    "artificial_probability": round(det.artificial_probability, 3) if det.artificial_probability is not None else None,
                    "natural_probability": round(det.natural_probability, 3) if det.natural_probability is not None else None,
                    "artificialness_score": round(det.artificialness_score, 3),
                    "priority_score": round(det.priority_score, 3),
                    "latitude": det.latitude,
                    "longitude": det.longitude,
                    "timestamp": det.timestamp,
                    "status": "verified" if is_accepted else "rejected",
                    "crop_path": det.crop_path,
                    "crop_url": f"/api/crop/{det.detection_id}" if det.crop_path else None,
                })
        return {
            "mission_id": mission_id,
            "detections": detections,
            "count": len(detections),
            "total_frames": len(results),
            "frames": [r.frame_id for r in results],
        }
    else:
        reports_dir = OUTPUT_DIR / "reports"
        matches = sorted(reports_dir.glob(f"mission_{mission_id}*.json"), reverse=True)
        if matches:
            data = json.loads(matches[0].read_text())
            detections = []
            for frame in data.get("frames", []):
                fid = frame.get("frame_id", "")
                for det in frame.get("detections", []):
                    det_copy = dict(det)
                    det_copy["frame_id"] = fid
                    det_copy["mission_id"] = mission_id
                    if "classification" not in det_copy:
                        det_copy["classification"] = det_copy.get("detector_class", "candidate")
                    if "verifier_status" not in det_copy:
                        vc = det_copy.get("verifier_class")
                        det_copy["verifier_status"] = "accepted" if vc == "artificial" else ("rejected" if vc == "natural" else "pending")
                    if "bbox" in det_copy:
                        b = det_copy["bbox"]
                        if "w" not in b and "x2" in b and "x1" in b:
                            b["w"] = round(b["x2"] - b["x1"], 1)
                        if "h" not in b and "y2" in b and "y1" in b:
                            b["h"] = round(b["y2"] - b["y1"], 1)
                    if "crop_url" not in det_copy:
                        det_copy["crop_url"] = f"/api/crop/{det_copy.get('detection_id')}"
                    detections.append(det_copy)
            return {
                "mission_id": mission_id,
                "total_frames": data.get("total_frames", 0),
                "total_detections": len(detections),
                "timestamp": data.get("export_timestamp", ""),
                "detections": detections,
                "count": len(detections),
                "frames": [f.get("frame_id", "") for f in data.get("frames", [])],
            }
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")


@app.post("/api/detect")
async def run_detection(
    raw_request: Request,
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
):
    """Run detection on a single image (via file upload or image_path)."""
    from inference.detector import CandidateDetector
    from inference.crop_extractor import CropExtractor
    from inference.verifier import CandidateVerifier
    from inference.evidence_fusion import EvidenceFusion
    from metadata.alignment import MetadataAligner
    from scoring.features import AcousticFeatureExtractor
    from metadata.schema import FrameMetadata

    detector = CandidateDetector(
        detector_config=_CONFIG.get("detector", {}),
        preprocessing_config=_CONFIG.get("preprocessing", {}),
    )
    verifier = CandidateVerifier(
        verifier_config=_CONFIG.get("verifier", {}),
    )
    crop_dir = OUTPUT_DIR / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_extractor = CropExtractor(crop_size=128, bbox_padding=0.15, save_dir=str(crop_dir))
    fusion = EvidenceFusion(scoring_config=_CONFIG.get("scoring", {}))

    image = None
    resolved_path = None
    image_url = None

    # 1. Check if multipart file upload
    if file is not None and file.filename:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        upload_dir = OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"det_{uuid.uuid4().hex[:8]}_{file.filename}"
        saved_path = upload_dir / safe_name
        with open(saved_path, "wb") as f_out:
            f_out.write(contents)
        resolved_path = str(saved_path)
        image_url = f"/api/uploads/{safe_name}"

    # 2. Check if Form image_path or JSON image_path
    if image is None:
        target_path = image_path
        content_type = raw_request.headers.get("content-type", "")
        if not target_path and "application/json" in content_type:
            try:
                body = await raw_request.json()
                target_path = body.get("image_path")
            except Exception:
                pass
        if target_path and Path(target_path).exists():
            image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            resolved_path = target_path

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="No valid image provided. Upload an image file or provide a valid 'image_path'.",
        )

    frame_f32 = image.astype(np.float32) / 255.0
    detections, _, _, _ = detector.detect(image, frame_id=Path(resolved_path).stem if resolved_path else "single")

    results = []
    run_prefix = uuid.uuid4().hex[:6]

    for i, det in enumerate(detections):
        det_id = f"det_{run_prefix}_{i+1:03d}"
        crop_tensor, crop_path = crop_extractor.extract(frame_f32, det.bbox, detection_id=det_id)
        verification = verifier.verify(crop_tensor)

        features = AcousticFeatureExtractor().extract(frame_f32, det.bbox)
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

        w = round(det.bbox.x2 - det.bbox.x1, 1)
        h = round(det.bbox.y2 - det.bbox.y1, 1)
        is_accepted = verification.predicted_class == "artificial"

        results.append({
            "detection_id": det_id,
            "bbox": {
                "x1": round(det.bbox.x1, 1),
                "y1": round(det.bbox.y1, 1),
                "x2": round(det.bbox.x2, 1),
                "y2": round(det.bbox.y2, 1),
                "w": w,
                "h": h,
            },
            "detector_confidence": round(det.detector_confidence, 3),
            "classification": det.class_name,
            "detector_class": det.class_name,
            "verifier_class": verification.predicted_class,
            "verifier_status": "accepted" if is_accepted else "rejected",
            "artificial_probability": round(verification.artificial_probability, 3),
            "natural_probability": round(verification.natural_probability, 3),
            "artificialness_score": round(artificialness, 3),
            "priority_score": round(priority, 3),
            "status": "verified" if is_accepted else "rejected",
            "image_id": Path(resolved_path).name if resolved_path else "uploaded_frame",
            "crop_path": crop_path,
            "crop_url": f"/api/crop/{det_id}",
        })

    return {
        "image_path": resolved_path,
        "image_url": image_url,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "detections": results,
        "count": len(results),
        "verified_count": sum(1 for d in results if d["verifier_status"] == "accepted"),
    }


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
