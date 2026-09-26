# PRODUCT — NEMO Dashboard

## Product truth
NEMO is an AI-powered side-scan sonar marine debris detection system. The dashboard is the operational interface where users monitor missions, view detections, manage data exports, and track the AI pipeline's analysis.

## Who uses this
Marine operations personnel, researchers, and field engineers who need to process sonar missions and verify debris detections.

## What the visitor does
- **Process missions** by pointing the pipeline at a directory of sonar frames
- **View detections** with bounding boxes, confidence scores, and verification status
- **Inspect crops** of candidate objects
- **Export results** as JSON or CSV
- **Monitor system health** and model status
- **Access demo data** for testing without real missions

## Key constraint
The backend is a FastAPI REST server at `localhost:8000`. The frontend is a single HTML file with vanilla JS — no build step, no framework. All data comes from API calls.

## Mental model
A dark monitoring station: instruments reading sonar data, status indicators, mission queues, and detection panels. Clinical, precise, oceanic.
