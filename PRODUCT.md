# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: public demo visitors seeing NEMO for the first time — they must grasp what NEMO does, watch the pipeline work a real mission, and inspect detections within seconds. Secondary: marine operations personnel and researchers who process sonar missions and verify debris detections.

## Product Purpose

NEMO is an AI-powered side-scan sonar marine debris detection system. The dashboard is the operational interface where users run missions through a two-model pipeline (YOLO11n candidate detection + MobileNetV3 verification), view detections with evidence, and export reviewable reports. Success for a demo visitor means understanding the mechanism and seeing real results; success for an operator means turning sonar frames into location-aware evidence for inspection and cleanup decisions.

## Positioning

A two-stage detect-then-verify pipeline tuned for side-scan sonar: YOLO11n proposes candidates, MobileNetV3 rejects seabed false positives, and every surviving detection ships with bounding box, confidence, verifier score, crop evidence, and geotag. No generic object-detection demo can truthfully show that chain on sonar data.

## Operating Context

Dim operations-room usage is the design scene; visitors evaluate the product in a browser against a live backend. Core workflows: process a mission directory or upload a mission video, watch pipeline stages complete, inspect detections on the sonar canvas with overlay toggles, locate geotagged detections on the map, filter the anomaly report table, export JSON/CSV.

## Capabilities and Constraints

Confirmed functionality: mission list/select, mission-directory processing, video upload with background job polling, single-image detection via file upload, sonar canvas with candidate/verified/label/rejected overlays and detection drawer, Leaflet geospatial view, filterable anomaly reports table, JSON/CSV export, system status and evaluation metrics, about page.
Constraints: FastAPI REST backend is the system of record — frontend requires the live API and states backend connection honestly (no silent demo fallback; sample mission is an explicit user action). Frontend is a single `dashboard/index.html` with vanilla JS — no build step, no framework. Leaflet via CDN for maps.
Undecided: live-vehicle integration and field validation remain future deployment stages.

## Brand Commitments

Name: NEMO. Committed aesthetic (user-pinned): deep-sea ops console — a dark, cinematic sonar-command room, phosphor-on-abyss instrumentation, not a light lab instrument. Prototype version label `v0.1.0-mdp` and "recorded mission / prototype, not for operational deployment" honesty notice must survive the redesign.

## Evidence on Hand

Real assets: validation-set side-scan sonar imagery served via `GET /api/sample-image`; trained YOLO11n + MobileNetV3 models with preliminary T4 evaluation metrics (precision 0.7377, F1 0.7965 with verifier); 110 passing backend tests. No invented customers, prices, benchmarks, or capabilities. Synthetic sonar-canvas rendering is labeled as illustration where it stands in for unprocessed frames.

## Product Principles

1. Backend truth first — every number on screen traces to an API response or an explicit sample action; connection state is always visible and honest.
2. Evidence over claims — detections ship with boxes, scores, crops, and coordinates, never bare counts.
3. Prototype honesty — recorded-mission and preliminary-evaluation labels stay on screen; nothing poses as operational or field-validated.
4. Operate at a glance — status, pipeline stage, and next action are readable within seconds on every screen.
5. One room, five stations — overview, analysis, map, reports, and system read as stations of a single console, not five pages.

## Accessibility & Inclusion

Keyboard-operable nav and controls with visible focus; status never carried by color alone (labels accompany dots); `prefers-reduced-motion` disables sweep and transitions. No product-specific standard beyond WCAG-minded contrast (body ≥4.5:1).
