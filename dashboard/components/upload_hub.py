"""
dashboard/components/upload_hub.py — Live Side-Scan Sonar Upload & Pipeline Execution Hub.

Enables operators to upload raw side-scan sonar images (single scans or batch
mission archives with navigation logs) and execute the complete two-stage AI
detection pipeline directly from the dashboard.
"""
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from inference.pipeline import MissionPipeline
from inference.types import MissionFrame
from mission.recorded import RecordedMissionSource
from reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


def get_cached_pipeline(config: dict, navigation_file: Optional[str] = None) -> MissionPipeline:
    """
    Get or build MissionPipeline.
    Uses st.cache_resource internally for persistent model weights.
    """
    return MissionPipeline.from_config(config, navigation_file=navigation_file)


def render_upload_hub(config: dict, on_complete: Optional[Callable] = None):
    """
    Render the live upload and ingestion hub.
    Allows single scan uploads or batch mission archives.
    """
    st.markdown("### 🚀 Live Side-Scan Sonar Upload & AI Pipeline Hub")
    st.markdown(
        """
        <div style="background: #0E172A; border-left: 4px solid #00F0FF; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 13px; color: #CBD5E1;">
            Upload raw side-scan sonar waterfall imagery directly into the NEMO platform. 
            The system executes the full end-to-end AI pipeline: 
            <strong>SonarPreprocessor → Stage-1 RCDI-YOLO → Crop Extractor → Stage-2 MobileNetV3 Verifier → Evidence Fusion → GIS Report</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active mission status chip if an uploaded mission is loaded
    if "last_processed_mission" in st.session_state:
        last = st.session_state["last_processed_mission"]
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #10B981; font-weight: 700;">● ACTIVE ANALYZED MISSION:</span> 
                    <strong>{last.get('mission_id')}</strong> ({last.get('total_detections', 0)} anomalies across {last.get('total_frames', 1)} frame(s))
                </div>
                <div style="font-family: monospace; font-size: 11px; color: #00F0FF;">
                    LOADED ACROSS ALL TABS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_single, tab_batch = st.tabs([
        "⚡ Single-Scan Rapid Ingest",
        "🚢 Batch AUV Mission Ingest (Multi-Image / ZIP + GPS)",
    ])

    # =========================================================================
    # TAB 1: SINGLE SCAN INGESTION
    # =========================================================================
    with tab_single:
        st.markdown("#### Analyze Single Side-Scan Sonar Waterfall Scan")

        up_col1, up_col2 = st.columns([6, 5])

        with up_col1:
            uploaded_img = st.file_uploader(
                "Upload Sonar Scan Image",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                key="single_sonar_uploader",
                help="Accepts high-frequency acoustic intensity side-scan sonar image files.",
            )

            if uploaded_img is not None:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                uploaded_img.seek(0)  # Reset pointer

                if img is not None:
                    h, w = img.shape[:2]
                    st.image(
                        uploaded_img,
                        caption=f"Uploaded Scan: {uploaded_img.name} ({w}×{h} px, Grayscale Intensity)",
                        use_column_width=True,
                    )
                else:
                    st.error("Failed to decode acoustic image. Please upload a valid image file.")

        with up_col2:
            st.markdown("##### Mission & Acoustic Telemetry")

            scan_id = st.text_input(
                "Scan Identifier / Frame ID",
                value=f"SCAN_{datetime.now().strftime('%H%M%S')}",
            )

            c_meta1, c_meta2 = st.columns(2)
            with c_meta1:
                slant_range = st.number_input("Swath Range (m)", value=75.0, step=5.0)
                freq_khz = st.number_input("Frequency (kHz)", value=455.0, step=50.0)
                depth_m = st.number_input("Vehicle Depth (m)", value=25.0, step=1.0)
            with c_meta2:
                has_gps = st.checkbox("Attach GPS Georeference", value=True)
                lat = st.number_input("Latitude (°N)", value=9.2876, format="%.5f", disabled=not has_gps)
                lon = st.number_input("Longitude (°E)", value=79.1245, format="%.5f", disabled=not has_gps)
                heading = st.number_input("Heading (°)", value=90.0, step=1.0)

            st.markdown("##### Detection Threshold Overrides")
            conf_thresh = st.slider(
                "Detector Confidence Threshold",
                min_value=0.01,
                max_value=0.95,
                value=float(config.get("detector", {}).get("confidence_threshold", 0.25)),
                step=0.01,
                help="Lower to detect smaller or subtle acoustic anomalies (e.g. 0.05 when running with initialized weights).",
            )

            execute_btn = st.button(
                "⚡ Execute Live AI Analysis",
                type="primary",
                use_container_width=True,
                disabled=(uploaded_img is None),
            )

        if execute_btn and uploaded_img is not None:
            with st.spinner("Executing two-stage AI detection & evidence fusion..."):
                t_start = time.perf_counter()

                # 1. Prepare temporary session directory
                temp_dir = Path("outputs/uploads") / f"single_{scan_id}"
                sonar_dir = temp_dir / "sonar"
                sonar_dir.mkdir(parents=True, exist_ok=True)

                saved_img_path = str(sonar_dir / f"{scan_id}.png")
                # Save image to disk for pipeline
                cv2.imwrite(saved_img_path, img)

                # Save navigation CSV if GPS provided
                nav_path = None
                if has_gps:
                    nav_path = str(temp_dir / "navigation.csv")
                    with open(nav_path, "w") as f:
                        f.write("timestamp,latitude,longitude,heading,depth,altitude,speed\n")
                        f.write(f"1000.0,{lat},{lon},{heading},{depth_m},5.0,3.0\n")

                # 2. Build pipeline with config overrides
                active_cfg = dict(config)
                active_cfg["detector"] = dict(config.get("detector", {}))
                active_cfg["detector"]["confidence_threshold"] = conf_thresh

                pipeline = get_cached_pipeline(active_cfg, navigation_file=nav_path)

                # 3. Process frame
                frame = MissionFrame(
                    mission_id=f"MISSION_{scan_id}",
                    frame_id=scan_id,
                    sonar_path=saved_img_path,
                    image_path=saved_img_path,
                    timestamp=1000.0 if has_gps else None,
                    latitude=lat if has_gps else None,
                    longitude=lon if has_gps else None,
                    heading=heading if has_gps else None,
                )

                mission_result = pipeline.process(frame)
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                # 4. Generate official reports
                reporter = ReportGenerator(active_cfg.get("output", {}))
                report_files = reporter.generate([mission_result], mission_id=f"SINGLE_{scan_id}")

                # 5. Format mission dictionary for UI
                formatted_dets = []
                for det in mission_result.detections:
                    formatted_dets.append({
                        "detection_id": det.detection_id,
                        "bbox": {
                            "x1": det.bbox.x1,
                            "y1": det.bbox.y1,
                            "x2": det.bbox.x2,
                            "y2": det.bbox.y2,
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
                        "label": f"Target ({det.verifier_class})",
                    })

                new_mission_data = {
                    "mission_id": f"UPLOAD-{scan_id}",
                    "survey_name": f"Ad-Hoc Sonar Scan: {uploaded_img.name}",
                    "vessel_auv": "User Uploaded Side-Scan Sonar",
                    "export_timestamp": datetime.now().isoformat(),
                    "total_frames": 1,
                    "total_detections": len(formatted_dets),
                    "frames": [{
                        "frame_id": scan_id,
                        "image_path": saved_img_path,
                        "latitude": lat if has_gps else None,
                        "longitude": lon if has_gps else None,
                        "heading": heading,
                        "depth": depth_m,
                        "range_m": slant_range,
                        "frequency_khz": freq_khz,
                        "detections": formatted_dets,
                        "_raw_targets": formatted_dets,
                    }],
                }

                # Push to session state
                session_key = f"mission_data_UPLOAD-{scan_id}"
                st.session_state[session_key] = new_mission_data
                st.session_state["active_mission_key"] = session_key
                st.session_state["last_processed_mission"] = new_mission_data

                if on_complete:
                    on_complete(new_mission_data)

    # =========================================================================
    # TAB 2: BATCH MISSION INGESTION
    # =========================================================================
    with tab_batch:
        st.markdown("#### Ingest Multi-Frame AUV Mission (Images + Navigation CSV or ZIP Archive)")

        b_col1, b_col2 = st.columns([6, 5])

        with b_col1:
            intake_mode = st.radio(
                "Batch Intake Mode",
                ["Upload Multiple Sonar Images + Nav CSV", "Upload Mission ZIP Archive"],
                horizontal=True,
            )

            multi_files = None
            zip_file = None
            nav_file_upload = None

            if intake_mode == "Upload Multiple Sonar Images + Nav CSV":
                multi_files = st.file_uploader(
                    "Select Sonar Waterfall Images",
                    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                    accept_multiple_files=True,
                    key="batch_multi_uploader",
                )
                nav_file_upload = st.file_uploader(
                    "Optional Navigation Log (navigation.csv)",
                    type=["csv"],
                    key="batch_nav_uploader",
                    help="CSV containing timestamp, latitude, longitude, heading, depth.",
                )
            else:
                zip_file = st.file_uploader(
                    "Upload AUV Mission ZIP Archive",
                    type=["zip"],
                    key="batch_zip_uploader",
                    help="ZIP archive containing sonar/ directory with images and optional navigation.csv.",
                )

        with b_col2:
            st.markdown("##### Batch Mission Setup")
            batch_mission_id = st.text_input(
                "Batch Mission ID",
                value=f"AUV_SURVEY_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )
            vessel_name = st.text_input("Vessel / AUV System", value="Autonomous Underwater Vehicle (EdgeTech SSS)")

            batch_conf = st.slider(
                "Detector Confidence Cutoff",
                min_value=0.01,
                max_value=0.95,
                value=float(config.get("detector", {}).get("confidence_threshold", 0.25)),
                step=0.01,
                key="batch_conf_slider",
                help="Confidence threshold for batch detection.",
            )

            ready_to_process = (
                (multi_files and len(multi_files) > 0) or (zip_file is not None)
            )

            launch_batch_btn = st.button(
                "🚢 Execute Full Mission Ingestion",
                type="primary",
                use_container_width=True,
                disabled=not ready_to_process,
            )

        if launch_batch_btn and ready_to_process:
            with st.spinner("Preparing mission workspace and extracting assets..."):
                t_batch_start = time.perf_counter()

                # Staging directory
                batch_dir = Path("outputs/uploads") / batch_mission_id
                sonar_staging = batch_dir / "sonar"
                sonar_staging.mkdir(parents=True, exist_ok=True)

                image_paths = []

                # Handle Multi-file mode
                if multi_files:
                    for f in multi_files:
                        dest = sonar_staging / f.name
                        dest.write_bytes(f.read())
                        image_paths.append(str(dest))

                    if nav_file_upload:
                        nav_dest = batch_dir / "navigation.csv"
                        nav_dest.write_bytes(nav_file_upload.read())

                # Handle ZIP mode
                elif zip_file:
                    with zipfile.ZipFile(zip_file, "r") as z:
                        z.extractall(batch_dir)
                    # Find extracted images
                    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                    image_paths = [
                        str(p) for p in sorted(batch_dir.rglob("*"))
                        if p.suffix.lower() in exts
                    ]

                image_paths = sorted(image_paths)

                if not image_paths:
                    st.error("No sonar images discovered in the uploaded archive. Please check folder contents.")
                    return

                # Check for navigation file
                nav_csv_path = None
                for nav_name in ["navigation.csv", "nav.csv", "gps.csv"]:
                    p = batch_dir / nav_name
                    if p.exists():
                        nav_csv_path = str(p)
                        break

            # Process with live progress bar
            progress_bar = st.progress(0, text="Initializing dual-stage AI detection pipeline...")

            active_cfg = dict(config)
            active_cfg["detector"] = dict(config.get("detector", {}))
            active_cfg["detector"]["confidence_threshold"] = batch_conf

            pipeline = get_cached_pipeline(active_cfg, navigation_file=nav_csv_path)
            source = RecordedMissionSource(str(batch_dir))

            total_frames = len(source)
            processed_results = []

            for idx, frame in enumerate(source):
                res = pipeline.process(frame)
                processed_results.append(res)
                progress_val = int(((idx + 1) / total_frames) * 100)
                progress_bar.progress(
                    progress_val,
                    text=f"Analyzing Swath Frame {idx + 1} of {total_frames} ({frame.frame_id})...",
                )

            # Export official reports
            reporter = ReportGenerator(active_cfg.get("output", {}))
            report_paths = reporter.generate(processed_results, mission_id=batch_mission_id)

            total_dets = sum(len(r.detections) for r in processed_results)
            total_elapsed = time.perf_counter() - t_batch_start

            # Read exported JSON for the dashboard
            json_report_path = report_paths.get("json")
            if json_report_path and os.path.exists(json_report_path):
                with open(json_report_path) as jf:
                    batch_mission_data = json.load(jf)
            else:
                batch_mission_data = {
                    "mission_id": batch_mission_id,
                    "survey_name": f"Batch Survey: {batch_mission_id}",
                    "total_frames": total_frames,
                    "total_detections": total_dets,
                    "frames": [],
                }

            batch_mission_data["vessel_auv"] = vessel_name

            # Push to session state
            session_key = f"mission_data_{batch_mission_id}"
            st.session_state[session_key] = batch_mission_data
            st.session_state["active_mission_key"] = session_key
            st.session_state["last_processed_mission"] = batch_mission_data

            if on_complete:
                on_complete(batch_mission_data)
