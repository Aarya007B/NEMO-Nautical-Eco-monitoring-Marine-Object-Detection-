"""
dashboard/app.py — NEMO Mission Operations Deck (Streamlit).

The primary operator interface for the SonarGuard / NEMO underwater
marine debris and acoustic anomaly detection system.

Features:
- Live side-scan sonar image upload & pipeline execution
- Batch mission ingestion (multi-image/ZIP with navigation GPS)
- Side-scan sonar waterfall viewer with acoustic colormaps
- Anomaly catalog with live operator triage
- Geospatial tactical mapping of AUV tracklines and anomalies
- Dual-stage model fusion analytics and GIS report exports
"""
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import streamlit as st
import yaml

from dashboard.components.theme import inject_custom_css, render_header, metric_card
from dashboard.components.demo_data import get_demo_mission
from dashboard.components.filters import render_filters
from dashboard.components.sonar_viewer import render_sonar_viewer
from dashboard.components.detection_panel import render_detection_panel
from dashboard.components.map_view import render_map_view
from dashboard.components.analytics import render_analytics
from dashboard.components.export_panel import render_export_panel
from dashboard.components.upload_hub import render_upload_hub

logger = logging.getLogger(__name__)


def load_project_config(config_path: str = "configs/config.yaml") -> dict:
    """Load config.yaml with fallback defaults."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file) as f:
            return yaml.safe_load(f)
    return {
        "runtime": {"device": "auto"},
        "preprocessing": {"image_size": 640, "normalize": True, "denoise": {"enabled": False}, "contrast": {"enabled": False}},
        "detector": {"config": "configs/rcdi_yolo_1c.yaml", "confidence_threshold": 0.25, "nms_iou": 0.70},
        "verifier": {"crop_size": 128, "bbox_padding": 0.15, "confidence_threshold": 0.50},
        "scoring": {
            "artificialness": {"detector_weight": 0.35, "verifier_weight": 0.45, "contrast_weight": 0.10, "geometry_weight": 0.10},
            "priority": {"artificialness_weight": 0.70, "mission_confidence_weight": 0.20, "metadata_quality_weight": 0.10},
        },
        "output": {"directory": "outputs/", "save_crops": True, "save_json": True, "save_csv": True},
    }


def discover_reports(reports_dir: str = "outputs/reports") -> list:
    """Discover all mission JSON reports in outputs/reports."""
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        return []
    return sorted(reports_path.glob("mission_*.json"), reverse=True)


def load_mission_file(filepath: str) -> dict:
    """Load and parse a mission JSON file."""
    with open(filepath) as f:
        return json.load(f)


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="NEMO — Autonomous Marine Debris & Sonar Detection Deck",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject Oceanic Command Deck Theme
    inject_custom_css()

    # Load system configuration
    config = load_project_config()

    # 3. Sidebar: Mission Source & Selector
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 24px;">🌊</span>
                <div>
                    <strong style="color: #00F0FF; font-size: 16px; letter-spacing: 1px;">NEMO</strong>
                    <div style="font-size: 10px; color: #94A3B8; font-family: monospace;">SONARGUARD COMMAND DECK</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📂 Mission Source")
        source_type = st.radio(
            "Select Data Source",
            [
                "🚀 Upload & Ingest Raw Sonar Scan(s)",
                "🌟 Live Demonstration Mission",
                "📁 Archived Reports (outputs/reports/)",
                "📤 Import Pre-Computed JSON",
            ],
            index=0,
            label_visibility="collapsed",
        )

        raw_mission_data = None
        mission_status_badge = "MISSION ARCHIVE"

        if source_type == "🚀 Upload & Ingest Raw Sonar Scan(s)":
            mission_status_badge = "LIVE PIPELINE INGRESS"
            # If an active uploaded mission exists in session, use it
            active_key = st.session_state.get("active_mission_key", None)
            if active_key and active_key in st.session_state:
                raw_mission_data = st.session_state[active_key]
            else:
                # Default to demo data until scan is executed
                raw_mission_data = get_demo_mission("NEMO-INGEST-STANDBY")

        elif source_type == "🌟 Live Demonstration Mission":
            demo_choice = st.selectbox(
                "Select Scenario",
                [
                    "Gulf of Mannar — Ghost Net & Marine Debris Recon",
                    "Baltic Sea — Submerged Cargo & UXO Ordnance Survey",
                ],
            )
            demo_id = "NEMO-DEMO-MANNAR-01" if "Mannar" in demo_choice else "NEMO-DEMO-BALTIC-02"
            raw_mission_data = get_demo_mission(demo_id)
            mission_status_badge = "SIMULATED TELEMETRY"

        elif source_type == "📁 Archived Reports (outputs/reports/)":
            reports = discover_reports()
            if not reports:
                st.warning("No mission reports found in `outputs/reports/`.")
                st.caption("Using demo mission as fallback.")
                raw_mission_data = get_demo_mission("NEMO-FALLBACK-01")
                mission_status_badge = "DEMO FALLBACK"
            else:
                selected_report = st.selectbox(
                    "Select Report File",
                    reports,
                    format_func=lambda p: p.stem,
                )
                raw_mission_data = load_mission_file(str(selected_report))
                mission_status_badge = "ARCHIVED MISSION RECORD"

        else:
            uploaded_file = st.file_uploader("Upload Mission JSON", type=["json"])
            if uploaded_file:
                raw_mission_data = json.load(uploaded_file)
                mission_status_badge = "USER UPLOADED"
            else:
                st.info("Upload a mission report JSON to begin.")
                raw_mission_data = get_demo_mission("NEMO-UPLOAD-WAITING")
                mission_status_badge = "WAITING FOR UPLOAD"

    # Store in session state for persistent triage edits
    mission_id = raw_mission_data.get("mission_id", "NEMO-MISSION")
    session_key = f"mission_data_{mission_id}"
    if session_key not in st.session_state:
        st.session_state[session_key] = raw_mission_data

    active_mission = st.session_state[session_key]

    # Render Sidebar Filters
    with st.sidebar:
        filters = render_filters(active_mission)

    # 4. Header Banner
    render_header(mission_id=mission_id, status=mission_status_badge)

    # 5. Filter frames and detections
    raw_frames = active_mission.get("frames", [])
    filtered_frames = []

    for f in raw_frames:
        kept_dets = []
        for d in f.get("detections", []):
            art_score = float(d.get("artificialness_score", 0))
            priority = float(d.get("priority_score", 0))
            det_conf = float(d.get("detector_confidence", 0))
            status = str(d.get("status", "candidate")).lower()

            if art_score >= filters["min_artificialness"]:
                if priority >= filters["min_priority"]:
                    if det_conf >= filters["min_confidence"]:
                        if status in filters["selected_statuses"]:
                            kept_dets.append(d)

        if kept_dets or not filters["hide_empty"]:
            frame_copy = dict(f)
            frame_copy["detections"] = kept_dets
            filtered_frames.append(frame_copy)

    all_filtered_dets = [d for f in filtered_frames for d in f.get("detections", [])]

    # 6. Executive KPI Ribbon
    total_anomalies = len(all_filtered_dets)
    verified_debris = sum(1 for d in all_filtered_dets if d.get("status") == "verified")
    high_urgency = sum(1 for d in all_filtered_dets if float(d.get("priority_score", 0)) >= 0.7)
    avg_art = (
        sum(float(d.get("artificialness_score", 0)) for d in all_filtered_dets) / max(total_anomalies, 1)
        if total_anomalies > 0 else 0.0
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            metric_card("Total Swath Targets", f"{total_anomalies}", "Filtered Catalog", "#00F0FF"),
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            metric_card("Confirmed Debris", f"{verified_debris}", f"{(verified_debris/max(total_anomalies,1))*100:.0f}% of targets", "#10B981"),
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            metric_card("ROV Action Priority Queue", f"{high_urgency}", "Priority ≥ 0.70", "#EF4444"),
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            metric_card("Mean Artificialness Index", f"{avg_art:.2f}", "Dual-Stage Fusion", "#FFB703"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # 7. Navigation Tabs: Ingest Hub + 5 Tactical Views
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Upload & Ingest Scan(s)",
        "🔍 Sonar Waterfall View",
        "🎯 Anomaly Catalog & Triage",
        "🗺️ Tactical Geospatial Map",
        "📈 Model Fusion Analytics",
        "📤 Mission Audit & GIS Export",
    ])

    with tab0:
        def on_mission_created(new_mission):
            st.rerun()

        render_upload_hub(config, on_complete=on_mission_created)

    with tab1:
        render_sonar_viewer(filtered_frames)

    with tab2:
        render_detection_panel(all_filtered_dets)

    with tab3:
        render_map_view(all_filtered_dets, frames=filtered_frames)

    with tab4:
        render_analytics(all_filtered_dets)

    with tab5:
        render_export_panel(active_mission, all_filtered_dets)


if __name__ == "__main__":
    main()
