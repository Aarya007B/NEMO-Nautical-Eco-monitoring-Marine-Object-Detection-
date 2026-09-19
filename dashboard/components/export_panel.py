"""
dashboard/components/export_panel.py — Mission Audit & GIS Export Center.

Generates QGIS/ArcGIS-compatible GeoJSON, mission audit spreadsheets (CSV),
complete mission JSONs, and printable mission briefing certificates.
"""
import json
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st


def generate_geojson(detections: List[dict], mission_id: str) -> str:
    """Generate RFC 7946 compliant GeoJSON FeatureCollection."""
    features = []
    for d in detections:
        lat = d.get("latitude")
        lon = d.get("longitude")
        if lat is not None and lon is not None:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": {
                    "detection_id": d.get("detection_id"),
                    "mission_id": mission_id,
                    "label": d.get("label", d.get("detector_class", "Target")),
                    "status": d.get("status", "candidate"),
                    "artificialness_score": float(d.get("artificialness_score", 0)),
                    "priority_score": float(d.get("priority_score", 0)),
                    "detector_confidence": float(d.get("detector_confidence", 0)),
                    "verifier_probability": float(d.get("artificial_probability", 0)),
                    "timestamp": d.get("timestamp"),
                },
            })

    geojson_obj = {
        "type": "FeatureCollection",
        "name": f"NEMO_{mission_id}_Debris_Anomalies",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    return json.dumps(geojson_obj, indent=2)


def render_export_panel(data: dict, filtered_detections: List[dict]):
    """Render the master mission export and audit tab."""
    st.markdown("### 📤 Mission Audit Briefing & Multi-Format GIS Export")

    mission_id = data.get("mission_id", "NEMO-MISSION")
    all_dets = filtered_detections

    # 1. Executive Briefing Card
    verified_count = sum(1 for d in all_dets if d.get("status") == "verified")
    high_priority = sum(1 for d in all_dets if float(d.get("priority_score", 0)) >= 0.7)

    st.markdown(
        f"""
        <div style="background: #0E1626; border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 10px; padding: 20px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="margin: 0; color: #FFFFFF; font-size: 20px;">MISSION AUDIT RECORD: {mission_id}</h3>
                    <div style="color: #00F0FF; font-size: 12px; font-family: monospace; margin-top: 4px;">
                        SURVEY VESSEL / PAYLOAD: {data.get('vessel_auv', 'AUV Bluefin-21 (EdgeTech 4205)')}
                    </div>
                </div>
                <div style="text-align: right; font-family: monospace; font-size: 11px; color: #94A3B8;">
                    GENERATED: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 14px 0;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; text-align: center;">
                <div>
                    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Total Targets</div>
                    <div style="font-size: 24px; font-weight: 700; color: #F8FAFC;">{len(all_dets)}</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Confirmed Debris</div>
                    <div style="font-size: 24px; font-weight: 700; color: #10B981;">{verified_count}</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Urgent ROV Targets</div>
                    <div style="font-size: 24px; font-weight: 700; color: #EF4444;">{high_priority}</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Acoustic Swath Area</div>
                    <div style="font-size: 24px; font-weight: 700; color: #00F0FF;">{data.get('survey_area_sq_km', '1.4')} km²</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Export Download Buttons
    st.markdown("#### 💾 Export Package Generation")

    d_col1, d_col2, d_col3 = st.columns(3)

    # GeoJSON
    geojson_str = generate_geojson(all_dets, mission_id)
    with d_col1:
        st.download_button(
            "🌍 Download QGIS / GIS GeoJSON",
            data=geojson_str,
            file_name=f"nemo_{mission_id}_targets.geojson",
            mime="application/geo+json",
            use_container_width=True,
            help="Direct vector import for QGIS, ArcGIS, and marine bathymetric navigation chart plotters.",
        )

    # CSV
    csv_rows = []
    for d in all_dets:
        b = d.get("bbox", {})
        csv_rows.append({
            "detection_id": d.get("detection_id"),
            "mission_id": mission_id,
            "label": d.get("label", "target"),
            "status": d.get("status"),
            "detector_confidence": d.get("detector_confidence"),
            "artificial_probability": d.get("artificial_probability"),
            "artificialness_score": d.get("artificialness_score"),
            "priority_score": d.get("priority_score"),
            "latitude": d.get("latitude"),
            "longitude": d.get("longitude"),
            "timestamp": d.get("timestamp"),
            "bbox_x1": b.get("x1"),
            "bbox_y1": b.get("y1"),
            "bbox_x2": b.get("x2"),
            "bbox_y2": b.get("y2"),
        })
    csv_str = pd.DataFrame(csv_rows).to_csv(index=False) if csv_rows else ""
    with d_col2:
        st.download_button(
            "📊 Download Mission CSV",
            data=csv_str,
            file_name=f"nemo_{mission_id}_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
            help="Comma-separated spreadsheet of all cataloged detections.",
        )

    # JSON
    full_json_str = json.dumps(data, indent=2)
    with d_col3:
        st.download_button(
            "📄 Download Full Mission JSON",
            data=full_json_str,
            file_name=f"nemo_{mission_id}_full_telemetry.json",
            mime="application/json",
            use_container_width=True,
            help="Comprehensive machine-readable telemetry and pipeline audit schema.",
        )

    # 3. Anomaly Summary Table Preview
    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("#### 📋 High-Priority ROV Intervention Queue Preview")

    urgent_dets = [d for d in all_dets if float(d.get("priority_score", 0)) >= 0.7]
    if urgent_dets:
        st.dataframe(pd.DataFrame([
            {
                "Target ID": d.get("detection_id", "")[:14],
                "Classification": d.get("label", "Target"),
                "Priority Score": round(float(d.get("priority_score", 0)), 3),
                "Artificialness Score": round(float(d.get("artificialness_score", 0)), 3),
                "Latitude": d.get("latitude"),
                "Longitude": d.get("longitude"),
                "Action Required": "Dispatch ROV / Diver Tagging",
            }
            for d in urgent_dets
        ]), use_container_width=True)
    else:
        st.success("No critical high-priority anomalies requiring immediate intervention.")
