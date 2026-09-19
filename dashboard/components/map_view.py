"""
dashboard/components/map_view.py — Tactical Geospatial Oceanographic Map.

Renders high-resolution interactive Folium / Leaflet maps with AUV survey
trajectories, swath corridors, and geo-referenced acoustic anomalies.
"""
from typing import List, Optional

import pandas as pd
import streamlit as st


def render_map_view(detections: List[dict], frames: Optional[List[dict]] = None):
    """Render the master tactical geospatial oceanographic map tab."""
    st.markdown("### 🗺️ AUV Trajectory & Acoustic Anomaly Geospatial Plot")

    frames = frames or []

    # Collect GPS-tagged detections
    geo_dets = [
        d for d in detections
        if d.get("latitude") is not None and d.get("longitude") is not None
    ]

    # Collect AUV trackline from frames
    track_coords = []
    for f in frames:
        lat = f.get("latitude")
        lon = f.get("longitude")
        if lat is not None and lon is not None:
            track_coords.append([float(lat), float(lon)])

    if not geo_dets and not track_coords:
        st.info(
            "No GPS navigation coordinates available in this mission report.\n\n"
            "To enable geospatial mapping, ensure your mission directory contains `navigation.csv` "
            "with `latitude` and `longitude` fields."
        )
        return

    # Map settings bar
    map_ctrl1, map_ctrl2, map_ctrl3 = st.columns([2, 2, 3])
    with map_ctrl1:
        basemap_choice = st.selectbox(
            "Oceanic Basemap",
            ["CartoDB dark_matter", "OpenStreetMap", "Esri WorldImagery Satellite"],
            index=0,
        )
    with map_ctrl2:
        show_trackline = st.checkbox("Show AUV Trajectory Polyline", value=True)
    with map_ctrl3:
        st.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94A3B8; margin-top: 24px;">
                GEO-REFERENCED ANOMALIES: <strong style="color: #00F0FF;">{len(geo_dets)}</strong> | TRACK WAYPOINTS: <strong style="color: #10B981;">{len(track_coords)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        import folium
        from streamlit_folium import st_folium

        # Determine center coordinates
        if geo_dets:
            center_lat = sum(float(d["latitude"]) for d in geo_dets) / len(geo_dets)
            center_lon = sum(float(d["longitude"]) for d in geo_dets) / len(geo_dets)
        elif track_coords:
            center_lat = sum(c[0] for c in track_coords) / len(track_coords)
            center_lon = sum(c[1] for c in track_coords) / len(track_coords)
        else:
            center_lat, center_lon = 0.0, 0.0

        # Create Folium Map
        if basemap_choice == "Esri WorldImagery Satellite":
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri WorldImagery",
            )
        elif basemap_choice == "CartoDB dark_matter":
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles="CartoDB dark_matter",
            )
        else:
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
            )

        # 1. Plot AUV Trajectory
        if show_trackline and len(track_coords) > 1:
            folium.PolyLine(
                track_coords,
                color="#00F0FF",
                weight=3,
                opacity=0.85,
                dash_array="5, 8",
                tooltip="AUV Swath Survey Path",
            ).add_to(m)

            # Start waypoint
            folium.CircleMarker(
                location=track_coords[0],
                radius=6,
                color="#10B981",
                fill=True,
                fill_color="#10B981",
                fill_opacity=0.9,
                tooltip="Mission Ingress (Start)",
            ).add_to(m)

            # End waypoint
            folium.CircleMarker(
                location=track_coords[-1],
                radius=6,
                color="#00F0FF",
                fill=True,
                fill_color="#00F0FF",
                fill_opacity=0.9,
                tooltip="AUV Current / Egress Waypoint",
            ).add_to(m)

        # 2. Plot Anomaly Targets
        for d in geo_dets:
            status = str(d.get("status", "candidate")).lower()
            art_score = float(d.get("artificialness_score", 0))
            priority = float(d.get("priority_score", 0))

            if "verified" in status:
                color = "#10B981"
                fill_color = "#34D399"
                icon_sym = "exclamation-circle"
            elif "rejected" in status:
                color = "#F43F5E"
                fill_color = "#FB7185"
                icon_sym = "times-circle"
            else:
                color = "#F59E0B"
                fill_color = "#FBBF24"
                icon_sym = "question-circle"

            det_label = d.get("label", d.get("detector_class", "Target"))
            det_id = d.get("detection_id", "Unknown")

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 220px; color: #0F172A;">
                <div style="font-weight: 700; font-size: 14px; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 6px;">
                    {det_label}
                </div>
                <div style="font-size: 11px; color: #64748B; font-family: monospace;">ID: {det_id[:16]}</div>
                <div style="margin: 8px 0; font-size: 12px; line-height: 1.6;">
                    <div><strong>Status:</strong> <span style="color: {color}; font-weight: 700;">{status.upper()}</span></div>
                    <div><strong>Artificialness:</strong> {art_score:.3f}</div>
                    <div><strong>Priority Score:</strong> {priority:.3f}</div>
                    <div><strong>Detector Conf:</strong> {float(d.get('detector_confidence', 0)):.3f}</div>
                    <div><strong>Coords:</strong> {float(d['latitude']):.5f}°N, {float(d['longitude']):.5f}°E</div>
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[float(d["latitude"]), float(d["longitude"])],
                radius=9,
                color=color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{det_label} (Priority: {priority:.2f})",
            ).add_to(m)

        # Render Folium Map in Streamlit
        st_folium(m, width=None, height=540, use_container_width=True)

    except Exception as e:
        st.warning(f"Interactive Leaflet view fallback: {e}")

    # Coordinate Telemetry Table
    with st.expander("📍 Geospatial Target Waypoints Table", expanded=False):
        table_rows = []
        for d in geo_dets:
            table_rows.append({
                "Target Label": d.get("label", "Target"),
                "ID": d.get("detection_id", "")[:14],
                "Latitude": f"{float(d['latitude']):.5f}°",
                "Longitude": f"{float(d['longitude']):.5f}°",
                "Artificialness": round(float(d.get("artificialness_score", 0)), 3),
                "Priority": round(float(d.get("priority_score", 0)), 3),
                "Status": d.get("status", "candidate"),
            })
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
