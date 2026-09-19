"""
dashboard/components/filters.py — Tactical Sidebar Filter Controls.

Provides mission selectors, threshold sliders, and status filters.
"""
import streamlit as st


def render_filters(data: dict) -> dict:
    """Render the sidebar filter controls and return the active filter state."""
    st.sidebar.markdown("### 🎛️ Anomaly Thresholds & Filters")

    min_art = st.sidebar.slider(
        "Min Artificialness Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter out detections with low evidence of being man-made.",
    )

    min_priority = st.sidebar.slider(
        "Min Operator Priority",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter by urgency of operator review.",
    )

    min_conf = st.sidebar.slider(
        "Min Detector Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter raw Stage-1 detector confidence.",
    )

    status_options = ["verified", "candidate", "rejected"]
    selected_statuses = st.sidebar.multiselect(
        "Triage Lifecycle State",
        status_options,
        default=status_options,
        help="Filter detections by operational verification state.",
    )

    hide_empty = st.sidebar.checkbox(
        "Hide Clean Frames (0 Detections)",
        value=False,
        help="Collapse swath frames where no anomalies were detected.",
    )

    # Telemetry Quick Readout
    st.sidebar.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📡 Sonar Hardware Telemetry")

    frames = data.get("frames", [])
    if frames:
        sample_f = frames[0]
        st.sidebar.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8; background: #0A101D; padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                <div>FREQUENCY: <strong style="color: #00F0FF;">{sample_f.get('frequency_khz', 455)} kHz</strong></div>
                <div>SWATH RANGE: <strong style="color: #00F0FF;">{sample_f.get('range_m', 75.0)} m</strong></div>
                <div>AVG DEPTH: <strong style="color: #10B981;">{sample_f.get('depth', 26.5)} m</strong></div>
                <div>AUV SPEED: <strong style="color: #10B981;">{sample_f.get('speed_knots', 3.2)} kts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "min_artificialness": min_art,
        "min_priority": min_priority,
        "min_confidence": min_conf,
        "selected_statuses": selected_statuses,
        "hide_empty": hide_empty,
    }
