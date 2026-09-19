"""
dashboard/components/detection_panel.py — Tactical Anomaly Catalog & Operator Triage.

Allows operators to review, filter, inspect, and update detection statuses
(Confirm Debris, Mark Natural, Mine as Hard Negative) with immediate state sync.
"""
from typing import List

import pandas as pd
import streamlit as st

from dashboard.components.theme import get_status_badge


def render_detection_panel(detections: List[dict]):
    """Render the master anomaly triage catalog tab."""
    st.markdown("### 🎯 Anomaly Catalog & Tactical Operator Triage")

    if not detections:
        st.info("No anomalies found matching current filters.")
        return

    # Operator summary stats
    total = len(detections)
    verified = sum(1 for d in detections if d.get("status") == "verified")
    rejected = sum(1 for d in detections if d.get("status") == "rejected")
    candidate = total - verified - rejected

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Cataloged Anomalies", total)
    with stat_col2:
        st.metric("Confirmed Debris", verified, f"{(verified/max(total,1))*100:.1f}%")
    with stat_col3:
        st.metric("Natural Seabed", rejected)
    with stat_col4:
        st.metric("Pending Review", candidate)

    st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # Grid search and sort controls
    f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
    with f_col1:
        search_query = st.text_input("Search Anomalies by Label or ID", "", placeholder="e.g. Ghost Net, Container, DET01...")
    with f_col2:
        sort_by = st.selectbox("Sort Anomaly Order", ["Priority Score (High to Low)", "Artificialness Score", "Detector Confidence", "Deepest Depth"])
    with f_col3:
        status_filter_local = st.selectbox("Triage Status Filter", ["All Statuses", "verified", "candidate", "rejected"])

    # Filter detections
    filtered_dets = detections
    if search_query:
        q = search_query.lower()
        filtered_dets = [
            d for d in filtered_dets
            if q in d.get("label", "").lower() or q in d.get("detection_id", "").lower()
        ]
    if status_filter_local != "All Statuses":
        filtered_dets = [d for d in filtered_dets if d.get("status") == status_filter_local]

    # Sort
    if "Priority" in sort_by:
        filtered_dets.sort(key=lambda x: float(x.get("priority_score", 0)), reverse=True)
    elif "Artificialness" in sort_by:
        filtered_dets.sort(key=lambda x: float(x.get("artificialness_score", 0)), reverse=True)
    elif "Detector" in sort_by:
        filtered_dets.sort(key=lambda x: float(x.get("detector_confidence", 0)), reverse=True)

    # Split: Left = Interactive Table, Right = Active Triage Action Box
    table_col, action_col = st.columns([7, 5])

    with table_col:
        st.markdown("#### Anomaly Inventory")

        table_data = []
        for i, d in enumerate(filtered_dets):
            table_data.append({
                "#": i + 1,
                "ID": d.get("detection_id", "")[:14],
                "Classification": d.get("label", d.get("detector_class", "Target")),
                "Detector Conf": f"{float(d.get('detector_confidence', 0)):.3f}",
                "Verifier Prob": f"{float(d.get('artificial_probability', 0)):.3f}",
                "Artificialness": f"{float(d.get('artificialness_score', 0)):.3f}",
                "Priority": f"{float(d.get('priority_score', 0)):.3f}",
                "Status": str(d.get("status", "candidate")).upper(),
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, height=420)

    with action_col:
        st.markdown("#### ⚡ Operator Triage Desk")

        if not filtered_dets:
            st.caption("No anomaly selected.")
            return

        selected_det_name = st.selectbox(
            "Select Target to Update",
            range(len(filtered_dets)),
            format_func=lambda i: f"#{i+1}: {filtered_dets[i].get('label', 'Target')} (Priority: {float(filtered_dets[i].get('priority_score', 0)):.2f})",
        )

        active_det = filtered_dets[selected_det_name]
        det_id = active_det.get("detection_id")

        # Telemetry Card
        st.markdown(
            f"""
            <div style="background: #0E172A; border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 14px; margin-bottom: 14px;">
                <div style="font-weight: 700; font-size: 15px; color: #FFFFFF;">
                    {active_det.get('label', 'Acoustic Anomaly')}
                </div>
                <div style="font-size: 11px; color: #94A3B8; font-family: monospace; margin-top: 2px;">
                    ID: {det_id}
                </div>
                <div style="margin-top: 10px;">
                    Current Status: {get_status_badge(active_det.get('status', 'candidate'))}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; font-size: 12px;">
                    <div>Stage-1 Conf: <strong>{float(active_det.get('detector_confidence',0)):.3f}</strong></div>
                    <div>Stage-2 Prob: <strong>{float(active_det.get('artificial_probability',0)):.3f}</strong></div>
                    <div>Acoustic Contrast: <strong>{float(active_det.get('acoustic_contrast',0.75)):.2f}</strong></div>
                    <div>Seabed Depth: <strong>{active_det.get('depth_m', '25.0')}m</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Execute Operator Determination:")
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if st.button("✅ Confirm Debris", use_container_width=True):
                active_det["status"] = "verified"
                st.success(f"Marked {det_id[:12]} as VERIFIED DEBRIS.")
                st.rerun()

        with act_c2:
            if st.button("✕ Natural Feature", use_container_width=True):
                active_det["status"] = "rejected"
                st.warning(f"Marked {det_id[:12]} as NATURAL SEABED.")
                st.rerun()

        act_c3, act_c4 = st.columns(2)
        with act_c3:
            if st.button("⚠ Mine Hard Negative", use_container_width=True, help="Flag this crop for retraining the MobileNetV3 verifier."):
                active_det["status"] = "rejected"
                active_det["hard_negative_mined"] = True
                st.info(f"Flagged {det_id[:12]} for Hard Negative Mining.")
                st.rerun()

        with act_c4:
            if st.button("↺ Reset Status", use_container_width=True):
                active_det["status"] = "candidate"
                st.rerun()

    # Distribution Plots
    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Artificialness vs. Detector Confidence Correlation")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        try:
            import plotly.express as px
            scatter_df = pd.DataFrame([
                {
                    "Label": d.get("label", "Target"),
                    "Detector Confidence": float(d.get("detector_confidence", 0)),
                    "Artificialness Score": float(d.get("artificialness_score", 0)),
                    "Priority": float(d.get("priority_score", 0)),
                    "Status": str(d.get("status", "candidate")).upper(),
                }
                for d in filtered_dets
            ])
            fig = px.scatter(
                scatter_df,
                x="Detector Confidence",
                y="Artificialness Score",
                color="Status",
                size="Priority",
                hover_data=["Label"],
                color_discrete_map={
                    "VERIFIED": "#10B981",
                    "CANDIDATE": "#F59E0B",
                    "REJECTED": "#F43F5E",
                },
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#0E1626",
                plot_bgcolor="#0A111E",
                font_family="Inter",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.caption("Install plotly for interactive scatter plots.")

    with chart_col2:
        try:
            import plotly.express as px
            hist_df = pd.DataFrame([
                {"Priority Score": float(d.get("priority_score", 0)), "Status": str(d.get("status", "candidate")).upper()}
                for d in filtered_dets
            ])
            fig_hist = px.histogram(
                hist_df,
                x="Priority Score",
                color="Status",
                nbins=10,
                color_discrete_map={
                    "VERIFIED": "#10B981",
                    "CANDIDATE": "#F59E0B",
                    "REJECTED": "#F43F5E",
                },
                template="plotly_dark",
            )
            fig_hist.update_layout(
                paper_bgcolor="#0E1626",
                plot_bgcolor="#0A111E",
                font_family="Inter",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception:
            st.caption("Install plotly for distribution histograms.")
