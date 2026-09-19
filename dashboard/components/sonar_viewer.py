"""
dashboard/components/sonar_viewer.py — Interactive Sonar Waterfall & Anomaly Inspector.

Renders raw side-scan sonar waterfall imagery with dynamic bounding boxes,
commercial acoustic colormaps (Copper, Phosphor, Viridis, Grayscale),
and target crop cross-section acoustic line scans.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from dashboard.components.demo_data import generate_synthetic_sonar_image


COLORMAPS = {
    "Acoustic Copper (Sonar standard)": "copper",
    "Amber Phosphor": "amber",
    "Scientific Viridis": "viridis",
    "Inferno / Specular Heatmap": "inferno",
    "Raw Grayscale Backscatter": "gray",
}


def apply_sonar_colormap(gray_img: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply specialized side-scan acoustic color palettes."""
    if colormap_name == "gray":
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    elif colormap_name == "copper":
        # Custom warm oceanic copper palette
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 1.25))
            g = min(255, int(i * 0.78))
            b = min(255, int(i * 0.45))
            lut[i, 0] = [b, g, r]  # BGR for cv2
        colored_bgr = cv2.LUT(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), lut)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    elif colormap_name == "amber":
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 1.1))
            g = min(255, int(i * 0.95))
            b = int(i * 0.1)
            lut[i, 0] = [b, g, r]
        colored_bgr = cv2.LUT(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), lut)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    elif colormap_name == "viridis":
        colored_bgr = cv2.applyColorMap(gray_img, cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    elif colormap_name == "inferno":
        colored_bgr = cv2.applyColorMap(gray_img, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)


def draw_bounding_boxes(
    rgb_img: np.ndarray,
    detections: List[dict],
    selected_det_id: Optional[str] = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw tactical detection bounding boxes and target tags on image."""
    img_pil = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(img_pil, "RGBA")

    for i, d in enumerate(detections):
        bbox = d.get("bbox", {})
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))
        x2 = float(bbox.get("x2", 0))
        y2 = float(bbox.get("y2", 0))

        status = str(d.get("status", "candidate")).lower()
        is_selected = (d.get("detection_id") == selected_det_id)

        # Color selection
        if "verified" in status:
            border_color = (16, 185, 129, 255)      # Emerald Green
            fill_color   = (16, 185, 129, 45)
        elif "rejected" in status:
            border_color = (244, 63, 94, 255)       # Rose Red
            fill_color   = (244, 63, 94, 35)
        else:
            border_color = (245, 158, 11, 255)      # Amber
            fill_color   = (245, 158, 11, 40)

        # Glow / extra border if actively selected
        line_width = 4 if is_selected else 2
        if is_selected:
            draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(0, 240, 255, 255), width=2)

        draw.rectangle([x1, y1, x2, y2], outline=border_color, fill=fill_color, width=line_width)

        # Tag label
        if show_labels:
            conf = d.get("detector_confidence", 0)
            art_score = d.get("artificialness_score", 0)
            tag_text = f"#{i+1} [Art: {art_score:.2f} | Conf: {conf:.2f}]"

            # Draw tag background
            text_w = len(tag_text) * 7.2
            tag_y = max(0, y1 - 18)
            draw.rectangle([x1, tag_y, x1 + text_w, tag_y + 16], fill=border_color)
            draw.text((x1 + 3, tag_y + 1), tag_text, fill=(10, 15, 25, 255))

    return np.array(img_pil)


def render_sonar_viewer(frames: List[dict]):
    """Render the master sonar waterfall inspection interface."""
    st.markdown("### 🔍 Side-Scan Sonar Waterfall Telemetry")

    if not frames:
        st.info("No frames match the active telemetry filters.")
        return

    # Frame selector and controls header
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])

    frame_options = [
        f"{f.get('frame_id', f'frame_{i}')} — {len(f.get('detections', []))} detections"
        for i, f in enumerate(frames)
    ]

    with ctrl_col1:
        selected_idx = st.selectbox(
            "Select Frame / Ping Swath",
            range(len(frames)),
            format_func=lambda i: frame_options[i],
            help="Select a sonar swath frame for tactical inspection.",
        )

    with ctrl_col2:
        cmap_label = st.selectbox(
            "Acoustic Colormap",
            list(COLORMAPS.keys()),
            index=0,
            help="Choose standard commercial side-scan acoustic color palettes.",
        )
        selected_cmap = COLORMAPS[cmap_label]

    with ctrl_col3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        show_boxes = st.checkbox("Show AI Bounding Boxes", value=True)
        enhance_contrast = st.checkbox("Dynamic CLAHE Contrast", value=False)

    frame = frames[selected_idx]
    detections = frame.get("detections", [])

    # Load or generate sonar image
    image_path = frame.get("image_path", "")
    sonar_img = None
    if image_path and Path(image_path).exists():
        sonar_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if sonar_img is None:
        # Fall back to synthetic high-fidelity simulation
        raw_tgts = frame.get("_raw_targets", detections)
        # Format bboxes for generator
        gen_tgts = []
        for t in raw_tgts:
            b = t.get("bbox", {})
            gen_tgts.append({"bbox": (b.get("x1", 100), b.get("y1", 100), b.get("x2", 200), b.get("y2", 200))})
        sonar_img = generate_synthetic_sonar_image(width=800, height=550, targets=gen_tgts, seed=selected_idx + 10)

    # Apply CLAHE if enabled
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        sonar_img = clahe.apply(sonar_img)

    # Colorize image
    color_img = apply_sonar_colormap(sonar_img, selected_cmap)

    # Selection state for detections
    selected_det_id = st.session_state.get("selected_detection_id", None)

    # Draw boxes
    if show_boxes and detections:
        annotated_img = draw_bounding_boxes(
            color_img, detections, selected_det_id=selected_det_id, show_labels=True
        )
    else:
        annotated_img = color_img

    # Main view layout: Left = Waterfall image, Right = Telemetry & Zoom Inspector
    view_col, inspect_col = st.columns([7, 4])

    with view_col:
        st.image(
            annotated_img,
            caption=f"Waterfall Frame: {frame.get('frame_id')} | Slant Range: {frame.get('range_m', 75)}m | Frequency: {frame.get('frequency_khz', 455)} kHz",
            use_column_width=True,
        )

        # Tactical legend
        st.markdown(
            """
            <div style="display: flex; gap: 16px; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin-top: 4px; color: #94A3B8;">
                <span><span style="color: #10B981;">■</span> Verified Man-Made Debris</span>
                <span><span style="color: #F59E0B;">■</span> Unconfirmed Candidate</span>
                <span><span style="color: #F43F5E;">■</span> Natural Seabed Structure</span>
                <span><span style="color: #00F0FF;">■</span> Selected Focus</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with inspect_col:
        st.markdown("#### 🎯 Target Crop & Acoustic Transect")

        if not detections:
            st.info("No anomalies detected in this swath frame.")
            # Swath telemetry
            st.markdown("##### Frame Telemetry")
            st.json({
                "frame_id": frame.get("frame_id"),
                "latitude": frame.get("latitude"),
                "longitude": frame.get("longitude"),
                "heading_deg": frame.get("heading"),
                "depth_m": frame.get("depth"),
            })
            return

        # Target selector
        det_names = [
            f"Anomaly #{i+1}: {d.get('label', d.get('detector_class', 'Target'))} ({d.get('status', 'candidate')})"
            for i, d in enumerate(detections)
        ]
        chosen_det_idx = st.selectbox(
            "Select Anomaly for Deep Inspection",
            range(len(detections)),
            format_func=lambda i: det_names[i],
        )

        curr_det = detections[chosen_det_idx]
        st.session_state["selected_detection_id"] = curr_det.get("detection_id")

        # Extract target crop
        b = curr_det.get("bbox", {})
        bx1, by1 = max(0, int(b.get("x1", 0))), max(0, int(b.get("y1", 0)))
        bx2, by2 = min(sonar_img.shape[1], int(b.get("x2", 100))), min(sonar_img.shape[0], int(b.get("y2", 100)))

        # Add 25% padding for acoustic context
        pad_x = int((bx2 - bx1) * 0.25)
        pad_y = int((by2 - by1) * 0.25)
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(sonar_img.shape[1], bx2 + pad_x)
        cy2 = min(sonar_img.shape[0], by2 + pad_y)

        crop_gray = sonar_img[cy1:cy2, cx1:cx2]
        if crop_gray.size > 0:
            crop_color = apply_sonar_colormap(crop_gray, selected_cmap)
            crop_resized = cv2.resize(crop_color, (240, 160), interpolation=cv2.INTER_NEAREST)

            c_col1, c_col2 = st.columns([1, 1])
            with c_col1:
                st.image(crop_resized, caption="4x Acoustic Zoom Crop", use_column_width=True)
            with c_col2:
                # Acoustic line-scan intensity profile
                line_profile = crop_gray[crop_gray.shape[0] // 2, :]
                import pandas as pd
                chart_df = pd.DataFrame({"Backscatter Intensity": line_profile})
                st.caption("Acoustic Line-Scan (Highlight → Shadow)")
                st.line_chart(chart_df, height=120)

        # Deep evidence scores
        st.markdown(
            f"""
            <div style="background: #0E1626; border: 1px solid rgba(0,240,255,0.2); border-radius: 8px; padding: 12px; margin-top: 10px;">
                <div style="font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">ANOMALY METRICS: <strong>{curr_det.get('detection_id', '')[:16]}</strong></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; font-size: 13px;">
                    <div>Stage-1 Detector Conf: <span style="color: #00F0FF; font-weight:700;">{curr_det.get('detector_confidence',0):.3f}</span></div>
                    <div>Stage-2 Verifier Prob: <span style="color: #10B981; font-weight:700;">{curr_det.get('artificial_probability',0):.3f}</span></div>
                    <div>Fused Artificialness: <span style="color: #FFB703; font-weight:700;">{curr_det.get('artificialness_score',0):.3f}</span></div>
                    <div>Operator Priority: <span style="color: #EF4444; font-weight:700;">{curr_det.get('priority_score',0):.3f}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
