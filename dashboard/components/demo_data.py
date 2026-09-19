"""
dashboard/components/demo_data.py — Synthetic high-fidelity demo mission data.

Generates rich, realistic side-scan sonar mission profiles and waterfall images
with targets (ghost nets, cargo containers, sunken drums, natural rock outcrops,
acoustic shadows) so the dashboard looks production-ready immediately.
"""
import math
import os
import cv2
import numpy as np


def generate_synthetic_sonar_image(
    width: int = 800,
    height: int = 600,
    targets: list = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a photorealistic acoustic side-scan sonar image.
    Features:
    - Central nadir blind zone (water column return)
    - Slant-range acoustic backscatter decay
    - Seabed sand ripples and acoustic speckle noise
    - Specular highlight target returns with dark acoustic shadows
    """
    rng = np.random.default_rng(seed)

    # 1. Base acoustic speckle noise (Rayleigh / Gamma distributed)
    base = rng.gamma(shape=2.0, scale=25.0, size=(height, width)).astype(np.float32)

    # 2. Sand ripple seabed texture (low-frequency sinusoidal patterns)
    x = np.linspace(0, 15 * np.pi, width)
    y = np.linspace(0, 10 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    ripples = 15.0 * np.sin(xx + 0.3 * np.cos(yy)) + 10.0 * np.cos(0.5 * xx)
    base += ripples

    # 3. Port & Starboard acoustic attenuation / TVG (Time Varied Gain) curve
    center_x = width // 2
    dist_from_nadir = np.abs(np.arange(width) - center_x).astype(np.float32)
    attenuation = np.exp(-dist_from_nadir / (width * 0.45))
    base *= (0.4 + 0.6 * attenuation)[np.newaxis, :]

    # 4. Central Nadir blind zone (water column - dark center band with altitude line)
    nadir_width = int(width * 0.08)
    nadir_start = center_x - nadir_width // 2
    nadir_end = center_x + nadir_width // 2
    base[:, nadir_start:nadir_end] *= 0.15

    # Bright seabed first-return lines on both sides of nadir
    base[:, nadir_start] = np.clip(base[:, nadir_start] * 2.5, 0, 255)
    base[:, nadir_end] = np.clip(base[:, nadir_end] * 2.5, 0, 255)

    img = np.clip(base, 0, 255).astype(np.uint8)

    # 5. Inject targets (Specular highlight + acoustic shadow extending away from nadir)
    targets = targets or []
    for tgt in targets:
        x1, y1, x2, y2 = tgt["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = max(int(x2 - x1), 10)
        h = max(int(y2 - y1), 8)

        # Draw highlight
        cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, 240, -1)

        # Shadow extends away from the central nadir line
        shadow_dir = 1 if cx > center_x else -1
        shadow_len = int(w * 2.5)
        shadow_x1 = cx if shadow_dir == 1 else max(0, cx - shadow_len)
        shadow_x2 = min(width - 1, cx + shadow_len) if shadow_dir == 1 else cx
        img[max(0, cy - h // 2):min(height, cy + h // 2), shadow_x1:shadow_x2] = (
            img[max(0, cy - h // 2):min(height, cy + h // 2), shadow_x1:shadow_x2] * 0.1
        ).astype(np.uint8)

    return img


def get_demo_mission(mission_id: str = "NEMO-SURVEY-ALPHA-01") -> dict:
    """Return a comprehensive high-fidelity synthetic mission report."""
    base_lat, base_lon = 9.2876, 79.1245  # Gulf of Mannar Marine Biosphere

    frames_data = []

    # Frame 1: Synthetic Ghost Net & Anchor Chain
    f1_targets = [
        {
            "detection_id": f"{mission_id}_F001_DET01",
            "bbox": {"x1": 180, "y1": 140, "x2": 260, "y2": 210},
            "detector_confidence": 0.942,
            "detector_class": "target",
            "verifier_class": "artificial",
            "artificial_probability": 0.918,
            "natural_probability": 0.082,
            "artificialness_score": 0.926,
            "priority_score": 0.948,
            "latitude": base_lat + 0.0012,
            "longitude": base_lon + 0.0008,
            "timestamp": "2026-09-19T08:14:22Z",
            "status": "verified",
            "label": "Discarded Ghost Fishing Net (Tangled)",
            "depth_m": 24.5,
            "acoustic_contrast": 0.88,
        },
        {
            "detection_id": f"{mission_id}_F001_DET02",
            "bbox": {"x1": 560, "y1": 320, "x2": 610, "y2": 370},
            "detector_confidence": 0.735,
            "detector_class": "target",
            "verifier_class": "natural",
            "artificial_probability": 0.312,
            "natural_probability": 0.688,
            "artificialness_score": 0.421,
            "priority_score": 0.380,
            "latitude": base_lat + 0.0013,
            "longitude": base_lon + 0.0015,
            "timestamp": "2026-09-19T08:14:26Z",
            "status": "rejected",
            "label": "Coral Head / Seabed Outcrop",
            "depth_m": 25.1,
            "acoustic_contrast": 0.46,
        },
    ]

    # Frame 2: Submerged Metallic Shipping Container
    f2_targets = [
        {
            "detection_id": f"{mission_id}_F002_DET01",
            "bbox": {"x1": 490, "y1": 180, "x2": 630, "y2": 260},
            "detector_confidence": 0.978,
            "detector_class": "target",
            "verifier_class": "artificial",
            "artificial_probability": 0.965,
            "natural_probability": 0.035,
            "artificialness_score": 0.969,
            "priority_score": 0.982,
            "latitude": base_lat + 0.0034,
            "longitude": base_lon + 0.0028,
            "timestamp": "2026-09-19T08:16:04Z",
            "status": "verified",
            "label": "20ft Intermodal Container",
            "depth_m": 31.8,
            "acoustic_contrast": 0.95,
        }
    ]

    # Frame 3: Sunken Drum / Cylindrical Anomaly (Unexploded Ordnance / UXO suspect)
    f3_targets = [
        {
            "detection_id": f"{mission_id}_F003_DET01",
            "bbox": {"x1": 210, "y1": 310, "x2": 270, "y2": 360},
            "detector_confidence": 0.884,
            "detector_class": "target",
            "verifier_class": "artificial",
            "artificial_probability": 0.892,
            "natural_probability": 0.108,
            "artificialness_score": 0.887,
            "priority_score": 0.915,
            "latitude": base_lat + 0.0051,
            "longitude": base_lon + 0.0042,
            "timestamp": "2026-09-19T08:18:10Z",
            "status": "verified",
            "label": "Industrial Chemical Drum / Steel Cylinder",
            "depth_m": 28.2,
            "acoustic_contrast": 0.83,
        },
        {
            "detection_id": f"{mission_id}_F003_DET02",
            "bbox": {"x1": 530, "y1": 440, "x2": 580, "y2": 490},
            "detector_confidence": 0.542,
            "detector_class": "target",
            "verifier_class": "natural",
            "artificial_probability": 0.410,
            "natural_probability": 0.590,
            "artificialness_score": 0.448,
            "priority_score": 0.412,
            "latitude": base_lat + 0.0053,
            "longitude": base_lon + 0.0049,
            "timestamp": "2026-09-19T08:18:15Z",
            "status": "candidate",
            "label": "Acoustic Ripple Shadow Anomaly",
            "depth_m": 28.6,
            "acoustic_contrast": 0.41,
        },
    ]

    # Frame 4: Clean Seabed Transect (Baseline validation)
    f4_targets = []

    frames_specs = [
        ("frame_0001", f1_targets, base_lat + 0.0010, base_lon + 0.0010, 24.8, 85.0),
        ("frame_0002", f2_targets, base_lat + 0.0030, base_lon + 0.0025, 31.2, 86.2),
        ("frame_0003", f3_targets, base_lat + 0.0050, base_lon + 0.0040, 28.4, 84.8),
        ("frame_0004", f4_targets, base_lat + 0.0070, base_lon + 0.0055, 29.1, 85.5),
    ]

    for fid, tgts, lat, lon, depth, heading in frames_specs:
        frames_data.append({
            "frame_id": fid,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "heading": heading,
            "altitude": 5.2,
            "speed_knots": 3.4,
            "frequency_khz": 455,
            "range_m": 75.0,
            "detections": tgts,
            "_raw_targets": tgts,  # Used by synthetic renderer
        })

    all_dets = [d for f in frames_data for d in f["detections"]]

    return {
        "mission_id": mission_id,
        "survey_name": "Gulf of Mannar Ecological Debris & Anomaly Survey",
        "vessel_auv": "AUV Bluefin-21 (Payload: EdgeTech 4205 Tri-Frequency)",
        "survey_date": "2026-09-19",
        "export_timestamp": "2026-09-19T08:30:00Z",
        "total_frames": len(frames_data),
        "total_detections": len(all_dets),
        "frames": frames_data,
        "operator": "Dr. A. Balaji (Lead Sonar Specialist)",
        "swath_width_m": 150.0,
        "survey_area_sq_km": 1.42,
    }
