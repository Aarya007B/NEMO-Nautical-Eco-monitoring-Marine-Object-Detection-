# NEMO 🌊🛥️
### AI-Powered Automated Underwater Marine Debris & Anomaly Detection

**NEMO** is an end-to-end automated computer vision pipeline designed to ingest Side-Scan Sonar (SSS) imagery, identify man-made debris (such as ghost nets, shipwrecks, and pipelines) against complex natural backgrounds, and generate actionable localized data.

Designed for the **Smart India Hackathon (SIH)**, NEMO leverages state-of-the-art 2025 acoustic computer vision research to deliver a highly accurate, edge-optimized application for marine conservationists and naval operators.

---

## 🚀 Core Innovation: The RCDI-YOLO Engine
Traditional sonar object detection suffers from high false-positive rates due to acoustic shadows, seabed clutter, and speckle noise. 

NEMO abandons slow, multi-stage verification pipelines in favor of a customized **RCDI-YOLO** architecture. Based on cutting-edge research, we modified the YOLOv8 backbone with advanced neural modules to natively handle sonar physics:
*   **LANConvNeXtv2:** Extracts multi-scale, low-contrast seabed features natively, replacing manual texture analysis.
*   **Dysample:** Dynamically adapts the network's upsampling rate based on target size, maintaining robust detection across varying spatial resolutions.
*   **ImplicitHead:** Utilizes implicit feature representations to natively filter background noise and distinguish true artificial targets from natural marine formations (rocks, ridges) without a secondary classifier.

## ✨ Key Features
*   **Acoustic-Optimized Preprocessing:** Automated CLAHE and denoising via OpenCV to normalize raw sonar intensity and preserve critical shadow boundaries.
*   **Edge-Ready AI Engine:** The 3.23M parameter RCDI-YOLO model compiles to ONNX and TensorRT (FP16), enabling real-time inference (≥ 15 FPS) on edge devices like NVIDIA Jetson AUV hardware.
*   **Geospatial Reporting & Geotagging:** Parses ping metadata to associate bounding box detections with precise GPS coordinates, exporting structured CSV and JSON reports.
*   **Operator UI Dashboard:** A React + Vite powered interface featuring MapLibre/Leaflet integration. Users can upload sonar logs, view bounding box overlays, track mission paths, and manage anomaly reviews in real-time.

---

## 🛠️ Technology Stack
**Machine Learning & Computer Vision**
*   PyTorch & Ultralytics (Modified YOLOv8 Core)
*   OpenCV & NumPy (Image Preprocessing)

**Backend & Data Layer**
*   Python & FastAPI (Inference API)
*   SQLite & Pandas (Mission metadata and reporting)

**Frontend & Visualization**
*   React + Vite (UI Dashboard)
*   Leaflet / MapLibre (Geospatial tracking)

**Deployment & Edge Optimization**
*   ONNX & TensorRT FP16

---

## ⚙️ Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/NEMO.git
cd NEMO

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts ctivate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000/docs`*

### 3. Running the Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
*The dashboard will be available at `http://localhost:5173`*

---

## 📖 Scientific References
The core detection engine in this project is engineered using the architectural specifications from:
*   *Zhang, J., & Gao, B. (2025). RCDI-YOLO: a target-detection method for complex environment side-scan sonar images based on improved YOLOv8. Frontiers in Marine Science, 12:1679077.*