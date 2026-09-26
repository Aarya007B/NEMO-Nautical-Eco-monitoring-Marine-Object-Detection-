# RCDI-YOLO — Experimental / Research Module

## Status: EXPERIMENTAL / FUTURE WORK

This directory contains the RCDI-YOLO implementation, an architecture based on Zhang, J. and Gao, B. (2025). *"RCDI-YOLO: a target-detection method for complex environment side-scan sonar images based on improved YOLOv8."* Frontiers in Marine Science 12:1679077.

## Important Notes

1. **Not used by default.** The validated NEMO MVP uses YOLO11n-1C as the Stage-1 detector. RCDI-YOLO is NOT loaded in the default inference path.

2. **No claim of validated NEMO performance.** While the RCDI-YOLO architecture was implemented and unit-tested, a controlled apples-to-apples comparison against YOLO11n-1C on the KLSG dataset (identical data, splits, and hyperparameters) was not completed. Therefore, no performance claims are made for RCDI-YOLO within the NEMO system.

3. **Candidate for future work.** A comparative training study is listed as a research item in `docs/FUTURE_WORK.md`. If RCDI-YOLO demonstrates measurable improvement over YOLO11n-1C on sonar data in a controlled experiment, it could be promoted to the default detector.

4. **Paper results are not NEMO results.** The published RCDI-YOLO paper reports results on the CESSSD dataset using an RTX 3090 with 3-channel RGB input. These numbers (95.7% mAP@0.5, ~163 FPS) are specific to the paper's experimental setup and should NOT be cited as NEMO benchmarks.

## Architecture

Three modifications to YOLOv8:

- **LANConvNeXtv2** — Multi-scale dilated convolution + channel attention
- **DySample** — Dynamic learnable upsampling in FPN neck
- **ImplicitHead** — Detection head with ImplicitA/M adapters and DFL

## NEMO Adaptation

- Input changed from 3-channel RGB to 1-channel sonar `[B, 1, H, W]`
- Single unified `target` class for MVP

## Usage (Research Only)

```bash
# Use the research RCDI configuration
python main.py --mode recorded --config configs/research_rcdi.yaml --mission path/to/mission/
```

## Files

| File | Description |
|------|-------------|
| `model.py` | Full RCDI-YOLO model (backbone + neck + head) |
| `backbone.py` | RCDI backbone with LANConvNeXtv2 |
| `neck.py` | FPN/PAN neck with DySample |
| `modules/` | Custom modules (LANConvNeXtv2, DySample, ImplicitHead) |
| `losses.py` | Training loss functions |
| `dataset.py` | Detection dataset loader |
| `train.py` | Training script |
| `validate.py` | Validation script |
| `parser.py` | Config file parser |

## Citation

```bibtex
@article{zhang2025rcdiyolo,
  author  = {Zhang, Jinhao and Gao, Bo},
  title   = {RCDI-YOLO: a target-detection method for complex environment
             side-scan sonar images based on improved YOLOv8},
  journal = {Frontiers in Marine Science},
  volume  = {12},
  pages   = {1679077},
  year    = {2025},
  doi     = {10.3389/fmars.2025.1679077}
}
```
