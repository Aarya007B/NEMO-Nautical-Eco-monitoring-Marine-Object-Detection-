"""
models/rcdi_yolo/model.py

RCDIYOLOModel — NEMO RCDI-YOLO 1-channel sonar detector.

SOURCE:
    Architecture based on Zhang, J. and Gao, B. (2025).
    "RCDI-YOLO: a target-detection method for complex environment
    side-scan sonar images based on improved YOLOv8."
    Frontiers in Marine Science 12:1679077.

NEMO ADAPTATIONS:
    - Input: 1-channel sonar acoustic-intensity images [B, 1, H, W]
      (paper uses 3-channel RGB)
    - Class: single unified 'target' class for MVP
    - Architecture otherwise follows published RCDI design

USAGE:
    model = RCDIYOLOModel(in_channels=1, num_classes=1)
    detections = model.predict(tensor)  # [1, 1, 640, 640]
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rcdi_yolo.backbone import RCDIBackbone
from models.rcdi_yolo.neck import RCDINeck
from models.rcdi_yolo.modules.implicit_head import ImplicitHead
from inference.types import Detection, BoundingBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anchor-free box decoding utilities
# ---------------------------------------------------------------------------

def make_anchors(
    feature_maps: List[torch.Tensor],
    strides: List[int],
    offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor center points for all detection scales.

    Returns:
        anchor_points: [total_anchors, 2]  (cx, cy in feature-map coords)
        stride_tensor: [total_anchors, 1]  (stride value per anchor)
    """
    all_points, all_strides = [], []
    for feat, stride in zip(feature_maps, strides):
        _, _, h, w = feat.shape
        sy = torch.arange(h, device=feat.device) + offset
        sx = torch.arange(w, device=feat.device) + offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        all_points.append(torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1))
        all_strides.append(torch.full((h * w, 1), stride, device=feat.device))
    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
) -> torch.Tensor:
    """
    Convert LTRB distance predictions to bounding boxes.

    Args:
        distance:      [N, 4] predicted LTRB distances from anchor centers.
        anchor_points: [N, 2] anchor center (cx, cy) in feature-map coords.
        xywh:          If True return cx,cy,w,h; else return x1,y1,x2,y2.

    Returns:
        [N, 4] bounding boxes.
    """
    lt, rb = distance.chunk(2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh   = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim=-1)
    return torch.cat([x1y1, x2y2], dim=-1)


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.70,
) -> torch.Tensor:
    """
    Non-Maximum Suppression.

    Args:
        boxes:  [N, 4] x1y1x2y2 boxes.
        scores: [N]    confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Keep indices tensor.
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RCDIYOLOModel(nn.Module):
    """
    NEMO RCDI-YOLO detector.

    Assembles backbone + neck + head into a complete detector.
    The CandidateDetector wrapper in inference/detector.py is the
    preferred external interface; this class handles the raw network.

    SOURCE: Zhang & Gao (2025) RCDI-YOLO; YOLOv8 base.
    NEMO: 1-channel input, single 'target' class.

    Args:
        in_channels (int):             Input channels (1 for sonar).
        num_classes (int):             Detection classes.
        base_channels (list):          Base channel widths.
        width_multiple (float):        Channel scaling factor.
        depth_multiple (float):        Depth scaling factor.
        reg_max (int):                 DFL distribution bins.
        backbone_lan_positions (list): Backbone C2f positions to use LANConvNeXtv2.
        neck_lan_positions (list):     Neck C2f positions to use LANConvNeXtv2.
        dysample_cfg (dict):           DySample kwargs.
        use_implicit_head (bool):      Use ImplicitHead vs standard head.
        strides (list):                Detection strides (default [8,16,32]).
        conf_threshold (float):        Inference confidence threshold.
        nms_iou (float):               NMS IoU threshold.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: List[int] = None,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        reg_max: int = 16,
        backbone_lan_positions: List[int] = None,
        neck_lan_positions: List[int] = None,
        dysample_cfg: Dict = None,
        use_implicit_head: bool = True,
        strides: List[int] = None,
        conf_threshold: float = 0.25,
        nms_iou: float = 0.70,
    ):
        super().__init__()

        if base_channels is None:
            base_channels = [64, 128, 256, 512, 512]
        if backbone_lan_positions is None:
            backbone_lan_positions = [1, 2]
        if neck_lan_positions is None:
            neck_lan_positions = [1, 2, 4]
        if dysample_cfg is None:
            dysample_cfg = {"scale": 2, "groups": 4}
        if strides is None:
            strides = [8, 16, 32]

        self.num_classes    = num_classes
        self.reg_max        = reg_max
        self.strides        = strides
        self.conf_threshold = conf_threshold
        self.nms_iou        = nms_iou

        # --- Backbone ---
        self.backbone = RCDIBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            backbone_lan_positions=backbone_lan_positions,
        )

        # --- Neck ---
        self.neck = RCDINeck(
            backbone_channels=self.backbone.out_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            neck_lan_positions=neck_lan_positions,
            dysample_cfg=dysample_cfg,
        )

        # --- Head ---
        self.head = ImplicitHead(
            in_channels=list(self.neck.out_channels),
            num_classes=num_classes,
            reg_max=reg_max,
        )

        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "RCDIYOLOModel: in_ch=%d, classes=%d, params=%.2fM",
            in_channels, num_classes, total_params / 1e6,
        )
        logger.info(
            "  backbone_out=%s, neck_out=%s",
            self.backbone.out_channels, self.neck.out_channels,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Raw forward pass.

        Args:
            x: [B, 1, H, W] normalized sonar tensor.

        Returns:
            List of raw prediction tensors per scale.
            Each: [B, 4*reg_max + num_classes, H_i, W_i].
        """
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds = self.head(list(feats))
        return preds

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        frame_id: str = "unknown",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run detector inference and return decoded detections.

        Applies decode (anchor-free dist2bbox) + confidence threshold +
        NMS. Returns a list of Detection objects (inference/types.py).

        Args:
            x:              [1, 1, H, W] sonar tensor (batch size 1 for inference).
            frame_id:       Frame identifier for returned Detection objects.
            conf_threshold: Optional override for confidence threshold.
            iou_threshold:  Optional override for NMS IoU threshold.

        Returns:
            List[Detection] — may be empty if no detections pass threshold.
        """
        was_training = self.training
        self.eval()

        preds = self.forward(x)  # List of raw tensors

        # Generate anchor points
        anchor_points, stride_tensor = make_anchors(preds, self.strides)

        # Decode all scales
        all_boxes, all_scores = [], []
        offset = 0
        for pred, stride in zip(preds, self.strides):
            B, C, H, W = pred.shape
            n_anchors = H * W

            # Split box and cls channels
            box_ch = 4 * self.reg_max
            box_raw = pred[:, :box_ch].permute(0, 2, 3, 1).reshape(B, n_anchors, box_ch)
            cls_raw = pred[:, box_ch:].permute(0, 2, 3, 1).reshape(B, n_anchors, self.num_classes)

            # DFL decode box distribution -> LTRB distances
            # Simplified: take argmax of each bin group (approximation)
            box_dist = box_raw.view(B, n_anchors, 4, self.reg_max).softmax(-1)
            bins = torch.arange(self.reg_max, device=x.device, dtype=x.dtype)
            ltrb = (box_dist * bins).sum(-1)  # [B, n_anchors, 4]

            # Convert LTRB to xyxy in image coords
            ap = anchor_points[offset:offset + n_anchors]
            lt, rb = ltrb[..., :2], ltrb[..., 2:]
            x1y1 = (ap - lt) * stride
            x2y2 = (ap + rb) * stride
            boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)  # [B, n_anchors, 4]

            # Class scores
            scores = cls_raw.sigmoid()  # [B, n_anchors, num_classes]

            all_boxes.append(boxes_xyxy)
            all_scores.append(scores)
            offset += n_anchors

        all_boxes  = torch.cat(all_boxes,  dim=1)  # [B, total_anchors, 4]
        all_scores = torch.cat(all_scores, dim=1)  # [B, total_anchors, nc]

        # For batch_size=1, apply NMS
        detections = []
        boxes   = all_boxes[0]   # [total_anchors, 4]
        scores  = all_scores[0]  # [total_anchors, nc]
        max_scores, class_ids = scores.max(dim=-1)  # [total_anchors]

        # Confidence filter
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        effective_iou = iou_threshold if iou_threshold is not None else self.nms_iou

        mask = max_scores >= effective_conf
        boxes      = boxes[mask]
        max_scores = max_scores[mask]
        class_ids  = class_ids[mask]

        if boxes.numel() > 0:
            try:
                import torchvision
                keep = torchvision.ops.nms(boxes, max_scores, effective_iou)
            except Exception:
                # Fallback: return all above threshold if torchvision NMS fails
                keep = torch.arange(len(boxes))

            for idx in keep:
                b  = boxes[idx]
                sc = max_scores[idx].item()
                cl = class_ids[idx].item()
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x1=float(b[0]), y1=float(b[1]),
                            x2=float(b[2]), y2=float(b[3]),
                        ),
                        detector_confidence=sc,
                        class_id=cl,
                        class_name="target",
                        frame_id=frame_id,
                        detection_id=f"{frame_id}_{idx}",
                    )
                )

        if was_training:
            self.train()

        logger.debug(
            "Frame %s: %d candidate(s) above conf=%.2f",
            frame_id, len(detections), self.conf_threshold,
        )
        return detections

    @classmethod
    def from_config(cls, config_path: str, **overrides) -> "RCDIYOLOModel":
        """
        Construct model from a YAML config file.

        Args:
            config_path: Path to configs/rcdi_yolo_1c.yaml.
            **overrides: Any constructor argument overrides.

        Returns:
            Initialized RCDIYOLOModel.
        """
        from models.rcdi_yolo.parser import get_model_kwargs_from_file
        kwargs = get_model_kwargs_from_file(config_path)
        kwargs.update(overrides)
        return cls(**kwargs)

    def load_weights(self, weights_path: str) -> None:
        """
        Load pretrained weights from a .pt file.

        Args:
            weights_path: Path to weight file.
        """
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. "
                "Place weights in models/rcdi_yolo/weights/. "
                "See README.md for details."
            )
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        self.load_state_dict(state, strict=False)
        logger.info("Loaded weights from: %s", weights_path)
