"""
models/rcdi_yolo/modules/implicit_head.py

ImplicitHead — Detection head with implicit feature adapters.

SOURCE:
    Zhang, J. and Gao, B. (2025). "RCDI-YOLO: a target-detection method for
    complex environment side-scan sonar images based on improved YOLOv8."
    Frontiers in Marine Science 12:1679077.

DESIGN PURPOSE (from paper):
    Combines:
      - Sequential convolutional feature processing
      - DFL bounding-box regression
      - ImplicitA and ImplicitM adapters
    The head produces standard detection outputs while adapting to the
    noisy sonar feature distribution.

OUTPUT API:
    Returns raw tensors (before decode/NMS) for each detection scale.
    The RCDIYOLOModel applies decode + NMS.
"""
import logging
import math
from typing import List

import torch
import torch.nn as nn

from models.rcdi_yolo.modules.implicit import ImplicitA, ImplicitM

logger = logging.getLogger(__name__)


class DFL(nn.Module):
    """
    Distribution Focal Loss distribution-to-value layer.

    Converts the raw DFL distribution logits (reg_max bins per
    bounding-box edge) into a single distance value via softmax + linear.

    SOURCE: Li, X. et al. (2022). "Generalized Focal Loss V2." — DFL.
    Used in YOLOv8 and retained in RCDI-YOLO.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        # Fixed weight: [0, 1, ..., reg_max-1]
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        self.conv.weight.data.copy_(
            torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        )
        self.conv.requires_grad_(False)  # non-trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4*reg_max, H, W] raw distribution logits.

        Returns:
            [B, 4, H, W] decoded box distances (ltrb).
        """
        B, _, H, W = x.shape
        # Reshape: [B, 4*reg_max, H, W] -> [B*H*W*4, reg_max]
        x = x.view(B, 4, self.reg_max, H * W).permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, self.reg_max)
        x = x.softmax(dim=-1)                               # [N, reg_max]
        x = x.unsqueeze(-1).unsqueeze(-1)                    # [N, reg_max, 1, 1]
        x = self.conv(x).view(-1)                            # [N]
        x = x.view(B, H * W, 4).permute(0, 2, 1).contiguous()  # [B, 4, H*W]
        return x.view(B, 4, H, W)


class HeadBranch(nn.Module):
    """
    Single-scale detection head branch with ImplicitA/M adapters.

    Processes features from one FPN level and produces raw box + cls predictions.

    Args:
        in_ch (int):     Input channels from neck.
        num_classes (int): Number of object classes.
        reg_max (int):   DFL distribution bins (default 16).
    """

    def __init__(self, in_ch: int, num_classes: int, reg_max: int = 16):
        super().__init__()
        self.reg_max     = reg_max
        self.num_classes = num_classes
        box_out = 4 * reg_max

        # --- Box regression branch ---
        self.box_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )
        self.ia_box = ImplicitA(in_ch)
        self.im_box = ImplicitM(box_out)
        self.box_pred = nn.Conv2d(in_ch, box_out, 1)

        # --- Classification branch ---
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )
        self.ia_cls = ImplicitA(in_ch)
        self.im_cls = ImplicitM(num_classes)
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        # Bias initialization: classification head prior ~P(object)=0.01
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.cls_pred.bias is not None:
            nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, x: torch.Tensor):
        """Returns (box_raw [B,4*reg_max,H,W], cls_raw [B,num_classes,H,W])."""
        box = self.ia_box(self.box_conv(x))
        box = self.im_box(self.box_pred(box))

        cls = self.ia_cls(self.cls_conv(x))
        cls = self.im_cls(self.cls_pred(cls))

        return box, cls


class ImplicitHead(nn.Module):
    """
    Multi-scale detection head using implicit feature adapters.

    Wraps one HeadBranch per detection scale (P3, P4, P5).
    The DFL module decodes raw box distributions into distances.

    SOURCE: Zhang & Gao (2025), ImplicitHead section.
    SONARGUARD: num_classes defaults to 1 (single 'target' class).

    Args:
        in_channels (List[int]): Feature channels for each detection scale.
        num_classes (int):       Number of detection classes.
        reg_max (int):           DFL distribution bins.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 1,
        reg_max: int = 16,
    ):
        super().__init__()
        self.reg_max     = reg_max
        self.num_classes = num_classes
        self.num_scales  = len(in_channels)

        self.branches = nn.ModuleList(
            [
                HeadBranch(c, num_classes, reg_max)
                for c in in_channels
            ]
        )
        self.dfl = DFL(reg_max)

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [P3, P4, P5], each [B, C_i, H_i, W_i].

        Returns:
            List of raw prediction tensors, one per scale.
            Each tensor: [B, 4*reg_max + num_classes, H_i, W_i].
            (decode and NMS happen in RCDIYOLOModel.predict)
        """
        outputs = []
        for feat, branch in zip(features, self.branches):
            box, cls = branch(feat)
            outputs.append(torch.cat([box, cls], dim=1))  # [B, 4*reg_max+nc, H, W]
        return outputs
