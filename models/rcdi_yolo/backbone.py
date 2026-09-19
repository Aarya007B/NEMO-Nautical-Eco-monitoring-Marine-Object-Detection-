"""
models/rcdi_yolo/backbone.py

SonarGuard RCDI-YOLO Backbone (YOLOv8-style, 1-channel input).

SOURCE:
    Based on YOLOv8 backbone architecture.
    RCDI modifications from Zhang, J. and Gao, B. (2025).
    Frontiers in Marine Science 12:1679077.

SONARGUARD ADAPTATIONS:
    1. First convolution: in_channels=1 (sonar acoustic-intensity, not RGB).
    2. LANConvNeXtv2 replaces C2f at backbone positions 1 and 2 (paper config).
    3. Remaining C2f blocks preserved as per YOLOv8 design.

OUTPUT:
    Returns feature maps at three scales for the FPN neck:
        P3: [B, c2, H/8,  W/8 ]  (small targets)
        P4: [B, c3, H/16, W/16]  (medium targets)
        P5: [B, c4, H/32, W/32]  (large targets, before SPPF)
"""
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from models.rcdi_yolo.modules.lanconvnextv2 import LANConvNeXtv2, ConvBnSilu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """YOLOv8-style bottleneck block."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBnSilu(c1, c_, k=3)
        self.cv2 = ConvBnSilu(c_, c2, k=3)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    YOLOv8 Cross Stage Partial Bottleneck with 2 convolutions.
    Preserved at backbone positions NOT replaced by LANConvNeXtv2.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBnSilu(c1, 2 * self.c, k=1)
        self.cv2 = ConvBnSilu((2 + n) * self.c, c2, k=1)
        self.m = nn.ModuleList(
            [Bottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (from YOLOv8)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBnSilu(c1, c_, k=1)
        self.cv2 = ConvBnSilu(c_ * 4, c2, k=1)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# RCDIBackbone
# ---------------------------------------------------------------------------

class RCDIBackbone(nn.Module):
    """
    RCDI-YOLO backbone with 1-channel sonar input and LANConvNeXtv2
    at configured positions.

    Channel widths are scaled by width_multiple from base_channels.
    Depth (n bottlenecks in C2f) is scaled by depth_multiple.

    SOURCE: YOLOv8 backbone; Zhang & Gao (2025) modifications.
    SONARGUARD: First conv changed to in_channels=1.

    Args:
        in_channels (int):    Input channels (1 for sonar).
        base_channels (list): Base channel widths before scaling.
        width_multiple (float): Channel width multiplier.
        depth_multiple (float): Bottleneck depth multiplier.
        backbone_lan_positions (list): C2f positions to replace with LANConvNeXtv2.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: List[int] = None,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        backbone_lan_positions: List[int] = None,
    ):
        super().__init__()

        if base_channels is None:
            base_channels = [64, 128, 256, 512, 512]
        if backbone_lan_positions is None:
            backbone_lan_positions = [1, 2]  # paper default

        # Scale channels
        def ch(i): return max(round(base_channels[i] * width_multiple), 1)
        # Scale depth
        def depth(n): return max(round(n * depth_multiple), 1)

        c0 = ch(0)  # stem
        c1 = ch(1)  # P2
        c2 = ch(2)  # P3
        c3 = ch(3)  # P4
        c4 = ch(4)  # P5

        # ----- Stem -----
        # SONARGUARD: in_channels=1 (not 3)
        self.stem = ConvBnSilu(in_channels, c0, k=3, s=2)   # /2 -> 320x320

        # ----- P2 stage -----
        self.down1 = ConvBnSilu(c0, c1, k=3, s=2)           # /4 -> 160x160
        # Backbone position 1
        if 1 in backbone_lan_positions:
            self.stage1 = LANConvNeXtv2(c1, c1)
            logger.debug("Backbone position 1: LANConvNeXtv2(%d, %d)", c1, c1)
        else:
            self.stage1 = C2f(c1, c1, n=depth(3), shortcut=True)

        # ----- P3 stage -----
        self.down2 = ConvBnSilu(c1, c2, k=3, s=2)           # /8 -> 80x80
        # Backbone position 2
        if 2 in backbone_lan_positions:
            self.stage2 = LANConvNeXtv2(c2, c2)
            logger.debug("Backbone position 2: LANConvNeXtv2(%d, %d)", c2, c2)
        else:
            self.stage2 = C2f(c2, c2, n=depth(6), shortcut=True)

        # ----- P4 stage -----
        self.down3 = ConvBnSilu(c2, c3, k=3, s=2)           # /16 -> 40x40
        self.stage3 = C2f(c3, c3, n=depth(6), shortcut=True)  # C2f #3 (not replaced)

        # ----- P5 stage -----
        self.down4 = ConvBnSilu(c3, c4, k=3, s=2)           # /32 -> 20x20
        self.stage4 = C2f(c4, c4, n=depth(3), shortcut=True)  # C2f #4 (not replaced)
        self.sppf   = SPPF(c4, c4)

        # Store output channel sizes for neck
        self.out_channels = (c2, c3, c4)  # P3, P4, P5

        logger.info(
            "RCDIBackbone: in=%d, channels=%s, LAN@backbone=%s",
            in_channels, self.out_channels, backbone_lan_positions,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, H, W] sonar tensor.

        Returns:
            (P3, P4, P5) feature maps.
        """
        x  = self.stem(x)
        x  = self.down1(x)
        x  = self.stage1(x)
        p3 = self.stage2(self.down2(x))   # P3
        p4 = self.stage3(self.down3(p3))  # P4
        p5 = self.sppf(self.stage4(self.down4(p4)))  # P5
        return p3, p4, p5
