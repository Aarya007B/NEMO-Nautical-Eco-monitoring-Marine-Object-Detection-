"""
models/rcdi_yolo/modules/lanconvnextv2.py

LANConvNeXtv2 — Lightweight Attention Network with ConvNeXt v2-inspired design.

SOURCE:
    Zhang, J. and Gao, B. (2025). "RCDI-YOLO: a target-detection method for
    complex environment side-scan sonar images based on improved YOLOv8."
    Frontiers in Marine Science 12:1679077.

DESIGN PURPOSE (from paper):
    LANConvNeXtv2 is designed to improve feature extraction for low-contrast,
    blurred, noisy, and partially occluded sonar targets by combining:
      - Lightweight attention for feature selection
      - ConvNeXt v2-inspired inverted bottleneck design
      - Multi-scale dilated convolution for varying target scales
      - Channel attention to suppress noisy sonar backgrounds

PLACEMENT (from paper):
    Backbone: replaces C2f at positions 1, 2
    Neck:     replaces C2f at positions 1, 2, 4

SONARGUARD ADAPTATION:
    This module operates on feature maps and is channel-agnostic after
    the backbone stem. It is compatible with the 1-channel sonar input
    adaptation used by SonarGuard (the first conv layer handles 1-channel;
    this module sees intermediate feature maps at any channel width).
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnSilu(nn.Module):
    """Conv2d + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        p: int = None,
        g: int = 1,
        d: int = 1,
    ):
        super().__init__()
        if p is None:
            p = (k - 1) // 2 * d
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-scale dilated convolution.

    Three parallel branches with dilation rates 1, 2, 4 capture features
    at different effective receptive fields simultaneously.

    This directly addresses the variable target scale problem in side-scan
    sonar imagery described in Zhang & Gao (2025).

    SOURCE: Zhang & Gao (2025), LANConvNeXtv2 design section.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Split channels across branches (ensure divisible by 3)
        b = max(channels // 3, 1)
        self.b1 = ConvBnSilu(channels, b, k=3, d=1)
        self.b2 = ConvBnSilu(channels, b, k=3, d=2)
        self.b3 = ConvBnSilu(channels, b, k=3, d=4)
        # Fuse: may not recover exactly `channels` if not divisible by 3
        self.fuse = ConvBnSilu(b * 3, channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Emphasizes feature channels carrying target evidence and suppresses
    channels dominated by speckle noise or seabed clutter.

    SOURCE: Zhang & Gao (2025), attention component of LANConvNeXtv2.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# LANConvNeXtv2
# ---------------------------------------------------------------------------

class LANConvNeXtv2(nn.Module):
    """
    LANConvNeXtv2 — Drop-in replacement for selected C2f blocks.

    Architecture:
        Input
          ↓
        Depthwise 7×7 conv (ConvNeXt v2 style)
          ↓
        Pointwise BN+SiLU
          ↓
        Multi-scale dilated convolution (d=1,2,4)
          ↓
        Channel attention (SE-style)
          ↓
        Output projection
          +
        Residual shortcut (with projection if channels differ)

    SOURCE: Zhang, J. and Gao, B. (2025), Frontiers in Marine Science 12:1679077.
    SONARGUARD: Compatible with 1-channel sonar adaptation.

    Args:
        in_channels (int):  Input channel count.
        out_channels (int): Output channel count.
        expansion (float):  Hidden channel expansion factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 1.0,
    ):
        super().__init__()
        hidden = max(int(out_channels * expansion), 1)

        # ConvNeXt v2 depthwise: groups=in_channels gives per-channel spatial mixing
        dw_groups = in_channels if in_channels == hidden else 1
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3, groups=dw_groups, bias=False),
            nn.BatchNorm2d(hidden),
        )

        # Pointwise projection
        self.pw = ConvBnSilu(hidden, hidden, k=1)

        # Multi-scale dilated convolution
        self.ms_dilated = MultiScaleDilatedConv(hidden)

        # Channel attention
        self.attn = ChannelAttention(hidden)

        # Output projection
        self.out_proj = ConvBnSilu(hidden, out_channels, k=1)

        # Residual connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dw_conv(x)
        out = self.pw(out)
        out = self.ms_dilated(out)
        out = self.attn(out)
        out = self.out_proj(out)
        return self.act(out + residual)
