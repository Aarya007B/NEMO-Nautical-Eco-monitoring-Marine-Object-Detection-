"""
models/rcdi_yolo/modules/dysample.py

DySample — Dynamic Upsampling Module.

SOURCE:
    Zhang, J. and Gao, B. (2025). "RCDI-YOLO: a target-detection method for
    complex environment side-scan sonar images based on improved YOLOv8."
    Frontiers in Marine Science 12:1679077.

Core DySample mechanism:
    Liu, W. et al. (2023). "DySample: An Ultra-Simple Dynamic Upsampler."
    ICCV 2023.

DESIGN PURPOSE (from paper):
    Replaces fixed bilinear/nearest upsampling in the FPN neck with a
    learnable sampling mechanism that adapts to target scale and local
    feature characteristics. Better preserves target edges and boundaries
    in noisy sonar imagery.

NEMO:
    Used in the RCDI neck to upsample feature maps in the FPN top-down path.
    Module is channel-agnostic and works with 1-channel sonar adaptation.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DySample(nn.Module):
    """
    Dynamic Upsampling module.

    Generates learnable offset sampling positions from the input feature map
    and uses differentiable bilinear sampling (F.grid_sample) to produce
    an upsampled output.

    SOURCE: Zhang & Gao (2025) neck modification; Liu et al. ICCV 2023.

    Args:
        in_channels (int): Input feature channel count.
        scale (int):       Upsampling scale factor (default: 2).
        groups (int):      Number of offset groups for multi-group sampling.
                           Must divide in_channels evenly.
    """

    def __init__(self, in_channels: int, scale: int = 2, groups: int = 4):
        super().__init__()
        # Ensure groups divides channels
        while in_channels % groups != 0 and groups > 1:
            groups -= 1

        self.scale  = scale
        self.groups = groups
        self.in_channels = in_channels

        # Offset generator: produces (dx, dy) per group
        self.offset_conv = nn.Conv2d(in_channels, groups * 2, kernel_size=1, bias=True)
        # Scope: learnable scale for offset magnitude
        self.scope_conv  = nn.Conv2d(in_channels, groups * 2, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.scope_conv.weight)
        nn.init.constant_(self.scope_conv.bias, 1.0)  # initial scope = 1

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create a [H, W, 2] normalized grid in [-1, 1] for grid_sample."""
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W].

        Returns:
            Upsampled feature map [B, C, H*scale, W*scale].
        """
        B, C, H, W = x.shape
        H_out, W_out = H * self.scale, W * self.scale
        G = self.groups
        Cg = C // G  # channels per group

        # Generate offsets and scopes at input resolution
        offsets = self.offset_conv(x)                       # [B, G*2, H, W]
        scope   = torch.sigmoid(self.scope_conv(x))         # [B, G*2, H, W] in (0,1)

        # Upsample offset maps to output resolution
        offsets = F.interpolate(offsets, size=(H_out, W_out), mode="bilinear", align_corners=False)
        scope   = F.interpolate(scope,   size=(H_out, W_out), mode="bilinear", align_corners=False)

        # Reshape to [B, G, H_out, W_out, 2]
        offsets = offsets.view(B, G, 2, H_out, W_out).permute(0, 1, 3, 4, 2).contiguous()
        scope   = scope.view(  B, G, 2, H_out, W_out).permute(0, 1, 3, 4, 2).contiguous()

        # Scale offsets; (2/H_out) brings them into [-1,1] normalised space
        normalized_offsets = offsets * scope * (2.0 / max(H_out, W_out))

        # Base sampling grid [H_out, W_out, 2] -> [B, H_out, W_out, 2]
        base = self._make_base_grid(H_out, W_out, x.device)  # [H_out, W_out, 2]
        base = base.unsqueeze(0).expand(B, -1, -1, -1)        # [B, H_out, W_out, 2]
        # Expand base to [B, G, H_out, W_out, 2]
        base = base.unsqueeze(1).expand(-1, G, -1, -1, -1)

        # Final sampling grid: base + learned offsets
        sample_grid = (base + normalized_offsets).clamp(-1.0, 1.0)  # [B, G, H_out, W_out, 2]

        # Bilinear upsample x for content (then add offset-based correction)
        x_up = F.interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=False)

        # Per-group differentiable sampling
        out_groups = []
        for g in range(G):
            feat = x_up[:, g * Cg:(g + 1) * Cg]          # [B, Cg, H_out, W_out]
            grid = sample_grid[:, g]                       # [B, H_out, W_out, 2]
            pad_mode = "zeros" if feat.device.type == "mps" else "border"
            sampled = F.grid_sample(
                feat, grid,
                mode="bilinear",
                align_corners=True,
                padding_mode=pad_mode,
            )                                              # [B, Cg, H_out, W_out]
            out_groups.append(sampled)

        return torch.cat(out_groups, dim=1)  # [B, C, H_out, W_out]
