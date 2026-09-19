"""
models/rcdi_yolo/neck.py

SonarGuard RCDI-YOLO Neck (FPN + PAN with DySample + LANConvNeXtv2).

SOURCE:
    Based on YOLOv8 FPN+PAN neck.
    RCDI modifications from Zhang, J. and Gao, B. (2025).
    Frontiers in Marine Science 12:1679077.

RCDI NECK MODIFICATIONS (from paper):
    1. Fixed bilinear upsampling -> DySample (dynamic learnable upsampling).
    2. LANConvNeXtv2 replaces C2f at neck positions 1, 2, 4.
    3. C2f retained at neck position 3.

Neck positions (in order):
    1 -> FPN: after concat(DySample(P5), P4)  [LANConvNeXtv2]
    2 -> FPN: after concat(DySample(n1), P3)  [LANConvNeXtv2]
    3 -> PAN: after concat(dn(n2), n1)        [C2f kept]
    4 -> PAN: after concat(dn(n3), P5)        [LANConvNeXtv2]

OUTPUT:
    Returns three feature maps for the detection head:
        small  (P3 scale, stride=8,  high resolution)
        medium (P4 scale, stride=16, mid resolution)
        large  (P5 scale, stride=32, low resolution)
"""
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from models.rcdi_yolo.modules.lanconvnextv2 import LANConvNeXtv2, ConvBnSilu
from models.rcdi_yolo.modules.dysample import DySample
from models.rcdi_yolo.backbone import C2f

logger = logging.getLogger(__name__)


class RCDINeck(nn.Module):
    """
    RCDI-YOLO FPN + PAN neck.

    Takes P3, P4, P5 features from the backbone and produces
    three multi-scale feature maps for the detection head.

    SOURCE: YOLOv8 neck; Zhang & Gao (2025) modifications.
    SONARGUARD: Channel-agnostic after backbone stem adaptation.

    Args:
        backbone_channels (tuple): (c_p3, c_p4, c_p5) from backbone.
        width_multiple (float):    Channel scaling factor.
        depth_multiple (float):    Depth scaling factor.
        neck_lan_positions (list): Neck C2f positions to replace with LANConvNeXtv2.
        dysample_cfg (dict):       DySample configuration.
    """

    def __init__(
        self,
        backbone_channels: Tuple[int, int, int],
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        neck_lan_positions: List[int] = None,
        dysample_cfg: dict = None,
    ):
        super().__init__()

        if neck_lan_positions is None:
            neck_lan_positions = [1, 2, 4]  # paper default
        if dysample_cfg is None:
            dysample_cfg = {"scale": 2, "groups": 4}

        def depth(n): return max(round(n * depth_multiple), 1)

        c_p3, c_p4, c_p5 = backbone_channels

        # ----------------------------------------------------------------
        # Top-down FPN path
        # ----------------------------------------------------------------

        # DySample replaces fixed upsample (SOURCE: RCDI paper)
        self.up1 = DySample(c_p5, scale=dysample_cfg.get("scale", 2),
                            groups=dysample_cfg.get("groups", 4))

        # Neck position 1: concat(up1, P4) -> feature
        n1_in = c_p5 + c_p4
        n1_out = c_p4
        if 1 in neck_lan_positions:
            self.n1 = LANConvNeXtv2(n1_in, n1_out)
            logger.debug("Neck position 1: LANConvNeXtv2(%d, %d)", n1_in, n1_out)
        else:
            self.n1 = C2f(n1_in, n1_out, n=depth(3))

        self.up2 = DySample(n1_out, scale=dysample_cfg.get("scale", 2),
                            groups=dysample_cfg.get("groups", 4))

        # Neck position 2: concat(up2, P3) -> feature (also P3 detection output)
        n2_in = n1_out + c_p3
        n2_out = c_p3
        if 2 in neck_lan_positions:
            self.n2 = LANConvNeXtv2(n2_in, n2_out)
            logger.debug("Neck position 2: LANConvNeXtv2(%d, %d)", n2_in, n2_out)
        else:
            self.n2 = C2f(n2_in, n2_out, n=depth(3))

        # ----------------------------------------------------------------
        # Bottom-up PAN path
        # ----------------------------------------------------------------

        # Downsample n2 (stride-2 conv)
        self.dn1 = ConvBnSilu(n2_out, n2_out, k=3, s=2)

        # Neck position 3: concat(dn1, n1) -> feature (P4 detection output)
        n3_in = n2_out + n1_out
        n3_out = n1_out
        if 3 in neck_lan_positions:
            self.n3 = LANConvNeXtv2(n3_in, n3_out)
            logger.debug("Neck position 3: LANConvNeXtv2(%d, %d)", n3_in, n3_out)
        else:
            # C2f kept at position 3 per paper config
            self.n3 = C2f(n3_in, n3_out, n=depth(3))

        # Downsample n3 (stride-2 conv)
        self.dn2 = ConvBnSilu(n3_out, n3_out, k=3, s=2)

        # Neck position 4: concat(dn2, P5) -> feature (P5 detection output)
        n4_in = n3_out + c_p5
        n4_out = c_p5
        if 4 in neck_lan_positions:
            self.n4 = LANConvNeXtv2(n4_in, n4_out)
            logger.debug("Neck position 4: LANConvNeXtv2(%d, %d)", n4_in, n4_out)
        else:
            self.n4 = C2f(n4_in, n4_out, n=depth(3))

        # Output channel counts for the detection head
        self.out_channels = (n2_out, n3_out, n4_out)  # P3, P4, P5 scales

        logger.info(
            "RCDINeck: backbone_channels=%s out_channels=%s LAN@neck=%s",
            backbone_channels, self.out_channels, neck_lan_positions,
        )

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (P3, P4, P5) from backbone.

        Returns:
            (small, medium, large) detection feature maps.
        """
        p3, p4, p5 = features

        # FPN top-down
        up1_out = self.up1(p5)                                 # upsample P5
        n1_out  = self.n1(torch.cat([up1_out, p4], dim=1))    # neck pos 1

        up2_out = self.up2(n1_out)                             # upsample n1
        n2_out  = self.n2(torch.cat([up2_out, p3], dim=1))    # neck pos 2 -> P3 det

        # PAN bottom-up
        dn1_out = self.dn1(n2_out)
        n3_out  = self.n3(torch.cat([dn1_out, n1_out], dim=1))  # neck pos 3 -> P4 det

        dn2_out = self.dn2(n3_out)
        n4_out  = self.n4(torch.cat([dn2_out, p5], dim=1))    # neck pos 4 -> P5 det

        return n2_out, n3_out, n4_out  # small, medium, large
