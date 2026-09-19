"""
models/rcdi_yolo/modules/implicit.py

ImplicitA and ImplicitM — Learnable implicit feature adapters.

SOURCE:
    Zhang, J. and Gao, B. (2025). "RCDI-YOLO: a target-detection method for
    complex environment side-scan sonar images based on improved YOLOv8."
    Frontiers in Marine Science 12:1679077.

    Original implicit feature concept:
    Wang, C. et al. (2021). "You Only Learn One Representation: Unified
    Network for Multiple Tasks." YOLOR.

DESIGN PURPOSE (from paper):
    ImplicitA (additive) and ImplicitM (multiplicative) are learnable
    channel-wise feature adapters that improve detection head robustness
    in noisy sonar backgrounds by adapting feature distributions.
"""
import torch
import torch.nn as nn


class ImplicitA(nn.Module):
    """
    Implicit Additive Feature.

    Learnable additive vector broadcast over spatial dimensions.
    Initialized ~N(0, 0.02) so initial effect is minimal.

    SOURCE: Zhang & Gao (2025), ImplicitHead section; YOLOR.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.normal_(self.implicit, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.implicit


class ImplicitM(nn.Module):
    """
    Implicit Multiplicative Feature.

    Learnable multiplicative vector broadcast over spatial dimensions.
    Initialized ~N(1, 0.02) so initial effect is near identity.

    SOURCE: Zhang & Gao (2025), ImplicitHead section; YOLOR.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, channels, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.implicit
