"""
scoring/artificialness.py — Artificialness score computation.

Combines detector confidence, verifier probability, and acoustic features
into a single artificialness score in [0, 1].

This score answers: "How strongly does the evidence suggest this is a
man-made object?"
"""
import logging
from typing import Dict

from inference.types import AcousticFeatures

logger = logging.getLogger(__name__)


class ArtificialnessScorer:
    """
    Weighted combination of evidence sources into an artificialness score.

    Args:
        weights: Dict with keys detector_weight, verifier_weight,
                 contrast_weight, geometry_weight.
    """

    def __init__(self, weights: Dict[str, float]):
        self.w_det  = float(weights.get("detector_weight", 0.35))
        self.w_ver  = float(weights.get("verifier_weight", 0.45))
        self.w_cont = float(weights.get("contrast_weight", 0.10))
        self.w_geom = float(weights.get("geometry_weight", 0.10))
        total = self.w_det + self.w_ver + self.w_cont + self.w_geom
        if abs(total - 1.0) > 1e-4:
            logger.warning("Artificialness weights sum to %.4f, expected 1.0", total)

    def score(
        self,
        detector_confidence: float,
        artificial_probability: float,
        features: AcousticFeatures,
    ) -> float:
        """
        Compute weighted artificialness score.

        Returns:
            Score in [0, 1].
        """
        raw = (
            self.w_det  * detector_confidence
            + self.w_ver  * artificial_probability
            + self.w_cont * features.local_contrast
            + self.w_geom * (1.0 - abs(features.aspect_ratio - 1.0) / max(features.aspect_ratio, 1.0))
        )
        return max(0.0, min(1.0, raw))
