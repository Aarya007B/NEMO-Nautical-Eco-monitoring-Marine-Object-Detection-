"""
scoring/priority.py — Priority score computation.

Priority answers: "How urgently should an operator review this detection?"
This is DIFFERENT from artificialness (PRD §25).
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PriorityScorer:
    """
    Weighted combination into an operator priority score.

    Args:
        weights: Dict with keys artificialness_weight,
                 mission_confidence_weight, metadata_quality_weight.
    """

    def __init__(self, weights: Dict[str, float]):
        self.w_art   = float(weights.get("artificialness_weight", 0.70))
        self.w_conf  = float(weights.get("mission_confidence_weight", 0.20))
        self.w_meta  = float(weights.get("metadata_quality_weight", 0.10))

    def score(
        self,
        artificialness_score: float,
        detector_confidence: float,
        has_gps: bool = False,
        has_timestamp: bool = False,
    ) -> float:
        """
        Compute priority score.

        Args:
            artificialness_score: From ArtificialnessScorer.
            detector_confidence: Raw detector confidence.
            has_gps: Whether GPS metadata is available.
            has_timestamp: Whether timestamp metadata is available.

        Returns:
            Priority score in [0, 1].
        """
        metadata_quality = 0.0
        if has_gps:
            metadata_quality += 0.6
        if has_timestamp:
            metadata_quality += 0.4

        raw = (
            self.w_art  * artificialness_score
            + self.w_conf * detector_confidence
            + self.w_meta * metadata_quality
        )
        return max(0.0, min(1.0, raw))
