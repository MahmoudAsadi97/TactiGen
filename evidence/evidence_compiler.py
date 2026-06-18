"""
Evidence Compiler — assembles all pipeline outputs into a structured EvidenceObject.
This is the most critical component: it prevents the LLM from inventing unsupported claims.
"""
from typing import List, Tuple, Dict
from evidence.evidence_schema import EvidenceObject
from tactical_analysis.tactical_metrics_schema import TacticalMetrics
from action_anticipation.anticipation_schema import AnticipationResult
from loguru import logger

CONFIDENCE_THRESHOLDS = {
    "localization_min": 0.65,
    "anticipation_min": 0.50,
    "tactical_pattern_min": 0.50
}


def detect_tactical_pattern(metrics: TacticalMetrics) -> Tuple[str, float]:
    """
    Map tactical metrics to an ontology pattern label and confidence score.
    Returns: (pattern_label, confidence)
    """
    if metrics.overload_ratio > 1.5 and metrics.overload_channel == "right":
        return "right_side_overload", min(metrics.overload_ratio / 3.0, 0.95)
    elif metrics.overload_ratio > 1.5 and metrics.overload_channel == "left":
        return "left_side_overload", min(metrics.overload_ratio / 3.0, 0.95)
    elif metrics.compactness_score < 0.4:
        return "broken_defensive_line", 0.70
    elif metrics.compactness_score > 0.8 and metrics.defensive_line_height_meters < 30:
        return "compact_block", 0.75
    elif metrics.team_width_meters > 55:
        return "switch_of_play", 0.65
    else:
        return "general_possession", 0.50


def check_confidence_gate(
    localization_confidence: float,
    anticipation_confidence: float,
    tactical_pattern_confidence: float
) -> Tuple[bool, str]:
    """
    Returns (passes_gate, message).
    If any confidence is below threshold, the pipeline issues a low-confidence warning.
    """
    if localization_confidence < CONFIDENCE_THRESHOLDS["localization_min"]:
        return False, (
            f"Localization confidence ({localization_confidence:.2f}) is below the "
            f"minimum threshold ({CONFIDENCE_THRESHOLDS['localization_min']}). "
            "This clip requires manual review before tactical conclusions are made."
        )
    if tactical_pattern_confidence < CONFIDENCE_THRESHOLDS["tactical_pattern_min"]:
        return False, (
            f"Tactical pattern confidence ({tactical_pattern_confidence:.2f}) is too low "
            "for reliable interpretation. Manual review recommended."
        )
    return True, "Confidence thresholds met."


class EvidenceCompiler:
    def compile(
        self,
        clip_id: str,
        time_window: Tuple[float, float],
        metrics: TacticalMetrics,
        localization_confidence: float,
        anticipation_result: AnticipationResult,
        supporting_frame_paths: List[str],
        limitations: List[str]
    ) -> EvidenceObject:
        pattern, pattern_confidence = detect_tactical_pattern(metrics)
        passes_gate, gate_message = check_confidence_gate(
            localization_confidence,
            anticipation_result.confidence,
            pattern_confidence
        )
        if not passes_gate:
            logger.warning(f"Confidence gate FAILED for {clip_id}: {gate_message}")
            limitations.append(gate_message)

        tw_str = f"{int(time_window[0] // 60):02d}:{int(time_window[0] % 60):02d}" \
                 f"-{int(time_window[1] // 60):02d}:{int(time_window[1] % 60):02d}"

        return EvidenceObject(
            clip_id=clip_id,
            time_window=tw_str,
            detected_pattern=pattern,
            pattern_confidence=round(pattern_confidence, 3),
            supporting_metrics={
                "team_width_meters": metrics.team_width_meters,
                "compactness_score": metrics.compactness_score,
                "defensive_line_height_meters": metrics.defensive_line_height_meters,
                "overload_channel": metrics.overload_channel,
                "attacking_count": metrics.attacking_count,
                "defending_count": metrics.defending_count,
                "overload_ratio": metrics.overload_ratio
            },
            model_confidence={
                "localization": round(localization_confidence, 3),
                "action_anticipation": round(anticipation_result.confidence, 3),
                "tactical_pattern": round(pattern_confidence, 3)
            },
            supporting_frames=[str(f) for f in supporting_frame_paths[:3]],
            anticipated_action={
                "action": anticipation_result.predicted_action,
                "confidence": anticipation_result.confidence,
                "window_seconds": anticipation_result.anticipation_window_seconds,
                "model": anticipation_result.model_used
            },
            limitations=limitations,
            report_status="ok" if passes_gate else "low_confidence_review"
        )
