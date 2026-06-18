"""Baseline anticipation models for comparison."""
from action_anticipation.anticipation_schema import AnticipationResult, ACTION_CLASSES
from tactical_analysis.tactical_metrics_schema import TacticalMetrics


class MajorityClassBaseline:
    """Always predicts 'pass' — the most frequent action in football."""
    def predict(self, clip_id: str, timestamp: float) -> AnticipationResult:
        return AnticipationResult(
            clip_id=clip_id, timestamp=timestamp,
            predicted_action="pass", confidence=0.42,
            model_used="majority_class_baseline"
        )


class LastActionBaseline:
    """Predicts the same action as the last observed event."""
    def predict(self, clip_id: str, timestamp: float, last_action: str = "pass") -> AnticipationResult:
        return AnticipationResult(
            clip_id=clip_id, timestamp=timestamp,
            predicted_action=last_action, confidence=0.38,
            model_used="last_action_baseline"
        )


class HeuristicBaseline:
    """Rule-based anticipation using tactical metrics."""
    def predict(self, clip_id: str, timestamp: float, metrics: TacticalMetrics) -> AnticipationResult:
        if metrics.overload_ratio > 1.5 and metrics.overload_channel in ["right", "left"]:
            return AnticipationResult(
                clip_id=clip_id, timestamp=timestamp,
                predicted_action="cross", confidence=0.61,
                model_used="heuristic_baseline"
            )
        elif metrics.compactness_score < 0.4:
            return AnticipationResult(
                clip_id=clip_id, timestamp=timestamp,
                predicted_action="shot", confidence=0.55,
                model_used="heuristic_baseline"
            )
        elif metrics.defensive_line_height_meters > 50:
            return AnticipationResult(
                clip_id=clip_id, timestamp=timestamp,
                predicted_action="ball_progression", confidence=0.52,
                model_used="heuristic_baseline"
            )
        return AnticipationResult(
            clip_id=clip_id, timestamp=timestamp,
            predicted_action="pass", confidence=0.45,
            model_used="heuristic_baseline"
        )
