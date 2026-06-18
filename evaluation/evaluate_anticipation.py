"""
Evaluate action anticipation.
Builds a baseline comparison table (transformer vs majority vs last-action vs
heuristic), reports top-1 accuracy when ground truth is available, and the class
distribution of predictions. Saved to outputs/run_logs/eval_anticipation_<clip_id>.json.
Run: python evaluation/evaluate_anticipation.py
"""
import json
from collections import Counter
from pathlib import Path
from loguru import logger

RUN_LOGS = Path("outputs/run_logs")
SYNTHETIC = Path("data/sample_clips/synthetic_clip_001.json")


def latest_clip_id(default: str = "synthetic_clip_001") -> str:
    logs = sorted(p for p in RUN_LOGS.glob("*.json") if not p.name.startswith("eval_"))
    if logs:
        try:
            return json.loads(logs[-1].read_text()).get("clip_id", default)
        except Exception:
            return default
    return default


def build_metrics(clip_id: str):
    from localization.player_localization import PlayerLocalizer
    from tactical_analysis.team_width import compute_team_width
    from tactical_analysis.compactness import compute_compactness
    from tactical_analysis.defensive_line import compute_defensive_line_height
    from tactical_analysis.overload_detection import detect_worst_overload
    from tactical_analysis.tactical_metrics_schema import TacticalMetrics

    loc = PlayerLocalizer()
    records = ([r.model_dump() for r in loc.localize_from_synthetic(str(SYNTHETIC))]
               if SYNTHETIC.exists() else [])
    positions = [(r["x_world"], r["y_world"]) for r in records if r.get("x_world") is not None]
    mid = 52.5
    team_a = [p for p in positions if p[0] >= mid]
    team_b = [p for p in positions if p[0] < mid]
    ov = detect_worst_overload(team_a, team_b)
    return TacticalMetrics(
        clip_id=clip_id, time_window_start=0.0, time_window_end=5.0,
        team_width_meters=compute_team_width(team_a),
        compactness_score=compute_compactness(team_b),
        defensive_line_height_meters=compute_defensive_line_height(team_b),
        overload_channel=ov["channel"], attacking_count=ov["attacking_count"],
        defending_count=ov["defending_count"], overload_ratio=ov["overload_ratio"],
    )


def evaluate() -> dict:
    from action_anticipation.baseline_models import (
        MajorityClassBaseline, LastActionBaseline, HeuristicBaseline)

    clip_id = latest_clip_id()
    metrics = build_metrics(clip_id)

    table = {}
    table["majority_class"] = MajorityClassBaseline().predict(clip_id, 0.0).model_dump()
    table["last_action"] = LastActionBaseline().predict(clip_id, 0.0).model_dump()
    table["heuristic"] = HeuristicBaseline().predict(clip_id, 0.0, metrics).model_dump()

    # Transformer prediction — guarded because pretrained weights/torch may be unavailable.
    try:
        from action_anticipation.faantra_inference import FAAntraInference
        table["transformer"] = FAAntraInference().predict(clip_id, [], 0.0, metrics).model_dump()
    except Exception as e:
        logger.warning(f"Transformer unavailable ({e}); recording heuristic fallback.")
        fallback = HeuristicBaseline().predict(clip_id, 0.0, metrics).model_dump()
        fallback["model_used"] = "transformer_unavailable_fallback_heuristic"
        fallback["error"] = str(e)
        table["transformer"] = fallback

    predictions = [v["predicted_action"] for v in table.values()]
    class_distribution = dict(Counter(predictions))

    result = {
        "module": "anticipation",
        "clip_id": clip_id,
        "baseline_comparison": {
            name: {"predicted_action": v["predicted_action"],
                   "confidence": v["confidence"],
                   "model_used": v["model_used"]}
            for name, v in table.items()
        },
        "top1_accuracy": None,
        "accuracy_note": ("No ground-truth next-action labels available; top-1 accuracy "
                          "requires aligned SoccerNet action-spotting labels."),
        "prediction_class_distribution": class_distribution,
    }

    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    out = RUN_LOGS / f"eval_anticipation_{clip_id}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.success(f"Anticipation evaluation saved to {out}")
    logger.info(f"  predictions: {class_distribution}")
    return result


if __name__ == "__main__":
    evaluate()
