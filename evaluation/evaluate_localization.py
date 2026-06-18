"""
Evaluate player localization quality.
Computes confidence distribution, detection count per frame, a precision/recall
stub (placeholders when no labelled ground truth is available), and a visual
inspection flag. Results saved to outputs/run_logs/eval_localization_<clip_id>.json.
Run: python evaluation/evaluate_localization.py
"""
import json
import statistics
from pathlib import Path
from loguru import logger

RUN_LOGS = Path("outputs/run_logs")
SYNTHETIC = Path("data/sample_clips/synthetic_clip_001.json")
LOCALIZATION_MIN = 0.65


def latest_clip_id(default: str = "synthetic_clip_001") -> str:
    logs = sorted(p for p in RUN_LOGS.glob("*.json") if not p.name.startswith("eval_"))
    if logs:
        try:
            return json.loads(logs[-1].read_text()).get("clip_id", default)
        except Exception:
            return default
    return default


def load_localization_records() -> list:
    from localization.player_localization import PlayerLocalizer
    loc = PlayerLocalizer()
    if SYNTHETIC.exists():
        return [r.model_dump() for r in loc.localize_from_synthetic(str(SYNTHETIC))]
    records = []
    for fid in range(5):
        records.extend(r.model_dump() for r in loc._synthetic_localize(fid, fid * 0.2, "synthetic"))
    return records


def evaluate() -> dict:
    clip_id = latest_clip_id()
    records = load_localization_records()
    confs = [r["confidence"] for r in records]

    per_frame = {}
    for r in records:
        per_frame[r["frame_id"]] = per_frame.get(r["frame_id"], 0) + 1

    conf_stats = {
        "count": len(confs),
        "mean": round(statistics.mean(confs), 4) if confs else 0.0,
        "std": round(statistics.pstdev(confs), 4) if len(confs) > 1 else 0.0,
        "min": round(min(confs), 4) if confs else 0.0,
        "max": round(max(confs), 4) if confs else 0.0,
        "median": round(statistics.median(confs), 4) if confs else 0.0,
    }
    mean_det = round(statistics.mean(list(per_frame.values())), 2) if per_frame else 0.0

    result = {
        "module": "localization",
        "clip_id": clip_id,
        "confidence_distribution": conf_stats,
        "frames_evaluated": len(per_frame),
        "mean_detections_per_frame": mean_det,
        "precision_at_iou_0_5": None,
        "recall_at_iou_0_5": None,
        "precision_recall_note": (
            "No ground-truth bounding boxes available; precision/recall require "
            "labelled SoccerNet frames. Placeholder values returned."
        ),
        "visual_inspection_required": conf_stats["mean"] < LOCALIZATION_MIN,
    }

    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    out = RUN_LOGS / f"eval_localization_{clip_id}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.success(f"Localization evaluation saved to {out}")
    logger.info(f"  mean conf={conf_stats['mean']} over {conf_stats['count']} detections, "
                f"{mean_det} detections/frame")
    return result


if __name__ == "__main__":
    evaluate()
