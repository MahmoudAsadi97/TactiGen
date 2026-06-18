"""
Evaluate generated coaching reports.
Verifies all six section headers are present, computes a groundedness score
(fraction of sentences referencing a number or timestamp), counts hallucination
flags from analyst feedback, and checks that confidence values are present.
Saved to outputs/run_logs/eval_reports_<clip_id>.json.
Run: python evaluation/evaluate_reports.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path
from loguru import logger

RUN_LOGS = Path("outputs/run_logs")
REPORTS = Path("outputs/reports")
FEEDBACK = Path("outputs/feedback")

SECTION_MARKERS = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]


def latest_clip_id(default: str = "synthetic_clip_001") -> str:
    logs = sorted(p for p in RUN_LOGS.glob("*.json") if not p.name.startswith("eval_"))
    if logs:
        try:
            return json.loads(logs[-1].read_text()).get("clip_id", default)
        except Exception:
            return default
    return default


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


def count_hallucination_flags(clip_id: str) -> int:
    if not FEEDBACK.exists():
        return 0
    count = 0
    for fp in FEEDBACK.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
            if data.get("hallucination_flag"):
                count += 1
        except Exception:
            continue
    return count


def evaluate() -> dict:
    clip_id = latest_clip_id()
    rp = REPORTS / f"{clip_id}_report.txt"
    report_text = rp.read_text(encoding="utf-8") if rp.exists() else ""

    sections_present = {m: (m in report_text) for m in SECTION_MARKERS}
    all_sections = all(sections_present.values())

    sentences = split_sentences(report_text)
    grounded = [s for s in sentences if re.search(r"\d", s)]
    groundedness = round(len(grounded) / max(len(sentences), 1), 3)

    confidence_present = "confidence" in report_text.lower()
    confidence_values = re.findall(r"[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)", report_text)

    result = {
        "module": "reports",
        "clip_id": clip_id,
        "report_found": rp.exists(),
        "six_sections_present": all_sections,
        "section_detail": sections_present,
        "sentences_evaluated": len(sentences),
        "groundedness_score": groundedness,
        "confidence_values_present": confidence_present,
        "confidence_values_found": confidence_values,
        "hallucination_flag_count": count_hallucination_flags(clip_id),
    }

    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    out = RUN_LOGS / f"eval_reports_{clip_id}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.success(f"Report evaluation saved to {out}")
    logger.info(f"  six_sections={all_sections}, groundedness={groundedness}, "
                f"confidence_present={confidence_present}")
    return result


if __name__ == "__main__":
    evaluate()
