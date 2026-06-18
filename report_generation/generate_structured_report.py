"""Generate the structured JSON tactical report from the evidence object (no LLM needed)."""
import json
from pathlib import Path
from evidence.evidence_schema import EvidenceObject


def generate_structured_report(evidence: EvidenceObject, output_dir: str = "outputs/reports") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    m = evidence.supporting_metrics
    report = {
        "clip_summary": {
            "main_phase": "attacking_transition" if evidence.detected_pattern in
                          ["right_side_overload", "left_side_overload", "switch_of_play"]
                          else "out_of_possession",
            "primary_pattern": evidence.detected_pattern,
            "anticipated_action": evidence.anticipated_action.get("action", "pass"),
            "confidence": evidence.anticipated_action.get("confidence", 0.0)
        },
        "metrics": {
            "team_width": m.get("team_width_meters", 0),
            "compactness": m.get("compactness_score", 0),
            "overload_ratio": m.get("overload_ratio", 0),
            "defensive_line_height": m.get("defensive_line_height_meters", 0)
        },
        "evidence": [
            {
                "timestamp_start": float(evidence.time_window.split("-")[0].replace(":", ".")),
                "timestamp_end": float(evidence.time_window.split("-")[1].replace(":", ".")),
                "claim": evidence.detected_pattern.replace("_", " "),
                "support": (f"{m.get('attacking_count',0)} attackers vs "
                            f"{m.get('defending_count',0)} defenders "
                            f"in {m.get('overload_channel','unknown')} channel "
                            f"(ratio {m.get('overload_ratio',0):.2f})")
            }
        ],
        "recommendations": [],
        "limitations": evidence.limitations,
        "report_status": evidence.report_status
    }

    # Add recommendation based on pattern
    if "overload" in evidence.detected_pattern:
        report["recommendations"].append({
            "type": "defensive_adjustment",
            "text": ("The weak-side midfielder should shift earlier toward the ball side "
                     "to reduce the wide overload and provide defensive cover.")
        })
    elif evidence.detected_pattern == "broken_defensive_line":
        report["recommendations"].append({
            "type": "recovery_run",
            "text": "Defenders must compact the shape immediately and reduce space between lines."
        })
    elif evidence.detected_pattern == "compact_block":
        report["recommendations"].append({
            "type": "attacking_overload_exploitation",
            "text": "Switch play quickly to exploit the opposite flank against the compact block."
        })

    out_path = Path(output_dir) / f"{evidence.clip_id}_structured.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
