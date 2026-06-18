"""Generate the natural-language coaching report using GPT-4o or template fallback."""
import json
import os
from pathlib import Path
from loguru import logger
from evidence.evidence_schema import EvidenceObject


def generate_text_report(
    evidence: EvidenceObject,
    retrieved_passages: list,
    structured_report: dict,
    ontology: dict,
    output_dir: str = "outputs/reports"
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        report = _generate_with_openai(evidence, retrieved_passages, ontology, api_key)
    else:
        logger.warning("OPENAI_API_KEY not set — using template-based report generation.")
        report = _generate_template_report(evidence, retrieved_passages, structured_report)

    out_path = Path(output_dir) / f"{evidence.clip_id}_report.txt"
    out_path.write_text(report, encoding="utf-8")
    logger.success(f"Report saved to {out_path}")
    return report


def _generate_with_openai(evidence: EvidenceObject, passages: list, ontology: dict, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_prompt_path = Path("report_generation/prompts/system_prompt.txt")
    template_path = Path("report_generation/prompts/report_prompt_template.txt")
    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""
    template = template_path.read_text() if template_path.exists() else "{evidence_json}"

    passage_text = "\n\n".join(
        f"[{p.get('source_id','unknown')}] {p.get('text','')}" for p in passages
    )
    valid_patterns = ", ".join(
        ontology.get("attacking_patterns", []) + ontology.get("defensive_patterns", [])
    )
    valid_phases = ", ".join(ontology.get("phases", []))

    # NOTE: the template intentionally contains a literal JSON example with { } braces,
    # so str.format() cannot be used here. We substitute placeholders explicitly instead.
    user_prompt = (
        template
        .replace("{clip_id}", evidence.clip_id)
        .replace("{time_window}", evidence.time_window)
        .replace("{evidence_json}", evidence.model_dump_json(indent=2))
        .replace("{retrieved_passages}", passage_text)
        .replace("{valid_patterns}", valid_patterns)
        .replace("{valid_phases}", valid_phases)
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def _generate_template_report(evidence: EvidenceObject, passages: list, structured: dict) -> str:
    m = evidence.supporting_metrics
    ant = evidence.anticipated_action
    low_conf = evidence.report_status == "low_confidence_review"
    header = "LOW CONFIDENCE WARNING: This clip requires manual review.\n\n" if low_conf else ""
    passage_summary = " ".join(p.get("text", "")[:100] for p in passages[:2])
    recs = structured.get("recommendations", [])
    rec_text = recs[0]["text"] if recs else "No specific recommendation generated."

    return f"""{header}TACTIGEN TACTICAL REPORT — Clip: {evidence.clip_id}
Time Window: {evidence.time_window}
{'=' * 60}

(1) OBSERVED FACTS
Between {evidence.time_window}, the system detected a {evidence.detected_pattern.replace('_', ' ')} pattern.
{m.get('attacking_count', 0)} attacking players were observed against {m.get('defending_count', 0)} defending players
in the {m.get('overload_channel', 'unknown')} channel during this window.
Supporting frames: {', '.join(evidence.supporting_frames[:2]) or 'N/A'}

(2) COMPUTED TACTICAL METRICS
Team Width: {m.get('team_width_meters', 0):.1f} m
Compactness Score: {m.get('compactness_score', 0):.3f}
Defensive Line Height: {m.get('defensive_line_height_meters', 0):.1f} m
Overload Ratio: {m.get('overload_ratio', 0):.2f} ({m.get('attacking_count',0)} vs {m.get('defending_count',0)} in {m.get('overload_channel','unknown')} channel)

(3) MODEL PREDICTIONS
The action anticipation model predicted '{ant.get('action', 'unknown')}' within the next
{ant.get('window_seconds', 5)} seconds with a confidence of {ant.get('confidence', 0):.2f}
(Model: {ant.get('model', 'unknown')}).

(4) TACTICAL INTERPRETATION
The detected pattern '{evidence.detected_pattern.replace('_', ' ')}' is consistent with established football
tactical concepts. Retrieved knowledge: {passage_summary}...

(5) COACHING RECOMMENDATION
{rec_text}

(6) CONFIDENCE AND LIMITATIONS
Localization confidence: {evidence.model_confidence.get('localization', 0):.3f}
Action anticipation confidence: {evidence.model_confidence.get('action_anticipation', 0):.3f}
Tactical pattern confidence: {evidence.model_confidence.get('tactical_pattern', 0):.3f}
Limitations: {'; '.join(evidence.limitations) if evidence.limitations else 'None reported.'}
"""
