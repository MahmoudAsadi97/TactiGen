"""
Evaluate RAG retrieval and grounding.
Checks that k=4 passages are retrieved, measures citation coverage (how many
retrieved source_ids appear in the generated report), and the unsupported-claim
rate (report sentences containing no metric value and no source_id).
Saved to outputs/run_logs/eval_rag_<clip_id>.json.
Run: python evaluation/evaluate_rag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path
from loguru import logger

RUN_LOGS = Path("outputs/run_logs")
REPORTS = Path("outputs/reports")


def latest_clip_id(default: str = "synthetic_clip_001") -> str:
    logs = sorted(p for p in RUN_LOGS.glob("*.json") if not p.name.startswith("eval_"))
    if logs:
        try:
            return json.loads(logs[-1].read_text()).get("clip_id", default)
        except Exception:
            return default
    return default


def detected_pattern_for(clip_id: str) -> str:
    sr = REPORTS / f"{clip_id}_structured.json"
    if sr.exists():
        try:
            return json.loads(sr.read_text()).get("clip_summary", {}).get("primary_pattern",
                                                                          "general_possession")
        except Exception:
            pass
    return "general_possession"


def read_report_text(clip_id: str) -> str:
    rp = REPORTS / f"{clip_id}_report.txt"
    return rp.read_text(encoding="utf-8") if rp.exists() else ""


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


def evaluate(k: int = 4) -> dict:
    from rag.retrieve_context import TacticalRetriever

    clip_id = latest_clip_id()
    pattern = detected_pattern_for(clip_id)

    retriever = TacticalRetriever()
    passages = retriever.retrieve_for_pattern(pattern)
    source_ids = sorted({p.get("source_id", "unknown") for p in passages})

    report_text = read_report_text(clip_id)
    report_lower = report_text.lower()

    cited = [sid for sid in source_ids if sid.lower() in report_lower]
    citation_coverage = round(len(cited) / max(len(source_ids), 1), 3)

    sentences = split_sentences(report_text)
    has_number = re.compile(r"\d")
    unsupported = []
    for s in sentences:
        supported = bool(has_number.search(s)) or any(sid.lower() in s.lower() for sid in source_ids)
        if not supported:
            unsupported.append(s)
    unsupported_rate = round(len(unsupported) / max(len(sentences), 1), 3)

    result = {
        "module": "rag",
        "clip_id": clip_id,
        "query_pattern": pattern,
        "retrieved_count": len(passages),
        "expected_k": k,
        "retrieval_count_ok": len(passages) == k,
        "retrieved_source_ids": source_ids,
        "citation_coverage": citation_coverage,
        "cited_source_ids": cited,
        "sentences_evaluated": len(sentences),
        "unsupported_claim_rate": unsupported_rate,
        "note": ("Template reports rarely cite source_ids explicitly, so citation "
                 "coverage is expected to be higher when GPT-4o generation is enabled."),
    }

    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    out = RUN_LOGS / f"eval_rag_{clip_id}.json"
    out.write_text(json.dumps(result, indent=2))
    logger.success(f"RAG evaluation saved to {out}")
    logger.info(f"  retrieved={len(passages)} (k={k}), citation_coverage={citation_coverage}, "
                f"unsupported_rate={unsupported_rate}")
    return result


if __name__ == "__main__":
    evaluate()
