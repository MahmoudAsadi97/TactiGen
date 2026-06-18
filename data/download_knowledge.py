"""
TactiGen Knowledge Base Verification Script.

The TactiGen tactical knowledge base is original work that ships with the
repository under knowledge/tactical_kb/. There is nothing to download from a
remote source; instead this script verifies that every expected document is
present, reports a per-document word count, and writes a manifest the RAG
index builder can rely on as a validated corpus.

Run: python data/download_knowledge.py
"""
import json
from pathlib import Path
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "knowledge" / "tactical_kb"
REGISTRY = REPO_ROOT / "knowledge" / "source_registry.csv"
MANIFEST = REPO_ROOT / "knowledge" / "kb_manifest.json"

EXPECTED_DOCS = [
    "wide_overload",
    "defensive_compactness",
    "high_press",
    "team_width",
    "overload_underload",
    "defensive_line_height",
    "attacking_transition",
    "defensive_transition",
    "cross_creation",
    "pressing_intensity",
]


def verify_knowledge_base() -> dict:
    """Verify all expected KB documents exist and build a manifest dict."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    documents = []
    missing = []

    for name in EXPECTED_DOCS:
        path = KB_DIR / f"{name}.txt"
        if not path.exists():
            missing.append(name)
            logger.warning(f"Missing knowledge document: {name}.txt")
            continue
        text = path.read_text(encoding="utf-8")
        word_count = len(text.split())
        documents.append({
            "source_id": name,
            "file": str(path.relative_to(REPO_ROOT)),
            "word_count": word_count,
        })
        logger.info(f"  [ok] {name}.txt ({word_count} words)")

    manifest = {
        "knowledge_base": "TactiGen tactical knowledge base (original work)",
        "registry": str(REGISTRY.relative_to(REPO_ROOT)) if REGISTRY.exists() else None,
        "expected": len(EXPECTED_DOCS),
        "found": len(documents),
        "missing": missing,
        "documents": documents,
    }
    return manifest


def main():
    logger.info("=== TactiGen Knowledge Base Verification ===")
    manifest = verify_knowledge_base()
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if manifest["missing"]:
        logger.warning(
            f"{len(manifest['missing'])} document(s) missing: {manifest['missing']}. "
            "Run the knowledge-base creation step before building the RAG index."
        )
    else:
        logger.success(
            f"All {manifest['found']} knowledge documents present. "
            f"Manifest written to {MANIFEST}"
        )


if __name__ == "__main__":
    main()
