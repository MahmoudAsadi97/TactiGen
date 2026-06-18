"""RAG retrieval: find relevant tactical knowledge passages."""
import json
from pathlib import Path
from typing import List, Dict
from loguru import logger

INDEX_DIR = Path("knowledge/faiss_index")


class TacticalRetriever:
    def __init__(self):
        self._vectorstore = None
        self._docs_fallback: List[Dict] = []
        self._load()

    def _load(self):
        meta_path = INDEX_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._docs_fallback = json.load(f)

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self._vectorstore = FAISS.load_local(
                str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            logger.warning(f"FAISS load failed ({e}). Using keyword fallback.")

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if self._vectorstore:
            results = self._vectorstore.similarity_search(query, k=k)
            return [{"source_id": r.metadata.get("source_id", "unknown"),
                     "text": r.page_content} for r in results]
        # Keyword fallback
        query_lower = query.lower()
        scored = [(sum(w in d["text"].lower() for w in query_lower.split()), d)
                  for d in self._docs_fallback]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"source_id": d["source_id"], "text": d["text"]} for _, d in scored[:k]]

    def retrieve_for_pattern(self, pattern: str, phase: str = "") -> List[Dict]:
        query = f"{pattern.replace('_', ' ')} {phase.replace('_', ' ')} football tactics"
        return self.retrieve(query)
