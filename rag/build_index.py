"""
Build FAISS vector index from tactical knowledge base documents.
Run: python rag/build_index.py
"""
import json
import os
from pathlib import Path
from loguru import logger

KNOWLEDGE_DIR = Path("knowledge/tactical_kb")
INDEX_DIR = Path("knowledge/faiss_index")


def build_tactical_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load documents
    docs = []
    for txt_file in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        source_id = txt_file.stem
        # Chunk into 400-char segments with 80-char overlap
        chunk_size, overlap = 400, 80
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 50:
                docs.append({"source_id": source_id, "chunk_index": i, "text": chunk})

    if not docs:
        logger.error("No documents found in knowledge/tactical_kb/. Run Step 3 first.")
        return

    logger.info(f"Building FAISS index from {len(docs)} chunks...")

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        lc_docs = [
            Document(page_content=d["text"], metadata={"source_id": d["source_id"], "chunk_index": d["chunk_index"]})
            for d in docs
        ]
        vectorstore = FAISS.from_documents(lc_docs, embeddings)
        vectorstore.save_local(str(INDEX_DIR))
        logger.success(f"FAISS index saved to {INDEX_DIR}")

    except ImportError as e:
        logger.warning(f"LangChain/FAISS unavailable ({e}). Saving docs as JSON fallback.")

    # Always save metadata JSON
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(docs, f, indent=2)
    logger.success(f"Chunk metadata saved: {len(docs)} chunks")


if __name__ == "__main__":
    build_tactical_index()
