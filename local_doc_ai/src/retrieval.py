"""
retrieval.py
============
Semantic search over ingested documents using:
    • SentenceTransformers  — for dense vector embeddings
    • FAISS                 — for fast approximate nearest-neighbour search

All models run locally; no internet or API keys required.

Index serialisation:
    retrieval.index/
        faiss.index        — FAISS flat index
        metadata.json      — per-chunk metadata (filename, class, snippet, text)
        config.json        — model name used to build the index
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Chunk size in characters — long documents are split to improve retrieval
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str      # "{filename}::chunk_{n}"
    filename: str
    doc_class: str
    text: str
    snippet: str       # first 200 chars for display


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping fixed-size character windows."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


# ── Retrieval engine ───────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Builds and queries a semantic search index over document chunks.

    Usage:
        engine = RetrievalEngine()
        engine.build_index(documents)       # list of ingestion.Document
        engine.save("./output/retrieval.index")

        engine2 = RetrievalEngine()
        engine2.load("./output/retrieval.index")
        results = engine2.search("payments due in January", top_k=5)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None          # FAISS index
        self._chunks: List[Chunk] = []

    # ── Model lazy-load ────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer: %s …", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for retrieval.\n"
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for retrieval.\n"
                "Install with: pip install faiss-cpu"
            )

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self, documents) -> None:
        """
        Build FAISS index from a list of ingestion.Document objects.

        Args:
            documents: list of Document (must have .filename, .clean_text, and .doc_class
                       OR we accept dicts with the same keys).
        """
        model = self._get_model()
        faiss = self._get_faiss()

        self._chunks = []
        texts_to_embed = []

        for doc in documents:
            # Accept both Document objects and dicts
            filename  = doc.filename  if hasattr(doc, "filename")  else doc["filename"]
            text      = doc.clean_text if hasattr(doc, "clean_text") else doc.get("clean_text", "")
            doc_class = doc.doc_class  if hasattr(doc, "doc_class")  else doc.get("doc_class", "Unknown")

            if not text:
                continue

            raw_chunks = _chunk_text(text)
            for i, chunk_text in enumerate(raw_chunks):
                chunk = Chunk(
                    chunk_id=f"{filename}::chunk_{i}",
                    filename=filename,
                    doc_class=doc_class,
                    text=chunk_text,
                    snippet=chunk_text[:200].replace("\n", " ").strip(),
                )
                self._chunks.append(chunk)
                texts_to_embed.append(chunk_text)

        if not texts_to_embed:
            logger.warning("No text to embed — index will be empty.")
            return

        logger.info("Embedding %d chunks …", len(texts_to_embed))
        embeddings = model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
        self._index.add(embeddings)
        logger.info("FAISS index built — %d vectors, dim=%d", self._index.ntotal, dim)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, filename, doc_class, snippet,
            text, score.  Sorted by score descending.
        """
        if self._index is None or not self._chunks:
            raise RuntimeError("Index not built or loaded. Call build_index() or load() first.")

        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)

        results = []
        seen_files = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]

            # De-duplicate: keep only the best chunk per file
            if chunk.filename in seen_files:
                continue
            seen_files.add(chunk.filename)

            results.append({
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "doc_class": chunk.doc_class,
                "snippet": chunk.snippet,
                "text": chunk.text,
                "score": float(score),
            })

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        """Persist the FAISS index and metadata to disk."""
        faiss = self._get_faiss()
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path / "faiss.index"))

        # Save metadata
        metadata = [asdict(c) for c in self._chunks]
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump({"model_name": self.model_name}, f)

        logger.info("Index saved to %s  (%d chunks)", path, len(self._chunks))

    def load(self, index_dir: str) -> None:
        """Load a previously saved index from disk."""
        faiss = self._get_faiss()
        path = Path(index_dir)

        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            loaded_model = cfg.get("model_name", self.model_name)
            if loaded_model != self.model_name:
                logger.warning(
                    "Index was built with model '%s' but current model is '%s'. "
                    "Results may be degraded.",
                    loaded_model, self.model_name
                )
            self.model_name = loaded_model

        self._index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "metadata.json", encoding="utf-8") as f:
            raw = json.load(f)
        self._chunks = [Chunk(**c) for c in raw]

        logger.info("Index loaded from %s  (%d chunks)", path, len(self._chunks))
