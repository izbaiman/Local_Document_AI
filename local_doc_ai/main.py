#!/usr/bin/env python3
"""
Local Document AI — Main CLI Entry Point
=========================================
Run classification, extraction, and semantic search on a folder of documents.

Usage:
    python main.py process --folder ./sample_docs
    python main.py search  --query "payments due in January"
    python main.py qa      --query "What is the total amount on invoice 1?" (optional bonus)
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import DocumentPipeline
from src.retrieval import RetrievalEngine
from src.utils import print_banner, print_results_table


def cmd_process(args):
    """Ingest, classify, and extract structured data from a document folder."""
    folder = Path(args.folder)
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = DocumentPipeline(
        model_name=args.embedding_model,
        use_zero_shot=args.use_zero_shot,
        zero_shot_model=args.zero_shot_model,
    )

    print(f"\n[INFO] Processing documents in: {folder}")
    results = pipeline.run(folder)

    # Write output.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Output written to: {output_path}")

    # Pretty-print summary table
    print_results_table(results)

    # Build / refresh retrieval index
    print("\n[INFO] Building semantic search index …")
    engine = RetrievalEngine(model_name=args.embedding_model)
    engine.build_index(pipeline.documents)
    engine.save(args.index_path)
    print(f"[OK] Index saved to: {args.index_path}")


def cmd_search(args):
    """Search previously indexed documents by meaning."""
    index_path = Path(args.index_path)
    if not index_path.exists():
        print("[ERROR] No index found. Run `process` first.")
        sys.exit(1)

    engine = RetrievalEngine(model_name=args.embedding_model)
    engine.load(args.index_path)

    print(f"\n[SEARCH] Query: \"{args.query}\"\n")
    hits = engine.search(args.query, top_k=args.top_k)

    if not hits:
        print("No results found.")
        return

    for rank, hit in enumerate(hits, 1):
        print(f"  [{rank}] {hit['filename']}  (score: {hit['score']:.4f})")
        print(f"       Class : {hit['doc_class']}")
        print(f"       Snippet: {hit['snippet']}\n")


def cmd_qa(args):
    """Optional bonus: local QA over retrieved documents using an open-source LLM."""
    try:
        from src.qa import LocalQAEngine
    except ImportError as exc:
        print(f"[ERROR] QA module unavailable: {exc}")
        sys.exit(1)

    index_path = Path(args.index_path)
    if not index_path.exists():
        print("[ERROR] No index found. Run `process` first.")
        sys.exit(1)

    retrieval = RetrievalEngine(model_name=args.embedding_model)
    retrieval.load(args.index_path)

    qa_engine = LocalQAEngine(llm_model=args.llm_model)
    hits = retrieval.search(args.query, top_k=args.top_k)
    context_docs = [h["text"] for h in hits]

    print(f"\n[QA] Question: \"{args.query}\"")
    answer = qa_engine.answer(args.query, context_docs)
    print(f"\n[ANSWER]\n{answer}\n")


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_doc_ai",
        description="Local AI document processing pipeline (no paid APIs).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── process ──
    p = sub.add_parser("process", help="Ingest, classify and extract documents")
    p.add_argument("--folder",    default="./sample_docs", help="Input folder with PDFs/TXTs")
    p.add_argument("--output",    default="./output/output.json", help="Path to write output.json")
    p.add_argument("--index-path",default="./output/retrieval.index", help="Where to save the FAISS index")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                   help="SentenceTransformers model name for embeddings")
    p.add_argument("--use-zero-shot", action="store_true",
                   help="Use a Hugging Face zero-shot model for classification fallback")
    p.add_argument("--zero-shot-model", default="typeform/distilbert-base-uncased-mnli",
                   help="Zero-shot classification model (only used with --use-zero-shot)")

    # ── search ──
    s = sub.add_parser("search", help="Semantic search over indexed documents")
    s.add_argument("--query",     required=True, help="Natural-language search query")
    s.add_argument("--top-k",     type=int, default=5, help="Number of results to return")
    s.add_argument("--index-path",default="./output/retrieval.index")
    s.add_argument("--embedding-model", default="all-MiniLM-L6-v2")

    # ── qa (optional bonus) ──
    q = sub.add_parser("qa", help="[BONUS] Local QA using an open-source LLM")
    q.add_argument("--query",     required=True, help="Question to answer")
    q.add_argument("--llm-model", default="google/flan-t5-base",
                   help="HuggingFace model for answer generation")
    q.add_argument("--top-k",     type=int, default=3)
    q.add_argument("--index-path",default="./output/retrieval.index")
    q.add_argument("--embedding-model", default="all-MiniLM-L6-v2")

    return parser


if __name__ == "__main__":
    print_banner()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {"process": cmd_process, "search": cmd_search, "qa": cmd_qa}
    dispatch[args.command](args)
