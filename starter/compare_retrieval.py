"""Benchmark dense B, TF-IDF, and BGE-reranked dense retrieval on one split."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from baseline import TfidfIndex, hit, pack_context, validate
from day6_retriever import DAY6_STARTER, Day6DenseRetriever


def load_strategy_b_chunks() -> list[dict]:
    """Recreate the exact Day 6 B chunks from its PDF and manifest for lexical search."""
    from src.chunking import by_paragraph
    from src.ingestion import load_pdf

    day6_root = DAY6_STARTER.parent
    manifest_path = day6_root / "data" / "manifest.json"
    pdf_path = day6_root / "data" / "progit.pdf"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    pages = load_pdf(pdf_path, manifest)
    chunks = [chunk for page in pages for chunk in by_paragraph(page, max_size=1200)]
    return [
        {
            "id": chunk.chunk_id,
            "source_id": chunk.document_id,
            "source": chunk.source,
            "page": chunk.page,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def pack_results(question: dict, method: str, results: list[dict], elapsed_ms: float,
                 top_k: int, budget: int, candidate_k: int | None = None) -> dict:
    results = results[:top_k]
    marked = [{**row, "hit": hit(row, question["gold"])} for row in results]
    context, context_ids = pack_context(results, budget=budget)
    return {
        "query_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "method": method,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "budget_chars": budget,
        "results": marked,
        "context": context,
        "context_ids": context_ids,
        "context_chars": len(context),
        "elapsed_ms": elapsed_ms,
        "hit1": bool(marked and marked[0]["hit"]),
        "evidence_hit3": any(row["hit"] for row in marked),
        "context_evidence": any(row["hit"] and row["id"] in context_ids for row in marked),
        # Gold is attached only after retrieval/context construction for offline scoring.
        "gold": question["gold"],
    }


def summarize(method: str, traces: list[dict], setup_seconds: float, top_k: int,
              budget: int, candidate_k: int | None = None) -> dict:
    n = len(traces)
    counts = {
        "hit1": sum(row["hit1"] for row in traces),
        "evidence_hit3": sum(row["evidence_hit3"] for row in traces),
        "context_evidence": sum(row["context_evidence"] for row in traces),
    }
    latencies = [row["elapsed_ms"] for row in traces]
    return {
        "method": method,
        "split": traces[0]["split"] if traces else None,
        "n_answerable": n,
        "configuration": {"top_k": top_k, "context_budget_chars": budget,
                          "candidate_k": candidate_k},
        "metrics": {
            name: {"n": value, "N": n, "rate": value / n if n else None}
            for name, value in counts.items()
        },
        "latency_ms_per_query": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "total": sum(latencies) if latencies else 0.0,
        },
        "setup_seconds": setup_seconds,
        "service_cost_usd": 0.0,
        "cost_note": "Local embedding, TF-IDF, pgvector, and reranker inference; electricity/local compute not priced.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("../data/benchmark_questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=Path("results/retrieval-dev"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Dense candidates for reranking; must be >= top-k")
    parser.add_argument("--budget", type=int, default=3500, help="Context character budget")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-base")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k or args.budget < 1:
        parser.error("require top-k >= 1, candidate-k >= top-k, and budget >= 1")

    starter_dir = Path(__file__).resolve().parent
    load_dotenv(DAY6_STARTER / ".env", override=False)
    load_dotenv(starter_dir / ".env", override=False)
    benchmark = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in benchmark if q.get("split") == args.split and q.get("answerable", True)]
    if not questions:
        raise ValueError(f"No answerable questions for split {args.split!r}")

    # Recreate the same deterministic B chunks for lexical retrieval and evidence checks.
    chunks = load_strategy_b_chunks()
    validate(chunks, questions)

    started = time.perf_counter()
    tfidf = TfidfIndex(chunks)
    tfidf_setup_seconds = time.perf_counter() - started

    started = time.perf_counter()
    dense = Day6DenseRetriever()
    dense_setup_seconds = time.perf_counter() - started

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "Install the Day 6 sentence-transformers dependency to use the BGE reranker."
        ) from exc
    started = time.perf_counter()
    reranker = CrossEncoder(args.reranker, device=None)
    reranker_setup_seconds = time.perf_counter() - started

    traces_by_method = {"dense_b": [], "tfidf": [], "dense_b_rerank": []}
    for question in questions:
        start = time.perf_counter()
        rows = dense.search(question["question"], filters=question.get("filter"), top_k=args.top_k)
        traces_by_method["dense_b"].append(
            pack_results(question, "dense_b", rows, (time.perf_counter() - start) * 1000,
                         args.top_k, args.budget)
        )

        start = time.perf_counter()
        rows = tfidf.search(question["question"], top_k=args.top_k)
        traces_by_method["tfidf"].append(
            pack_results(question, "tfidf", rows, (time.perf_counter() - start) * 1000,
                         args.top_k, args.budget)
        )

        start = time.perf_counter()
        candidates = dense.search(question["question"], filters=question.get("filter"),
                                  top_k=args.candidate_k)
        pairs = [(question["question"], row["text"]) for row in candidates]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False) if pairs else []
        reranked = [
            {**row, "dense_score": row["score"], "score": float(score),
             "rerank_score": float(score), "strategy": "dense_b_rerank"}
            for row, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda row: (-row["rerank_score"], row["id"]))
        traces_by_method["dense_b_rerank"].append(
            pack_results(question, "dense_b_rerank", reranked,
                         (time.perf_counter() - start) * 1000, args.top_k, args.budget,
                         candidate_k=args.candidate_k)
        )

    setup = {"dense_b": dense_setup_seconds, "tfidf": tfidf_setup_seconds,
             "dense_b_rerank": dense_setup_seconds + reranker_setup_seconds}
    summaries = [
        summarize(method, traces, setup[method], args.top_k, args.budget,
                  args.candidate_k if method == "dense_b_rerank" else None)
        for method, traces in traces_by_method.items()
    ]
    comparison = {
        "split": args.split,
        "corpus": {"source_id": "progit-v2-1-450", "day6_strategy": "B",
                   "chunking": "paragraph-aware, max 1200 characters", "chunk_count": len(chunks)},
        "embedding_model": dense.model_name,
        "embedding_dimensions": dense.embedder.dimensions,
        "reranker_model": args.reranker,
        "reranker_candidates": args.candidate_k,
        "methods": summaries,
        "comparison_note": "Same answerable questions, corpus, B chunks, top-k, context budget, and hit definition for every method.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for method, traces in traces_by_method.items():
        (args.output / f"{method}-traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Retrieval comparison saved to: {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
