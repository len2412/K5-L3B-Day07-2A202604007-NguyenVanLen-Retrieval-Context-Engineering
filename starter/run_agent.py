"""Run a question or an agent_cases split with Gemini and Day 6 strategy B."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from agentic import run_agent
from day6_retriever import DAY6_STARTER, Day6DenseRetriever
from gemini_controller import GeminiController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to answer from the Day 6 corpus")
    parser.add_argument("--cases", type=Path, help="Agent cases JSON; contains no gold answers")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3500, help="Maximum context characters")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional path to save the full JSON trace")
    args = parser.parse_args()
    if bool(args.question) == bool(args.cases):
        parser.error("provide exactly one question or --cases")
    if args.cases and not args.output:
        parser.error("--output is required with --cases")

    load_dotenv(DAY6_STARTER / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    retriever = Day6DenseRetriever()
    controller = GeminiController()
    config = {
        "strategy": "B",
        "embedding_model": retriever.model_name,
        "embedding_dimensions": retriever.embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector / ingestion_chunks",
        "gemini_model": controller.model,
    }
    if args.cases:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        selected = [case for case in cases if case.get("split") == args.split]
        if not selected:
            raise ValueError(f"No cases in split {args.split!r}: {args.cases}")
        traces = []
        for case in selected:
            query_id, question = case.get("id"), case.get("question")
            if not isinstance(query_id, str) or not isinstance(question, str) or not question.strip():
                raise ValueError("Each case must have a non-empty string id and question")
            started = time.perf_counter()
            try:
                trace = run_agent(
                    question,
                    controller,
                    retriever,
                    top_k=args.top_k,
                    budget=args.budget,
                    max_searches=args.max_searches,
                )
            except Exception as exc:
                trace = {"status": "error", "answer": "", "citations": [],
                         "events": [{"event": "error", "error": str(exc)}]}
            trace["query_id"] = query_id
            trace["split"] = args.split
            trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
            traces.append(trace)
        result = {"split": args.split, "retrieval_config": config, "traces": traces}
    else:
        result = run_agent(
            args.question,
            controller,
            retriever,
            top_k=args.top_k,
            budget=args.budget,
            max_searches=args.max_searches,
        )
        result["retrieval_config"] = config
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
