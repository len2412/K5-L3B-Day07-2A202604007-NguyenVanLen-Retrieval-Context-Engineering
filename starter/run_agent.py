"""Run one Day 7 question using Gemini plus the Day 6 strategy B vector index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agentic import run_agent
from day6_retriever import DAY6_STARTER, Day6DenseRetriever
from gemini_controller import GeminiController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer from the Day 6 corpus")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=3500, help="Maximum context characters")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional path to save the full JSON trace")
    args = parser.parse_args()

    load_dotenv(DAY6_STARTER / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    retriever = Day6DenseRetriever()
    controller = GeminiController()
    result = run_agent(
        args.question,
        controller,
        retriever,
        top_k=args.top_k,
        budget=args.budget,
        max_searches=args.max_searches,
    )
    result["retrieval_config"] = {
        "strategy": "B",
        "embedding_model": retriever.model_name,
        "embedding_dimensions": retriever.embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector / ingestion_chunks",
        "gemini_model": controller.model,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
