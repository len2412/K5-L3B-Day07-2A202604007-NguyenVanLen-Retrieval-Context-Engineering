"""Dense retrieval adapter for this workspace's Day 6 PostgreSQL/pgvector index.

The adapter deliberately reuses Day 6's BGE embedder and pgvector store. It does
not create or rebuild the index; strategy B must already have been ingested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _find_day6_starter() -> Path:
    configured = os.getenv("DAY6_STARTER")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not (path / "src" / "store.py").is_file():
            raise FileNotFoundError(f"DAY6_STARTER does not contain src/store.py: {path}")
        return path

    workspace = Path(__file__).resolve().parents[2]
    matches = sorted(workspace.glob("K5-L3-Day06-*-DataIngestionEngineering/starter"))
    matches = [path for path in matches if (path / "src" / "store.py").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate the Day 6 starter. Set DAY6_STARTER to its starter directory."
        )
    return matches[0]


DAY6_STARTER = _find_day6_starter()
if str(DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(DAY6_STARTER))

try:
    from dotenv import load_dotenv
    from src.embeddings import LocalBgeEmbedder
    from src.store import PgVectorStore
except ImportError as exc:
    raise ImportError(
        "Install Day 6 dependencies first: python -m pip install -r "
        f"\"{DAY6_STARTER / 'requirements.txt'}\""
    ) from exc


class Day6DenseRetriever:
    """Day 7 Retriever contract backed by Day 6's strategy B index."""

    strategy = "B"

    def __init__(self, database_url: str | None = None) -> None:
        load_dotenv(DAY6_STARTER / ".env", override=False)
        # This is deliberately distinct from Day 6's EMBEDDING_MODEL setting:
        # that variable configures ingestion, while this one must match the
        # already-built index. The committed Day 6 run record identifies BGE-large.
        self.model_name = os.getenv(
            "DAY6_INDEX_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
        ).strip()
        self.embedder = LocalBgeEmbedder(model_name=self.model_name)
        expected_dimensions = 1024 if self.model_name == "BAAI/bge-large-en-v1.5" else None
        if expected_dimensions is not None and self.embedder.dimensions != expected_dimensions:
            raise ValueError(
                f"The recorded Day 6 index expects {expected_dimensions} dimensions, "
                f"but {self.model_name} returned {self.embedder.dimensions}."
            )

        url = database_url or os.getenv("DATABASE_URL", "")
        if not url:
            raise EnvironmentError(
                f"DATABASE_URL is missing. Configure it in {DAY6_STARTER / '.env'} or the environment."
            )
        self.store = PgVectorStore(url, dimensions=self.embedder.dimensions)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        rows = self.store.search(
            strategy=self.strategy,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        # Keep Day 6's stable chunk ID and complete provenance for context/citations.
        return [
            {
                "id": row["chunk_id"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": row["score"],
                "strategy": self.strategy,
            }
            for row in rows
        ]
