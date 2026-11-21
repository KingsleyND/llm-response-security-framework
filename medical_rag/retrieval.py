"""Search the stored MACCROBAT passages with a question."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.errors import NotFoundError

from .embeddings import embed_texts, load_embedding_model
from .index import COLLECTION_NAME, DEFAULT_DB_PATH

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    case_id: str
    text: str
    entities: str
    distance: float


def open_index(db_path: Path) -> chromadb.Collection:
    """Open an existing Chroma collection for querying."""
    if not db_path.is_dir():
        raise FileNotFoundError(f"No index directory at {db_path}. Run medical_rag.index first.")

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def retrieve_passages(
    question: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    *,
    limit: int = 5,
) -> list[RetrievedPassage]:
    """Return the closest passages, ordered by increasing cosine distance."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a question.")
    if limit < 1:
        raise ValueError("The result limit must be at least 1.")

    results = collection.query(
        query_embeddings=embed_texts(model, [question]),
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedPassage(
            chunk_id=chunk_id,
            case_id=metadata["case_id"],
            text=text,
            entities=metadata["entities"],
            distance=distance,
        )
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the MACCROBAT vector index.")
    parser.add_argument("question")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    try:
        collection = open_index(args.db_path)
    except (FileNotFoundError, NotFoundError):
        parser.error(
            f"No MACCROBAT index at {args.db_path}. "
            "Build it with python -m medical_rag.index using the same --db-path."
        )
    if not args.question.strip():
        parser.error("Enter a question.")
    if args.limit < 1:
        parser.error("The result limit must be at least 1.")

    passages = retrieve_passages(
        args.question, load_embedding_model(), collection, limit=args.limit
    )
    if not passages:
        print("No passages found.")
    for passage in passages:
        print(f"\n{passage.chunk_id} ({passage.case_id}, distance {passage.distance:.4f})")
        print(passage.text.strip())


if __name__ == "__main__":
    main()
