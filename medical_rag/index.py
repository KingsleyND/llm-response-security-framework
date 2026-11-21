"""Build the persistent Chroma index from the pinned MACCROBAT dataset."""

import argparse
from pathlib import Path

import chromadb

from .chunking import chunk_report
from .dataset import load_cases
from .embeddings import embed_texts, load_embedding_model

DEFAULT_DB_PATH = Path("medical_rag_index")
COLLECTION_NAME = "maccrobat_documents"


def build_index(db_path: Path) -> int:
    """Embed the dataset's chunks and return the stored record count."""
    chunks = [chunk for report in load_cases() for chunk in chunk_report(report)]
    if not chunks:
        raise ValueError("The dataset produced no chunks.")

    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(load_embedding_model(), texts, show_progress=True)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "case_id": chunk.case_id,
                "start": chunk.start,
                "end": chunk.end,
                "entities": ", ".join(entity.text for entity in chunk.entities),
                "entity_types": ", ".join(entity.label for entity in chunk.entities),
            }
            for chunk in chunks
        ],
    )
    return collection.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MACCROBAT vector index.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    count = build_index(args.db_path)
    print(f"Indexed {count} chunks in {args.db_path}.")


if __name__ == "__main__":
    main()
