"""Encode passages and questions into the same vector space."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load MiniLM, downloading its weights on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(
    model: SentenceTransformer, texts: list[str], *, show_progress: bool = False
) -> list[list[float]]:
    """Return one normalized vector per text, in input order."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    ).tolist()
