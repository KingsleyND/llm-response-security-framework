"""Ask one question or start an interactive terminal chat."""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from chromadb.errors import NotFoundError
from ollama import Client, ResponseError

from .embeddings import load_embedding_model
from .generation import DEFAULT_MODEL, generate_answer
from .index import DEFAULT_DB_PATH
from .retrieval import open_index, retrieve_passages

GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}


def read_questions() -> Iterator[str]:
    print("Ask about the MACCROBAT case reports. Type /exit to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.casefold() == "/exit":
            return
        if question:
            yield question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    args = parser.parse_args()
    if args.question is not None and not args.question.strip():
        parser.error("Enter a question.")

    questions = [args.question.strip()] if args.question is not None else read_questions()
    client = Client()
    retrieval_resources = None
    for question in questions:
        if question.casefold().rstrip("!?.").strip() in GREETINGS:
            print("Hi! What would you like to know about the case reports?")
            continue

        if retrieval_resources is None:
            try:
                collection = open_index(args.db_path)
            except (FileNotFoundError, NotFoundError):
                parser.error(
                    f"No MACCROBAT index at {args.db_path}. "
                    "Run python -m medical_rag.index using the same --db-path."
                )
            retrieval_resources = (load_embedding_model(), collection)

        embedding_model, collection = retrieval_resources
        passages = retrieve_passages(question, embedding_model, collection)
        try:
            print(generate_answer(question, passages, client, model=args.model))
        except ConnectionError:
            print("Could not connect to Ollama. Open the Ollama app and try again.", file=sys.stderr)
            return 1
        except ResponseError as error:
            if error.status_code == 404:
                print(f"Ollama model unavailable. Run ollama pull {args.model}.", file=sys.stderr)
            else:
                print(f"Ollama error: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
