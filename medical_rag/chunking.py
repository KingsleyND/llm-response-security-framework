"""Split case reports into passages while preserving source offsets."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import CaseReport, Entity, load_cases

SENTENCE_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    case_id: str
    text: str
    start: int
    end: int
    entities: tuple[Entity, ...]


def chunk_report(report: CaseReport, max_chars: int = 1200) -> list[Chunk]:
    """Group whole sentences with one sentence shared between adjacent chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    sentences = [
        match.span()
        for match in SENTENCE_PATTERN.finditer(report.text)
        if match.group().strip()
    ]
    if not sentences:
        return []

    spans = []
    start, end = sentences[0]
    previous_start = start
    for sentence_start, sentence_end in sentences[1:]:
        if sentence_end - start > max_chars:
            spans.append((start, end))
            start = previous_start
        previous_start, end = sentence_start, sentence_end
    spans.append((start, end))

    return [
        Chunk(
            chunk_id=f"{report.case_id}_chunk_{index}",
            case_id=report.case_id,
            text=report.text[start:end],
            start=start,
            end=end,
            entities=tuple(
                entity for entity in report.entities
                if start <= entity.start and entity.end <= end
            ),
        )
        for index, (start, end) in enumerate(spans)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect passages for one MACCROBAT case.")
    parser.add_argument("--file", type=Path, help="Use a local copy of the dataset JSON.")
    parser.add_argument("--case-id", default="case_0", help="Case to inspect (default: case_0).")
    args = parser.parse_args()

    for report in load_cases(args.file):
        if report.case_id == args.case_id:
            for chunk in chunk_report(report):
                print(
                    f"{chunk.chunk_id} [{chunk.start}:{chunk.end}] "
                    f"({len(chunk.text)} characters, {len(chunk.entities)} entities)"
                )
            return
    parser.error(f"Unknown case ID: {args.case_id}")


if __name__ == "__main__":
    main()
