"""Load case reports and their annotations from MACCROBAT JSON."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO = "singh-aditya/MACCROBAT_biomedical_ner"
DATASET_FILENAME = "MACCROBAT2020-V2.json"
DATASET_REVISION = "1470ebf"


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    text: str
    entities: tuple[Entity, ...]


def load_cases(path: Path | None = None) -> list[CaseReport]:
    """Read a local copy or download the pinned dataset into the Hub cache."""
    if path is None:
        path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                filename=DATASET_FILENAME,
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )

    with path.open(encoding="utf-8") as dataset_file:
        rows = json.load(dataset_file)["data"]

    return [
        CaseReport(
            case_id=f"case_{index}",
            text=row["full_text"],
            entities=tuple(Entity(**annotation) for annotation in row["ner_info"]),
        )
        for index, row in enumerate(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the MACCROBAT case reports.")
    parser.add_argument("--file", type=Path, help="Use a local copy of the dataset JSON.")
    args = parser.parse_args()

    cases = load_cases(args.file)
    print(f"Loaded {len(cases)} case reports.")
    if cases:
        first = cases[0]
        print(f"{first.case_id} has {len(first.text)} characters and {len(first.entities)} entities.")
        print(first)


if __name__ == "__main__":
    main()
