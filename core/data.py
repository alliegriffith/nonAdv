# This function randomly samples a single question from the BBQ dataset and returns it as a string. 
# It is used to provide a neutral prompt for the simulator, which initiates a single conversation rollout.

"""
Utilities for loading and sampling prompts from the BBQ dataset.

The dataset is loaded once and cached for the duration of the process.
Sampling is uniform over all individual JSONL records across all 11
category files.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


DEFAULT_BBQ_DATA_DIR = Path("/home/allie11/BBQ/data")


@lru_cache(maxsize=None)
def load_bbq_records(
    data_dir: str | Path = DEFAULT_BBQ_DATA_DIR,
) -> tuple[dict[str, Any], ...]:
    """
    Load every record from all BBQ JSONL category files.

    Each returned record contains two extra fields:

        _source_file:
            Name of the JSONL file from which the record was loaded.

        _record_key:
            Composite identifier formed from category and example_id.

    The tuple is cached, so the files are parsed only once per Python process.
    """
    directory = Path(data_dir).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(
            f"BBQ data directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"BBQ data path is not a directory: {directory}"
        )

    jsonl_files = sorted(directory.glob("*.jsonl"))

    if len(jsonl_files) != 11:
        raise ValueError(
            "Expected exactly 11 BBQ JSONL files in "
            f"{directory}, but found {len(jsonl_files)}."
        )

    records: list[dict[str, Any]] = []

    required_fields = {
        "category",
        "example_id",
        "question_index",
        "context",
        "question",
        "ans0",
        "ans1",
        "ans2",
    }

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line "
                        f"{line_number}: {error}"
                    ) from error

                missing_fields = required_fields - record.keys()

                if missing_fields:
                    raise ValueError(
                        f"Record in {path} at line {line_number} "
                        f"is missing fields: {sorted(missing_fields)}"
                    )

                # Copy it so the source data is not modified in place.
                record = dict(record)

                category = str(record["category"])
                example_id = str(record["example_id"])

                record["_source_file"] = path.name
                record["_record_key"] = (
                    f"{category}:{example_id}"
                )

                records.append(record)

    if not records:
        raise ValueError(
            f"No BBQ records were loaded from {directory}."
        )

    return tuple(records)


def format_bbq_prompt(record: dict[str, Any]) -> str:
    """
    Format one BBQ record as a complete prompt for SimVBG.
    
    
    """
    return (
        #f"You have been thinking about the following scenario and want to talk to a chatbot assistant about it\n\n"
        f"{record['context']}\n"
        f"{record['question']}"
        # I don't think we need to include answer choices?
        # "Possible answers:\n"
        # f"A. {record['ans0']}\n"
        # f"B. {record['ans1']}\n"
        # f"C. {record['ans2']}"
    )


def sample_bbq_record(
    data_dir: str | Path = DEFAULT_BBQ_DATA_DIR,
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Uniformly sample one individual record from the entire BBQ dataset.

    This is uniform over records, not over categories.
    """
    records = load_bbq_records(data_dir)
    random_generator = rng if rng is not None else random

    return random_generator.choice(records)


def sample_neutral_prompt(
    data_dir: str | Path = DEFAULT_BBQ_DATA_DIR,
    *,
    rng: random.Random | None = None,
) -> str:
    """
    Uniformly sample and format one complete BBQ example.
    """
    record = sample_bbq_record(
        data_dir=data_dir,
        rng=rng,
    )

    return format_bbq_prompt(record)


def sample_neutral_prompt_with_metadata(
    data_dir: str | Path = DEFAULT_BBQ_DATA_DIR,
    *,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Sample one BBQ prompt and return useful experiment metadata.

    Returns:
        prompt:
            Complete context, question, and answer choices.

        metadata:
            IDs, category, labels, and source-file information for logging.
    """
    record = sample_bbq_record(
        data_dir=data_dir,
        rng=rng,
    )

    metadata = {
        "record_key": record["_record_key"],
        "source_file": record["_source_file"],
        "category": record["category"],
        "question_index": record["question_index"],
        "example_id": record["example_id"],
        "context_condition": record.get("context_condition"),
        "question_polarity": record.get("question_polarity"),
        "label": record.get("label"),
    }

    return format_bbq_prompt(record), metadata


def validate_unique_record_keys(
    records: Sequence[dict[str, Any]],
) -> None:
    """
    Verify that category + example_id uniquely identifies each loaded row.
    """
    keys = [
        str(record["_record_key"])
        for record in records
    ]

    unique_count = len(set(keys))

    if unique_count != len(keys):
        raise ValueError(
            "Duplicate (category, example_id) record keys found: "
            f"{len(keys) - unique_count} duplicates."
        )
        
if __name__ == "__main__":
    records = load_bbq_records()
    print(f"Loaded {len(records)} BBQ records")

    prompt = sample_neutral_prompt()
    print(prompt)