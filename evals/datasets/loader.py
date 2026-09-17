from pathlib import Path
from typing import Any

import yaml


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = yaml.safe_load(file)

    if not isinstance(dataset, dict):
        raise ValueError(
            f"Dataset must contain a YAML mapping: {dataset_path}"
        )

    if "cases" not in dataset:
        raise ValueError(
            f"Dataset is missing 'cases': {dataset_path}"
        )

    return dataset
