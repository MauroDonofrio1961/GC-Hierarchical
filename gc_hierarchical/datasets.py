from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """A named, validated table with its on-disk provenance."""

    name: str
    path: Path
    table: pd.DataFrame

    def __len__(self) -> int:
        return len(self.table)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.table.columns)


@dataclass(frozen=True)
class DatasetBundle:
    """The standard dataset interface consumed by likelihood models."""

    science: Dataset
    confirmed: Dataset
    background: Dataset
    manifest: Path
