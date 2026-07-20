from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..datasets import DatasetBundle


@dataclass(frozen=True)
class FitResult:
    model_name: str
    mode: str
    log_posterior: float
    parameters: dict[str, float]
    calls: int
    output_directory: Path
    membership: pd.DataFrame | None = None
    raw_result: dict[str, Any] | None = None


class LikelihoodModel(Protocol):
    name: str

    def build(self, datasets: DatasetBundle) -> None:
        ...

    def log_posterior(self, theta: np.ndarray) -> float:
        ...

    def fit(self, mode: str, output_directory: Path) -> FitResult:
        ...
