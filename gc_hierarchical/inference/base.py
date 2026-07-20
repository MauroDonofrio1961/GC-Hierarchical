from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    theta: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class SamplingResult:
    chain: np.ndarray | None
    summary: dict[str, Any] | None


class Optimizer(Protocol):
    def optimize(
        self,
        configuration: dict[str, Any],
        posterior,
        mode: str,
        output_directory: Path,
    ) -> OptimizationResult:
        ...


class Sampler(Protocol):
    def sample(
        self,
        configuration: dict[str, Any],
        posterior,
        map_theta: np.ndarray,
        output_directory: Path,
    ) -> SamplingResult:
        ...
