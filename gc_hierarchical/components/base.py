from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelComponents:
    selection: Any
    stellar_population: Any
    background: Any
    cluster_population: Any
    posterior: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "stellar_population": self.stellar_population,
            "background": self.background,
            "cluster_population": self.cluster_population,
            "posterior": self.posterior,
        }


class ComponentFactory:
    """Interface for constructing scientific components."""

    def build(self, configuration: dict, datasets) -> ModelComponents:
        raise NotImplementedError


class CallablePosterior:
    """Small posterior adapter useful for tests and future backends."""

    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        parameter_names: tuple[str, ...],
    ):
        self.function = function
        self.parameter_names = parameter_names
        self.calls = 0

    def __call__(self, theta, return_membership: bool = False):
        self.calls += 1
        theta = np.asarray(theta, dtype=float)
        value = float(self.function(theta))
        if not return_membership:
            return value
        membership = np.tile(
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
            (1, 1),
        )
        return value, membership
