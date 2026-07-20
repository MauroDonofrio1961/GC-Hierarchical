from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..components import V5ComponentFactory
from ..datasets import DatasetBundle
from ..galaxy import Galaxy
from ..science_config import HierarchicalModelConfig
from .hierarchical_gc import HierarchicalGCModel


class ThreeComponentV5Model:
    """
    Backward-compatible Step 2 adapter.

    New code should use HierarchicalGCModel.from_galaxy(). This class preserves
    the Step 2 API while delegating component ownership and inference to the
    Step 3 model.
    """

    name = "NGC5128-three-component-v5"

    def __init__(self, configuration: dict[str, Any]):
        self.configuration = configuration
        self.datasets: DatasetBundle | None = None
        self._model: HierarchicalGCModel | None = None
        self.posterior = None

    @classmethod
    def from_config_file(cls, path: str | Path) -> "ThreeComponentV5Model":
        config = HierarchicalModelConfig.from_json(path)
        return cls(config.raw)

    @staticmethod
    def verify_fsps() -> None:
        HierarchicalGCModel.verify_fsps()

    def build(self, datasets: DatasetBundle) -> None:
        self.verify_fsps()
        self.datasets = datasets

        # Resolve the project root from the standard NGC5128 data path.
        root = datasets.manifest.resolve().parents[3]
        galaxy = Galaxy("NGC5128", root)
        typed = HierarchicalModelConfig.from_dict(self.configuration)
        self._model = HierarchicalGCModel(
            galaxy,
            typed,
            component_factory=V5ComponentFactory(),
        ).build(datasets=datasets, verify_fsps=False)
        self.posterior = self._model.posterior

    def require_built(self):
        if self._model is None:
            raise RuntimeError("Model is not built. Call build(datasets) first.")
        return self._model

    def log_posterior(self, theta: np.ndarray) -> float:
        return self.require_built().log_posterior(theta)

    def fit(self, mode: str, output_directory: Path):
        result = self.require_built().fit(
            mode=mode,
            output_directory=output_directory,
        )
        # Preserve the old model name.
        return type(result)(
            model_name=self.name,
            mode=result.mode,
            log_posterior=result.log_posterior,
            parameters=result.parameters,
            calls=result.calls,
            output_directory=result.output_directory,
            membership=result.membership,
            raw_result=result.raw_result,
        )
