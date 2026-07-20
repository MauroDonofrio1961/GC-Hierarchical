from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..components import ComponentFactory, ModelComponents, V5ComponentFactory
from ..datasets import DatasetBundle
from ..exceptions import ConfigurationError
from ..galaxy import Galaxy
from ..inference import EmceeSampler, NoSampler, Optimizer, Sampler, V5Optimizer
from ..likelihood_v5.plots import make_plots
from ..likelihood_v5.report import write_report
from ..science_config import HierarchicalModelConfig
from .base import FitResult


class HierarchicalGCModel:
    """
    Object-owned scientific model.

    The model owns the galaxy, datasets, component graph, posterior,
    optimizer, sampler, outputs, and provenance. Frozen v5 functions remain
    unchanged and are accessed through injected adapters.
    """

    name = "hierarchical-gc-model"

    def __init__(
        self,
        galaxy: Galaxy,
        configuration: HierarchicalModelConfig,
        *,
        component_factory: ComponentFactory | None = None,
        optimizer: Optimizer | None = None,
        sampler: Sampler | None = None,
    ):
        configuration.validate()
        self.galaxy = galaxy
        self.configuration = configuration
        self.component_factory = component_factory or V5ComponentFactory()
        self.optimizer = optimizer or V5Optimizer()
        self.sampler = sampler or EmceeSampler()

        self.datasets: DatasetBundle | None = None
        self.components: ModelComponents | None = None
        self.last_result: FitResult | None = None

    @classmethod
    def from_galaxy(
        cls,
        galaxy: Galaxy,
        *,
        mode: str = "science",
        component_factory: ComponentFactory | None = None,
        optimizer: Optimizer | None = None,
        sampler: Sampler | None = None,
    ) -> "HierarchicalGCModel":
        likelihood = galaxy.config["validated_likelihood"]
        relative = (
            likelihood["fast_config"]
            if mode == "quick"
            else likelihood["science_config"]
        )
        path = galaxy.config_manager.root / "gc_hierarchical" / relative
        return cls(
            galaxy,
            HierarchicalModelConfig.from_json(path),
            component_factory=component_factory,
            optimizer=optimizer,
            sampler=sampler,
        )

    @staticmethod
    def verify_fsps() -> None:
        try:
            import fsps  # noqa: F401
        except ImportError as exc:
            raise ConfigurationError(
                "The Python package 'fsps' is not importable in this "
                "environment. Activate the environment where FSPS is installed, "
                "or install it with install_fsps.command."
            ) from exc

    @property
    def is_built(self) -> bool:
        return self.components is not None

    @property
    def posterior(self):
        if self.components is None:
            raise RuntimeError("Model is not built. Call build() first.")
        return self.components.posterior

    def build(
        self,
        datasets: DatasetBundle | None = None,
        *,
        force_data: bool = False,
        verify_fsps: bool = True,
    ) -> "HierarchicalGCModel":
        if verify_fsps and isinstance(self.component_factory, V5ComponentFactory):
            self.verify_fsps()
        self.datasets = datasets or self.galaxy.datasets(force=force_data)
        self.components = self.component_factory.build(
            self.configuration.raw,
            self.datasets,
        )
        return self

    def log_posterior(self, theta: np.ndarray) -> float:
        return float(self.posterior(np.asarray(theta, dtype=float)))

    def _membership(self, theta: np.ndarray) -> tuple[float, pd.DataFrame]:
        value, membership_array = self.posterior(
            np.asarray(theta, dtype=float),
            return_membership=True,
        )
        if membership_array is None:
            raise RuntimeError("Posterior did not return membership probabilities.")
        membership = pd.DataFrame(
            membership_array,
            columns=["p_blue", "p_red", "p_background"],
        )
        if self.datasets is not None and len(membership) == len(self.datasets.science):
            membership.insert(
                0,
                "catalogue_index",
                self.datasets.science.table.index.to_numpy(),
            )
        return float(value), membership

    def fit(
        self,
        *,
        mode: str = "science",
        output_directory: str | Path,
        make_diagnostics: bool = True,
    ) -> FitResult:
        if mode not in {"quick", "science", "mcmc"}:
            raise ValueError("mode must be 'quick', 'science', or 'mcmc'")
        if not self.is_built:
            self.build()

        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)

        optimization = self.optimizer.optimize(
            self.configuration.raw,
            self.posterior,
            mode,
            output,
        )
        log_posterior, membership = self._membership(optimization.theta)
        membership.to_csv(output / "membership_probabilities.csv", index=False)

        sampling = (
            self.sampler.sample(
                self.configuration.raw,
                self.posterior,
                optimization.theta,
                output,
            )
            if mode == "mcmc"
            else NoSampler().sample(
                self.configuration.raw,
                self.posterior,
                optimization.theta,
                output,
            )
        )

        if make_diagnostics and self.datasets is not None:
            make_plots(
                self.configuration.raw,
                self.datasets.science.table,
                self.datasets.background.table,
                optimization.theta,
                membership[
                    ["p_blue", "p_red", "p_background"]
                ].to_numpy(),
                output,
                sampling.chain,
            )
            write_report(
                optimization.report,
                self.datasets.science.table,
                membership[
                    ["p_blue", "p_red", "p_background"]
                ].to_numpy(),
                output,
                sampling.summary,
            )

        provenance = {
            "model": self.name,
            "galaxy": self.galaxy.key,
            "configuration_source": (
                None
                if self.configuration.source is None
                else str(self.configuration.source)
            ),
            "parameter_names": list(
                self.configuration.mixture.parameter_names
            ),
            "components": list(self.configuration.mixture.components),
            "mode": mode,
        }
        (output / "model_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "configuration_used.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in self.configuration.raw.items()
                    if not key.startswith("_")
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        result = FitResult(
            model_name=self.name,
            mode=mode,
            log_posterior=log_posterior,
            parameters={
                key: float(value)
                for key, value in zip(
                    self.configuration.mixture.parameter_names,
                    optimization.theta,
                )
            },
            calls=int(getattr(self.posterior, "calls", 0)),
            output_directory=output,
            membership=membership,
            raw_result=optimization.report,
        )
        self.last_result = result
        return result
