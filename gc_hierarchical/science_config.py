from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class StellarPopulationConfig:
    backend: str
    filters: tuple[str, ...]
    age_gyr: float
    imf_type: int


@dataclass(frozen=True)
class MixtureConfig:
    components: tuple[str, ...]
    rank_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class InferenceConfig:
    optimizer: str
    sampler: str
    seed: int


@dataclass(frozen=True)
class HierarchicalModelConfig:
    name: str
    stellar_population: StellarPopulationConfig
    mixture: MixtureConfig
    inference: InferenceConfig
    raw: dict[str, Any]
    source: Path | None = None

    @classmethod
    def from_dict(
        cls,
        configuration: dict[str, Any],
        source: str | Path | None = None,
    ) -> "HierarchicalModelConfig":
        stellar = configuration["stellar_population"]
        parameters = configuration["parameters"]
        return cls(
            name=configuration["project"]["name"],
            stellar_population=StellarPopulationConfig(
                backend="FSPS",
                filters=tuple(stellar["filters"]),
                age_gyr=float(stellar["age_gyr"]),
                imf_type=int(stellar["imf_type"]),
            ),
            mixture=MixtureConfig(
                components=("blue", "red", "background"),
                rank_names=("gold", "silver"),
                parameter_names=tuple(parameters["names"]),
                bounds=tuple(
                    (float(pair[0]), float(pair[1]))
                    for pair in parameters["bounds"]
                ),
            ),
            inference=InferenceConfig(
                optimizer="latin-hypercube+Powell",
                sampler="emcee",
                seed=int(configuration["project"]["seed"]),
            ),
            raw=configuration,
            source=None if source is None else Path(source).resolve(),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "HierarchicalModelConfig":
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw["_base"] = str(source.parent)
        return cls.from_dict(raw, source=source)

    @property
    def ndim(self) -> int:
        return len(self.mixture.parameter_names)

    def validate(self) -> None:
        if self.mixture.components != ("blue", "red", "background"):
            raise ValueError("The validated v5 model requires three components.")
        if len(self.mixture.parameter_names) != len(self.mixture.bounds):
            raise ValueError("Parameter names and bounds have different lengths.")
        if self.ndim != 17:
            raise ValueError(
                f"The validated v5 parameter contract has 17 dimensions, got {self.ndim}."
            )
        for name, (lower, upper) in zip(
            self.mixture.parameter_names, self.mixture.bounds
        ):
            if lower >= upper:
                raise ValueError(f"Invalid bounds for {name}: {lower}, {upper}")
