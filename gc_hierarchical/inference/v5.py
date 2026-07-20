from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OptimizationResult, SamplingResult
from ..likelihood_v5.mcmc import run_mcmc
from ..likelihood_v5.optimize import optimize_map


class V5Optimizer:
    def optimize(
        self,
        configuration: dict,
        posterior,
        mode: str,
        output_directory: Path,
    ) -> OptimizationResult:
        optimizer_mode = "science" if mode in {"science", "mcmc"} else "quick"
        theta, report = optimize_map(
            configuration,
            posterior,
            optimizer_mode,
            output_directory,
        )
        return OptimizationResult(np.asarray(theta, dtype=float), report)


class EmceeSampler:
    def sample(
        self,
        configuration: dict,
        posterior,
        map_theta: np.ndarray,
        output_directory: Path,
    ) -> SamplingResult:
        chain, summary = run_mcmc(
            configuration,
            posterior,
            map_theta,
            output_directory,
        )
        return SamplingResult(chain=chain, summary=summary)


class NoSampler:
    def sample(
        self,
        configuration: dict,
        posterior,
        map_theta: np.ndarray,
        output_directory: Path,
    ) -> SamplingResult:
        return SamplingResult(chain=None, summary=None)
