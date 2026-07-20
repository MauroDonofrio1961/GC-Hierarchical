from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gc_hierarchical.components import ModelComponents
from gc_hierarchical.galaxy import Galaxy
from gc_hierarchical.inference import OptimizationResult, SamplingResult
from gc_hierarchical.models import HierarchicalGCModel
from gc_hierarchical.science_config import HierarchicalModelConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "gc_hierarchical/likelihood_v5/config_fast.json"


class FakePosterior:
    def __init__(self, rows: int):
        self.rows = rows
        self.calls = 0

    def __call__(self, theta, return_membership=False):
        self.calls += 1
        theta = np.asarray(theta, dtype=float)
        value = -float(np.sum(theta**2))
        if not return_membership:
            return value
        membership = np.tile([0.5, 0.3, 0.2], (self.rows, 1))
        return value, membership


class FakeFactory:
    def build(self, configuration, datasets):
        posterior = FakePosterior(len(datasets.science))
        return ModelComponents(
            selection=object(),
            stellar_population=object(),
            background=object(),
            cluster_population=object(),
            posterior=posterior,
        )


class FakeOptimizer:
    def optimize(self, configuration, posterior, mode, output_directory):
        theta = np.zeros(len(configuration["parameters"]["names"]))
        return OptimizationResult(
            theta=theta,
            report={
                "mode": mode,
                "log_posterior": posterior(theta),
                "calls": posterior.calls,
                "parameters": dict(
                    zip(configuration["parameters"]["names"], theta)
                ),
            },
        )


class FakeSampler:
    def sample(self, configuration, posterior, map_theta, output_directory):
        return SamplingResult(chain=None, summary={"status": "fake"})


def make_model():
    galaxy = Galaxy("NGC5128", ROOT)
    config = HierarchicalModelConfig.from_json(CONFIG)
    return HierarchicalGCModel(
        galaxy,
        config,
        component_factory=FakeFactory(),
        optimizer=FakeOptimizer(),
        sampler=FakeSampler(),
    )


def test_model_owns_component_graph():
    model = make_model().build(verify_fsps=False)
    assert model.is_built
    assert model.components is not None
    assert set(model.components.as_dict()) == {
        "selection",
        "stellar_population",
        "background",
        "cluster_population",
        "posterior",
    }


def test_model_fit_writes_standard_outputs(tmp_path):
    model = make_model().build(verify_fsps=False)
    result = model.fit(
        mode="quick",
        output_directory=tmp_path,
        make_diagnostics=False,
    )
    assert result.model_name == "hierarchical-gc-model"
    assert len(result.parameters) == 17
    assert len(result.membership) == 1256
    assert np.allclose(
        result.membership[
            ["p_blue", "p_red", "p_background"]
        ].sum(axis=1),
        1.0,
    )
    assert (tmp_path / "membership_probabilities.csv").is_file()
    assert (tmp_path / "model_provenance.json").is_file()
    assert (tmp_path / "configuration_used.json").is_file()


def test_model_can_be_reused_for_log_posterior():
    model = make_model().build(verify_fsps=False)
    theta = np.zeros(17)
    assert model.log_posterior(theta) == 0.0
    assert model.posterior.calls == 1


def test_model_factory_from_galaxy():
    galaxy = Galaxy("NGC5128", ROOT)
    model = HierarchicalGCModel.from_galaxy(
        galaxy,
        mode="quick",
        component_factory=FakeFactory(),
        optimizer=FakeOptimizer(),
        sampler=FakeSampler(),
    )
    assert model.configuration.ndim == 17
    assert model.galaxy.key == "NGC5128"
