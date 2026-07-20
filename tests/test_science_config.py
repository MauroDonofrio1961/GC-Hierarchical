from pathlib import Path

import pytest

from gc_hierarchical.science_config import HierarchicalModelConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "gc_hierarchical/likelihood_v5/config_fast.json"


def test_typed_model_configuration():
    config = HierarchicalModelConfig.from_json(CONFIG)
    assert config.ndim == 17
    assert config.stellar_population.backend == "FSPS"
    assert config.stellar_population.filters == ("megacam_g", "megacam_r")
    assert config.mixture.components == ("blue", "red", "background")
    config.validate()


def test_invalid_dimension_is_rejected():
    config = HierarchicalModelConfig.from_json(CONFIG)
    raw = dict(config.raw)
    raw["parameters"] = dict(raw["parameters"])
    raw["parameters"]["names"] = raw["parameters"]["names"][:-1]
    with pytest.raises(ValueError, match="different lengths"):
        HierarchicalModelConfig.from_dict(raw).validate()
