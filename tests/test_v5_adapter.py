from pathlib import Path

import numpy as np
import pytest

from gc_hierarchical.galaxy import Galaxy
from gc_hierarchical.models import ThreeComponentV5Model


ROOT = Path(__file__).resolve().parents[1]


def config_path(name: str = "config_fast.json") -> Path:
    return ROOT / "gc_hierarchical" / "likelihood_v5" / name


def test_v5_model_configuration_contract():
    model = ThreeComponentV5Model.from_config_file(config_path())
    names = model.configuration["parameters"]["names"]
    bounds = model.configuration["parameters"]["bounds"]
    assert len(names) == 17
    assert len(bounds) == 17
    assert names[8:12] == [
        "gold_blue_logit",
        "gold_red_logit",
        "silver_blue_logit",
        "silver_red_logit",
    ]


def test_v5_softmax_contract():
    from gc_hierarchical.likelihood_v5.posterior import Posterior

    for pair in ((0.0, 0.0), (4.0, -4.0), (-4.0, 4.0)):
        p = Posterior.softmax3(*pair)
        assert np.all(p > 0)
        assert np.isclose(p.sum(), 1.0, atol=1e-12)


def test_model_requires_build_before_evaluation():
    model = ThreeComponentV5Model.from_config_file(config_path())
    with pytest.raises(RuntimeError, match="not built"):
        model.log_posterior(np.zeros(17))


def test_prepared_science_matches_frozen_v5_selection():
    from gc_hierarchical.likelihood_v5.catalogue import (
        load_catalogues,
        science_sample,
        background_sample,
    )

    model = ThreeComponentV5Model.from_config_file(config_path())
    cfg = model.configuration
    galaxy_dir = ROOT / "data" / "NGC5128"
    cfg["data"]["candidate_catalogue"] = str(
        galaxy_dir / "raw/Hughes/ngc5128_table1_parsed.csv"
    )
    cfg["data"]["confirmed_catalogue"] = str(
        galaxy_dir / "raw/Hughes/confirmed_gc_match_table.csv"
    )

    candidate, _ = load_catalogues(cfg)
    frozen_science = science_sample(candidate, cfg)
    frozen_background = background_sample(candidate, cfg)
    datasets = Galaxy("NGC5128", ROOT).datasets()

    assert len(frozen_science) == len(datasets.science)
    assert len(frozen_background) == len(datasets.background)
    assert frozen_science["ID"].astype(str).tolist() == (
        datasets.science.table["ID"].astype(str).tolist()
    )
    assert frozen_background["ID"].astype(str).tolist() == (
        datasets.background.table["ID"].astype(str).tolist()
    )
