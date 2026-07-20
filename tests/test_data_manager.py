from pathlib import Path

import pandas as pd

from gc_hierarchical.config import ConfigManager
from gc_hierarchical.data_manager import DataManager


ROOT = Path(__file__).resolve().parents[1]


def test_ngc5128_raw_data_validate():
    config = ConfigManager(ROOT)
    candidate, confirmed = DataManager(config, "NGC5128").validate_raw()
    assert len(candidate) == 40502
    assert len(confirmed) == 557


def test_prepare_catalogues():
    config = ConfigManager(ROOT)
    prepared = DataManager(config, "NGC5128").prepare(force=True)
    science = pd.read_csv(prepared.science)
    background = pd.read_csv(prepared.background)

    assert len(science) > 0
    assert len(background) > 0
    assert set(science["Rank"].unique()) <= {"gold", "silver"}
    assert science["TotL"].min() >= 0.85
    assert ((science["gmag"] >= 17.0) & (science["gmag"] <= 22.0)).all()
    assert ((science["rmag"] >= 17.0) & (science["rmag"] <= 22.0)).all()
    assert ((science["color"] >= 0.3) & (science["color"] <= 1.3)).all()
