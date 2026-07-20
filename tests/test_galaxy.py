from pathlib import Path

from gc_hierarchical.galaxy import Galaxy


ROOT = Path(__file__).resolve().parents[1]


def test_galaxy_metadata():
    galaxy = Galaxy("NGC5128", ROOT)
    metadata = galaxy.metadata
    assert metadata.name == "NGC 5128"
    assert metadata.filters == ("megacam_g", "megacam_r")
    assert metadata.distance_modulus == 27.91


def test_dataset_bundle():
    datasets = Galaxy("NGC5128", ROOT).datasets()
    assert len(datasets.science) == 1256
    assert len(datasets.confirmed) == 557
    assert len(datasets.background) == 2844
