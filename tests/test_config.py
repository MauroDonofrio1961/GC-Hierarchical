from pathlib import Path

from gc_hierarchical.config import ConfigManager


ROOT = Path(__file__).resolve().parents[1]


def test_project_config_loads():
    config = ConfigManager(ROOT)
    assert config.project["project"]["version"] == "1.0-alpha3"
    assert "NGC5128" in config.project["galaxies"]
