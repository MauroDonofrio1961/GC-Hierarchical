from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigManager
from .data_manager import DataManager
from .datasets import Dataset, DatasetBundle


@dataclass(frozen=True)
class GalaxyMetadata:
    key: str
    name: str
    aliases: tuple[str, ...]
    ra_deg: float
    dec_deg: float
    distance_modulus: float
    filters: tuple[str, ...]


class Galaxy:
    """Configuration and data access point for one galaxy."""

    def __init__(self, key: str, root: str | Path):
        self.key = key
        self.config_manager = ConfigManager(root)
        self.directory = self.config_manager.galaxy_directory(key)
        self.config: dict[str, Any] = self.config_manager.galaxy_config(key)
        self.data_manager = DataManager(self.config_manager, key)

    @property
    def metadata(self) -> GalaxyMetadata:
        center = self.config["center"]
        return GalaxyMetadata(
            key=self.key,
            name=self.config["name"],
            aliases=tuple(self.config.get("aliases", [])),
            ra_deg=float(center["ra_deg"]),
            dec_deg=float(center["dec_deg"]),
            distance_modulus=float(self.config["distance_modulus"]),
            filters=tuple(self.config["filters"]),
        )

    def datasets(self, force: bool = False) -> DatasetBundle:
        prepared = self.data_manager.prepare(force=force)
        return DatasetBundle(
            science=Dataset(
                "science", prepared.science, pd.read_csv(prepared.science)
            ),
            confirmed=Dataset(
                "confirmed", prepared.confirmed, pd.read_csv(prepared.confirmed)
            ),
            background=Dataset(
                "background", prepared.background, pd.read_csv(prepared.background)
            ),
            manifest=prepared.manifest,
        )
