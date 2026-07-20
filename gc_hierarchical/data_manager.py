from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigManager
from .exceptions import DataValidationError
from .validation import require_valid_dataframe


@dataclass(frozen=True)
class PreparedData:
    science: Path
    confirmed: Path
    background: Path
    manifest: Path
    science_rows: int
    confirmed_rows: int
    background_rows: int


class DataManager:
    def __init__(self, config: ConfigManager, galaxy: str):
        self.config = config
        self.galaxy = galaxy
        self.entry = config.galaxy_entry(galaxy)
        self.directory = config.galaxy_directory(galaxy)
        self.selection = config.selection_config(galaxy)

        self.raw_candidate = self.directory / self.entry["raw_candidate_catalogue"]
        self.raw_confirmed = self.directory / self.entry["raw_confirmed_catalogue"]
        self.processed = self.directory / self.entry["processed_directory"]
        self.cache = self.directory / self.entry["cache_directory"]
        self.candidate_schema_path = self.directory / self.entry["candidate_schema"]
        self.confirmed_schema_path = self.directory / self.entry["confirmed_schema"]

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _angular_radius_arcmin(
        ra_deg: pd.Series,
        dec_deg: pd.Series,
        ra0_deg: float,
        dec0_deg: float,
    ) -> np.ndarray:
        ra = np.deg2rad(pd.to_numeric(ra_deg, errors="coerce").to_numpy())
        dec = np.deg2rad(pd.to_numeric(dec_deg, errors="coerce").to_numpy())
        ra0 = np.deg2rad(ra0_deg)
        dec0 = np.deg2rad(dec0_deg)
        cosine = (
            np.sin(dec) * np.sin(dec0)
            + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
        )
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.rad2deg(angle) * 60.0

    def validate_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        for path in (
            self.raw_candidate,
            self.raw_confirmed,
            self.candidate_schema_path,
            self.confirmed_schema_path,
        ):
            if not path.is_file():
                raise DataValidationError(f"Required input not found: {path}")

        candidate = pd.read_csv(self.raw_candidate)
        confirmed = pd.read_csv(self.raw_confirmed)
        require_valid_dataframe(
            candidate, self._load_json(self.candidate_schema_path), self.raw_candidate
        )
        require_valid_dataframe(
            confirmed, self._load_json(self.confirmed_schema_path), self.raw_confirmed
        )
        return candidate, confirmed

    def _normalize_candidate(self, candidate: pd.DataFrame) -> pd.DataFrame:
        output = candidate.copy()
        output["Rank"] = output["Rank"].astype(str).str.lower().str.strip()
        numeric = ["RAdeg", "DEdeg", "gmag", "e_gmag", "rmag", "e_rmag", "TotL"]
        for column in numeric:
            output[column] = pd.to_numeric(output[column], errors="coerce")

        output["color"] = output["gmag"] - output["rmag"]
        galaxy_config = self.config.galaxy_config(self.galaxy)
        center = galaxy_config["center"]
        output["radius_arcmin"] = self._angular_radius_arcmin(
            output["RAdeg"],
            output["DEdeg"],
            center["ra_deg"],
            center["dec_deg"],
        )
        output["log_radius"] = np.log10(output["radius_arcmin"])
        return output

    @staticmethod
    def _observable_window(frame: pd.DataFrame, science: dict[str, Any]) -> pd.Series:
        return (
            frame["gmag"].between(*science["g_range"])
            & frame["rmag"].between(*science["r_range"])
            & frame["color"].between(*science["color_range"])
            & frame["radius_arcmin"].between(*science["radius_range_arcmin"])
        )

    def prepare(self, force: bool = False) -> PreparedData:
        candidate, confirmed = self.validate_raw()
        candidate = self._normalize_candidate(candidate)

        science_cfg = self.selection["science"]
        background_cfg = self.selection["background"]
        window = self._observable_window(candidate, science_cfg)

        science_mask = (
            candidate["Rank"].isin(science_cfg["ranks"])
            & (candidate["TotL"] >= science_cfg["minimum_total_likelihood"])
            & window
        )
        science = candidate.loc[science_mask].dropna(
            subset=["gmag", "rmag", "color", "radius_arcmin"]
        ).reset_index(drop=True)

        background_mask = candidate["Rank"].isin(background_cfg["ranks"])
        if background_cfg.get("apply_observable_window", True):
            background_mask &= window
        background = candidate.loc[background_mask].dropna(
            subset=["gmag", "rmag", "color", "radius_arcmin"]
        ).copy()

        maximum = int(background_cfg["maximum_objects"])
        if len(background) > maximum:
            background = background.sample(
                n=maximum,
                random_state=int(background_cfg["random_seed"]),
            )
        background = background.reset_index(drop=True)

        self.processed.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)

        science_path = self.processed / "science.csv"
        confirmed_path = self.processed / "confirmed.csv"
        background_path = self.processed / "background.csv"
        manifest_path = self.processed / "manifest.json"

        science.to_csv(science_path, index=False)
        confirmed.to_csv(confirmed_path, index=False)
        background.to_csv(background_path, index=False)

        manifest = {
            "galaxy": self.galaxy,
            "raw_inputs": {
                str(self.raw_candidate.relative_to(self.config.root)): self._sha256(
                    self.raw_candidate
                ),
                str(self.raw_confirmed.relative_to(self.config.root)): self._sha256(
                    self.raw_confirmed
                ),
            },
            "selection_config_sha256": self._sha256(
                self.directory / self.entry["selection_config"]
            ),
            "outputs": {
                "science.csv": len(science),
                "confirmed.csv": len(confirmed),
                "background.csv": len(background),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        return PreparedData(
            science=science_path,
            confirmed=confirmed_path,
            background=background_path,
            manifest=manifest_path,
            science_rows=len(science),
            confirmed_rows=len(confirmed),
            background_rows=len(background),
        )
