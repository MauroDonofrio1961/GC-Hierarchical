from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    project_config: Path
    results_root: Path


class ConfigManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.project_path = self.root / "config" / "project.json"
        self.project = self._read_json(self.project_path)
        self._validate_project()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc

    def _validate_project(self) -> None:
        for key in ("project", "galaxies"):
            if key not in self.project:
                raise ConfigurationError(
                    f"Missing top-level key '{key}' in {self.project_path}"
                )

    @property
    def paths(self) -> ProjectPaths:
        results = self.root / self.project["project"]["results_root"]
        return ProjectPaths(self.root, self.project_path, results)

    def galaxy_entry(self, galaxy: str) -> dict[str, Any]:
        try:
            return self.project["galaxies"][galaxy]
        except KeyError as exc:
            known = ", ".join(sorted(self.project["galaxies"]))
            raise ConfigurationError(
                f"Unknown galaxy '{galaxy}'. Available: {known}"
            ) from exc

    def galaxy_directory(self, galaxy: str) -> Path:
        return (self.root / self.galaxy_entry(galaxy)["directory"]).resolve()

    def galaxy_config(self, galaxy: str) -> dict[str, Any]:
        entry = self.galaxy_entry(galaxy)
        return self._read_json(self.galaxy_directory(galaxy) / entry["galaxy_config"])

    def selection_config(self, galaxy: str) -> dict[str, Any]:
        entry = self.galaxy_entry(galaxy)
        return self._read_json(
            self.galaxy_directory(galaxy) / entry["selection_config"]
        )
