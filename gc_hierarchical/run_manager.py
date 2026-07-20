from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import ConfigManager
from .galaxy import Galaxy
from .logging_utils import configure_logging
from .model_registry import create_model
from .models import HierarchicalGCModel


class RunManager:
    def __init__(self, root: str | Path, galaxy: str):
        self.config = ConfigManager(root)
        self.galaxy_key = galaxy
        self.galaxy = Galaxy(galaxy, root)

    def prepare(self, force: bool = False):
        log_path = self.config.paths.results_root / self.galaxy_key / "prepare.log"
        logger = configure_logging(
            log_path,
            self.config.project["project"].get("log_level", "INFO"),
        )
        logger.info("Preparing data for %s", self.galaxy_key)
        prepared = self.galaxy.data_manager.prepare(force=force)
        logger.info(
            "Prepared science=%d, confirmed=%d, background=%d rows",
            prepared.science_rows,
            prepared.confirmed_rows,
            prepared.background_rows,
        )
        return prepared

    def create_run_directory(self, mode: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            self.config.paths.results_root
            / self.galaxy_key
            / f"{timestamp}_{mode}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def run_validated_v5(self, mode: str) -> int:
        if mode == "check":
            HierarchicalGCModel.verify_fsps()
            print("FSPS Python import passed.")
            return 0
        if mode not in {"quick", "science", "mcmc"}:
            raise ValueError(f"Unsupported mode: {mode}")

        run_dir = self.create_run_directory(mode)
        logger = configure_logging(run_dir / "run.log")
        model = create_model(self.galaxy, mode=mode)
        logger.info(
            "Building %s for %s through the Step 3 object model",
            model.name,
            self.galaxy_key,
        )
        model.build()
        result = model.fit(mode=mode, output_directory=run_dir)

        (run_dir / "fit_result.json").write_text(
            json.dumps(
                {
                    "model_name": result.model_name,
                    "mode": result.mode,
                    "log_posterior": result.log_posterior,
                    "calls": result.calls,
                    "parameters": result.parameters,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result.raw_result, indent=2))
        print(f"Results: {run_dir.resolve()}")
        return 0
