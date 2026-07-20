from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigManager
from .run_manager import RunManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GC-Hierarchical",
        description="Reusable globular-cluster hierarchical forward-modelling framework.",
    )
    parser.add_argument(
        "command",
        choices=["prepare", "validate", "quick", "science", "mcmc", "check"],
        nargs="?",
        default="prepare",
    )
    parser.add_argument("--galaxy", default=None)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ConfigManager(args.root)
    galaxy = args.galaxy or config.project["project"]["default_galaxy"]
    manager = RunManager(args.root, galaxy)

    if args.command in {"prepare", "validate"}:
        prepared = manager.prepare(force=args.force)
        print()
        print("Data preparation completed.")
        print(f"Science catalogue:    {prepared.science} ({prepared.science_rows} rows)")
        print(f"Confirmed catalogue:  {prepared.confirmed} ({prepared.confirmed_rows} rows)")
        print(f"Background catalogue: {prepared.background} ({prepared.background_rows} rows)")
        print(f"Manifest:             {prepared.manifest}")
        return 0

    return manager.run_validated_v5(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
