from __future__ import annotations

from .galaxy import Galaxy
from .models import HierarchicalGCModel


def create_model(
    galaxy: Galaxy,
    model_name: str = "hierarchical_gc",
    *,
    mode: str = "science",
):
    aliases = {
        "hierarchical_gc",
        "three_component_v5",
        "NGC5128-three-component-v5",
    }
    if model_name not in aliases:
        raise ValueError(f"Unknown model '{model_name}'")
    return HierarchicalGCModel.from_galaxy(galaxy, mode=mode)
