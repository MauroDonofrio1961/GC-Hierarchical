from __future__ import annotations

from .base import ComponentFactory, ModelComponents
from ..datasets import DatasetBundle
from ..likelihood_v5.background import BackgroundDensity
from ..likelihood_v5.fsps_grid import FSPSGrid
from ..likelihood_v5.hierarchical import ThreeComponentDensity
from ..likelihood_v5.posterior import Posterior
from ..likelihood_v5.selection import SelectionModel


class V5ComponentFactory(ComponentFactory):
    """Construct the frozen v5 scientific component graph."""

    def build(
        self,
        configuration: dict,
        datasets: DatasetBundle,
    ) -> ModelComponents:
        selection = SelectionModel(configuration).fit(
            datasets.confirmed.table
        )
        stellar_population = FSPSGrid(configuration)
        background = BackgroundDensity(
            configuration,
            datasets.science.table,
            datasets.background.table,
        )
        cluster_population = ThreeComponentDensity(
            configuration,
            datasets.science.table,
            selection,
            stellar_population,
        )
        posterior = Posterior(
            configuration,
            datasets.science.table,
            cluster_population,
            background,
        )
        return ModelComponents(
            selection=selection,
            stellar_population=stellar_population,
            background=background,
            cluster_population=cluster_population,
            posterior=posterior,
        )
