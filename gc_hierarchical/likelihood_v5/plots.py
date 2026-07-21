from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _save(fig, out: Path, name: str, book: PdfPages) -> None:
    """Save one diagnostic as PNG, PDF, and one page of the figure book."""
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    book.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _parameter_dict(cfg: dict, theta: np.ndarray) -> dict[str, float]:
    names = cfg["parameters"]["names"]
    return {name: float(value) for name, value in zip(names, theta)}


def _normalised_grid_weights(grid: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-6)
    x = (np.asarray(grid, dtype=float) - float(mean)) / sigma
    weights = np.exp(-0.5 * x * x)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid latent-grid weights while generating diagnostics.")
    return weights / total


def _draw_selected_gc(
    component,
    *,
    n: int,
    radius: np.ndarray,
    e_gmag: np.ndarray,
    e_rmag: np.ndarray,
    logmass_mean: float,
    logmass_sigma: float,
    logz_mean: float,
    logz_sigma: float,
    delta_g: float,
    delta_r: float,
    extra_color_scatter: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw a selected GC population from the same latent grid used by the
    likelihood.  Selection is applied conditionally at each sampled radius.
    """
    if n == 0:
        return pd.DataFrame(columns=["gmag", "rmag", "color", "radius_arcmin"])

    radius = np.asarray(radius, dtype=float)
    e_gmag = np.asarray(e_gmag, dtype=float)
    e_rmag = np.asarray(e_rmag, dtype=float)

    wm = _normalised_grid_weights(
        component.logm, logmass_mean, logmass_sigma
    )
    wz = _normalised_grid_weights(
        component.zgrid, logz_mean, logz_sigma
    )

    latent_weights = np.multiply.outer(wm, wz).ravel()
    g_grid = (
        component.Mssp[:, 0][None, :]
        - 2.5 * component.logm[:, None]
        + component.dm
        + component.Ag
        + delta_g
    ).ravel()
    r_grid = (
        component.Mssp[:, 1][None, :]
        - 2.5 * component.logm[:, None]
        + component.dm
        + component.Ar
        + delta_r
    ).ravel()

    g_out = np.empty(n, dtype=float)
    r_out = np.empty(n, dtype=float)

    # Each draw is conditional on the radius of a bootstrapped observed object.
    # This avoids making the CMD/LF comparison depend on a separate radial draw.
    for i in range(n):
        selection = component.selection.probability(
            r_grid, np.full(r_grid.size, radius[i], dtype=float)
        )
        probabilities = latent_weights * selection
        probabilities /= probabilities.sum()
        j = rng.choice(r_grid.size, p=probabilities)

        sg = np.sqrt(max(e_gmag[i], 0.008) ** 2 + 0.5 * extra_color_scatter**2)
        sr = np.sqrt(max(e_rmag[i], 0.008) ** 2 + 0.5 * extra_color_scatter**2)
        g_out[i] = rng.normal(g_grid[j], sg)
        r_out[i] = rng.normal(r_grid[j], sr)

    return pd.DataFrame(
        {
            "gmag": g_out,
            "rmag": r_out,
            "color": g_out - r_out,
            "radius_arcmin": radius,
        }
    )


def _draw_background(background_density, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw background objects from the fitted GMM in observable space."""
    if n == 0:
        return pd.DataFrame(columns=["gmag", "rmag", "color", "radius_arcmin"])

    z, _ = background_density.gmm.sample(n)
    # sklearn's sample uses the model random state. Shuffle deterministically
    # with our run RNG so repeated population blocks do not retain ordering.
    z = z[rng.permutation(len(z))]
    x = background_density.scaler.inverse_transform(z)
    radius = np.power(10.0, x[:, 2])
    return pd.DataFrame(
        {
            "gmag": x[:, 0],
            "rmag": x[:, 1],
            "color": x[:, 0] - x[:, 1],
            "radius_arcmin": radius,
        }
    )


def generate_bestfit_catalogue(
    cfg: dict,
    science: pd.DataFrame,
    theta: np.ndarray,
    *,
    cluster_population,
    background_density,
    posterior,
    size: int | None = None,
) -> pd.DataFrame:
    """
    Generate a posterior-predictive catalogue at the best-fit parameter vector.

    The synthetic catalogue has the same Gold/Silver rank mixture and samples
    the observed radii and photometric uncertainties by bootstrap.  GC draws
    use the exact latent mass-metallicity grids and selection function used by
    the likelihood; background draws use the fitted background GMM.
    """
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 104729)
    n_observed = len(science)
    n = int(size or max(5000, 5 * n_observed))

    sampled_index = rng.integers(0, n_observed, size=n)
    template = science.iloc[sampled_index].reset_index(drop=True)
    ranks = template["Rank"].astype(str).str.lower().to_numpy()

    gold, silver = posterior.rank_fractions(np.asarray(theta, dtype=float))
    probabilities = np.empty((n, 3), dtype=float)
    probabilities[ranks == "gold"] = gold
    probabilities[ranks != "gold"] = silver

    u = rng.random(n)
    component_code = np.where(
        u < probabilities[:, 0],
        0,
        np.where(u < probabilities[:, 0] + probabilities[:, 1], 1, 2),
    )

    pars = _parameter_dict(cfg, theta)
    e_g = (
        pd.to_numeric(template.get("e_gmag", 0.04), errors="coerce")
        .fillna(0.04)
        .to_numpy(dtype=float)
    )
    e_r = (
        pd.to_numeric(template.get("e_rmag", 0.03), errors="coerce")
        .fillna(0.03)
        .to_numpy(dtype=float)
    )
    radius = template["radius_arcmin"].to_numpy(dtype=float)

    pieces: list[pd.DataFrame] = []
    for code, label, component, prefix in (
        (0, "blue", cluster_population.blue, "blue"),
        (1, "red", cluster_population.red, "red"),
    ):
        mask = component_code == code
        draw = _draw_selected_gc(
            component,
            n=int(mask.sum()),
            radius=radius[mask],
            e_gmag=e_g[mask],
            e_rmag=e_r[mask],
            logmass_mean=pars[f"logM0_{prefix}"],
            logmass_sigma=pars[f"sigma_{prefix}"],
            logz_mean=pars[f"{prefix}_logz_mean"],
            logz_sigma=pars[f"{prefix}_logz_sigma"],
            delta_g=pars["delta_g"],
            delta_r=pars["delta_r"],
            extra_color_scatter=pars["extra_color_scatter"],
            rng=rng,
        )
        draw["component"] = label
        pieces.append(draw)

    background = _draw_background(
        background_density, int((component_code == 2).sum()), rng
    )
    background["component"] = "background"
    pieces.append(background)

    synthetic = pd.concat(pieces, ignore_index=True)
    synthetic = synthetic.sample(
        frac=1.0, random_state=int(cfg["project"]["seed"]) + 271
    ).reset_index(drop=True)

    # Restrict the predictive display to the same observable window as the fit.
    sample = cfg["sample"]
    keep = (
        synthetic["gmag"].between(*sample["g_range"])
        & synthetic["rmag"].between(*sample["r_range"])
        & synthetic["color"].between(*sample["color_range"])
        & synthetic["radius_arcmin"].between(*sample["radius_range_arcmin"])
    )
    synthetic = synthetic.loc[keep].reset_index(drop=True)
    synthetic.to_csv(
        Path(cfg["_diagnostic_output"]) / "bestfit_predictive_catalogue.csv",
        index=False,
    )
    return synthetic


def _hist_comparison(
    observed: np.ndarray,
    model: np.ndarray,
    *,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    invert_x: bool = False,
):
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    obs_count, edges = np.histogram(observed, bins=bins)
    model_count, _ = np.histogram(model, bins=edges)
    if model_count.sum() > 0:
        model_count = model_count * (obs_count.sum() / model_count.sum())

    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    errors = np.sqrt(np.maximum(obs_count, 1.0))

    ax.errorbar(
        centres,
        obs_count,
        yerr=errors,
        fmt="o",
        markersize=4,
        capsize=2,
        label="Observed",
    )
    ax.step(edges, np.r_[model_count, model_count[-1]], where="post", linewidth=2, label="Best-fit model")
    ax.set(ylabel="Objects per bin", title=title)
    ax.legend()

    residual = (obs_count - model_count) / errors
    residual_ax.axhline(0.0, linewidth=1)
    residual_ax.bar(centres, residual, width=widths, alpha=0.65)
    residual_ax.set(xlabel=xlabel, ylabel=r"$(N_{\rm obs}-N_{\rm mod})/\sigma$")
    residual_ax.set_ylim(-5, 5)

    if invert_x:
        residual_ax.invert_xaxis()
    return fig


def make_plots(
    cfg,
    science,
    background,
    theta,
    membership,
    out,
    chain=None,
    *,
    cluster_population=None,
    background_density=None,
    posterior=None,
):
    out = Path(out)
    d = science.copy()
    d["P_blue"] = membership[:, 0]
    d["P_red"] = membership[:, 1]
    d["P_background"] = membership[:, 2]
    d["P_GC"] = d["P_blue"] + d["P_red"]
    d.to_csv(out / "object_membership_probabilities.csv", index=False)

    predictive = None
    if (
        cluster_population is not None
        and background_density is not None
        and posterior is not None
    ):
        diagnostic_cfg = dict(cfg)
        diagnostic_cfg["_diagnostic_output"] = str(out)
        predictive = generate_bestfit_catalogue(
            diagnostic_cfg,
            d,
            np.asarray(theta, dtype=float),
            cluster_population=cluster_population,
            background_density=background_density,
            posterior=posterior,
        )

    with PdfPages(out / "all_figures_v5.pdf") as book:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            d.color,
            d.rmag,
            c=d.P_background,
            s=14,
            vmin=0,
            vmax=1,
        )
        ax.invert_yaxis()
        ax.set(
            xlabel="g − r (AB)",
            ylabel="r (AB)",
            title="Posterior background probability",
        )
        fig.colorbar(sc, ax=ax, label="P(background | data)")
        _save(fig, out, "01_background_membership_cmd", book)

        fig, ax = plt.subplots(figsize=(7, 6))
        rgb = np.column_stack([d.P_red, d.P_blue, np.zeros(len(d))])
        ax.scatter(d.color, d.rmag, c=np.clip(rgb, 0, 1), s=14)
        ax.invert_yaxis()
        ax.set(
            xlabel="g − r (AB)",
            ylabel="r (AB)",
            title="Blue and red GC membership",
        )
        _save(fig, out, "02_blue_red_membership_cmd", book)

        for col, xlabel, name, rng in [
            ("rmag", "r (AB)", "03_r_membership", cfg["sample"]["r_range"]),
            (
                "color",
                "g − r",
                "04_color_membership",
                cfg["sample"]["color_range"],
            ),
            (
                "radius_arcmin",
                "Radius (arcmin)",
                "05_radius_membership",
                cfg["sample"]["radius_range_arcmin"],
            ),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            bins = np.linspace(*rng, 26)
            ax.hist(
                d[col],
                bins=bins,
                weights=d.P_blue,
                density=True,
                histtype="step",
                linewidth=2,
                label="Blue GC",
            )
            ax.hist(
                d[col],
                bins=bins,
                weights=d.P_red,
                density=True,
                histtype="step",
                linewidth=2,
                label="Red GC",
            )
            ax.hist(
                d[col],
                bins=bins,
                weights=d.P_background,
                density=True,
                histtype="step",
                linewidth=1.5,
                label="Background",
            )
            if col == "rmag":
                ax.invert_xaxis()
            ax.set(
                xlabel=xlabel,
                ylabel="Density",
                title=xlabel + " posterior decomposition",
            )
            ax.legend()
            _save(fig, out, name, book)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            d.P_background,
            bins=np.linspace(0, 1, 21),
            histtype="step",
            linewidth=2,
        )
        ax.set(
            xlabel="P(background | data)",
            ylabel="Objects",
            title="Background-membership distribution",
        )
        _save(fig, out, "06_background_probability_histogram", book)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d.TotL, d.P_background, s=10, alpha=0.45)
        ax.set(
            xlabel="Catalogue TotL",
            ylabel="P(background | data)",
            title="Background probability versus TotL",
        )
        _save(fig, out, "07_background_vs_TotL", book)

        if chain is not None:
            try:
                import corner

                fig = corner.corner(
                    chain,
                    labels=cfg["parameters"]["names"],
                    show_titles=True,
                )
                _save(fig, out, "08_corner", book)
            except Exception:
                pass

        if predictive is not None and len(predictive):
            # Direct CMD overlay. The predictive population is plotted lightly
            # so the observed catalogue remains visible.
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            ax.scatter(
                predictive["color"],
                predictive["rmag"],
                s=6,
                alpha=0.12,
                label="Best-fit predictive model",
                rasterized=True,
            )
            ax.scatter(
                d["color"],
                d["rmag"],
                s=15,
                facecolors="none",
                edgecolors="black",
                linewidths=0.55,
                label="Observed",
            )
            ax.set(
                xlim=cfg["sample"]["color_range"],
                ylim=cfg["sample"]["r_range"][::-1],
                xlabel="g − r (AB)",
                ylabel="r (AB)",
                title="Observed CMD and best-fit predictive model",
            )
            ax.legend(loc="best")
            _save(fig, out, "09_cmd_observed_vs_bestfit", book)

            fig = _hist_comparison(
                d["gmag"].to_numpy(dtype=float),
                predictive["gmag"].to_numpy(dtype=float),
                bins=np.linspace(*cfg["sample"]["g_range"], 26),
                xlabel="g (AB)",
                title="g-band luminosity function",
                invert_x=True,
            )
            _save(fig, out, "10_lf_g_observed_vs_bestfit", book)

            fig = _hist_comparison(
                d["rmag"].to_numpy(dtype=float),
                predictive["rmag"].to_numpy(dtype=float),
                bins=np.linspace(*cfg["sample"]["r_range"], 26),
                xlabel="r (AB)",
                title="r-band luminosity function",
                invert_x=True,
            )
            _save(fig, out, "11_lf_r_observed_vs_bestfit", book)

            fig = _hist_comparison(
                d["color"].to_numpy(dtype=float),
                predictive["color"].to_numpy(dtype=float),
                bins=np.linspace(*cfg["sample"]["color_range"], 26),
                xlabel="g − r (AB)",
                title="Colour distribution",
            )
            _save(fig, out, "12_colour_observed_vs_bestfit", book)

            fig = _hist_comparison(
                d["radius_arcmin"].to_numpy(dtype=float),
                predictive["radius_arcmin"].to_numpy(dtype=float),
                bins=np.linspace(*cfg["sample"]["radius_range_arcmin"], 26),
                xlabel="Radius (arcmin)",
                title="Projected radial distribution",
            )
            _save(fig, out, "13_radius_observed_vs_bestfit", book)
