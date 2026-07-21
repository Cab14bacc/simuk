"""Parameter recovery plotting for simulation-based calibration."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from arviz_plots import plot_ecdf_pit, style

style.use("arviz-variat")


def _extract_draws(posterior, name):
    """Extract posterior draws for a parameter as a numpy array.

    Handles both PyMC-style posteriors (with a ``sample`` dimension from
    ``arviz_base.extract``) and NumPyro-style posteriors (with ``chain``
    and ``draw`` dimensions from ``arviz_base.from_numpyro``).

    Parameters
    ----------
    posterior : xarray.Dataset
        Posterior draws for a single SBC iteration.
    name : str
        Parameter name.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_draws, *param_shape)``.
    """
    var = posterior[name]
    if "sample" in var.dims:
        return np.asarray(var.values)
    else:
        stacked = var.stack(__sample__=("chain", "draw"))
        return np.asarray(stacked.transpose("__sample__", ...).values)


def _extract_true_value(ref_params, name, idx):
    """Extract the true (reference) parameter value for a simulation.

    Parameters
    ----------
    ref_params : xarray.Dataset or dict
        Reference parameters. An xarray Dataset (PyMC) or a plain dict of
        arrays (NumPyro).
    name : str
        Parameter name.
    idx : int
        Simulation index.

    Returns
    -------
    numpy.ndarray
        The true parameter value (0-d for scalars, 1-d+ for vectors).
    """
    if hasattr(ref_params, "data_vars"):
        return np.asarray(ref_params[name].isel(sample=idx).values)
    else:
        return np.asarray(ref_params[name][idx])


def plot_parameter_recovery(
    sbc,
    ci_prob=0.94,
    point_estimate="mean",
    transform=None,
    var_names=None,
    figsize=None,
    axes=None,
    if_show=True,
):
    r"""Plot parameter recovery: posterior point estimate vs true simulated value.

    For each SBC iteration the true (reference) parameter value is plotted
    on the x-axis and the posterior point estimate (mean or median) on the
    y-axis, with credible-interval error bars.  Points whose credible
    interval covers the true value are coloured blue; misses are red.

    A *y = x* diagonal reference line indicates perfect recovery.

    Parameters
    ----------
    sbc : simuk.SBC
        An SBC instance that has completed simulations with
        ``keep_fits=True``.
    ci_prob : float, default 0.94
        Probability mass of the credible interval (e.g. 0.94 for a 94 %
        CI).  The interval is computed from posterior quantiles — no
        distributional assumptions (e.g. normality) are made.
    point_estimate : {"mean", "median"}, default "mean"
        Which posterior point estimate to plot on the y-axis.
    transform : callable, optional
        ``(param_name, param_value) -> transformed_value``.  Applied to
        both true values and posterior draws before summarising.  Defaults
        to the transform set at SBC initialisation.
    var_names : list[str], optional
        Subset of parameter names to plot.  Defaults to all free
        parameters.
    figsize : tuple[float, float], optional
        Figure size in inches.
    axes : matplotlib Axes or array of Axes, optional
        Pre-existing axes to draw on (one per scalar subplot).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the parameter recovery subplot(s).

    Notes
    -----
    **Credible intervals** are quantile-based: for a 94 % CI the lower
    and upper bounds are the 3rd and 97th percentiles of the posterior
    draws.

    **Coverage** reported in subplot titles is the empirical fraction of
    simulations whose CI contains the true value.  The "expected" range
    uses the binomial standard error

    .. math::

        \mathrm{SE} = \sqrt{\frac{p\,(1-p)}{N}}

    which accounts only for the sampling noise from having a finite number
    *N* of SBC iterations.  This is a **lower bound** on the true
    uncertainty: it ignores the additional noise introduced by estimating
    each CI from a finite number of MCMC draws.  When the number of
    posterior draws per fit is small, the actual uncertainty in empirical
    coverage will be larger than the displayed ± range.
    """
    # ---- validation --------------------------------------------------------
    if not sbc.keep_fits:
        raise ValueError(
            "`plot_parameter_recovery` requires the SBC instance to have "
            "been run with `keep_fits=True`."
        )
    if not sbc.posteriors:
        raise ValueError("No posteriors found. Call `sbc.run_simulations()` before plotting.")
    if point_estimate not in ("mean", "median"):
        raise ValueError(f"`point_estimate` must be 'mean' or 'median', got '{point_estimate}'")
    if transform is None:
        transform = sbc._transform
    elif not callable(transform):
        raise ValueError("`transform` should be a function or None")

    ref_params = sbc.kept_simulation_params.ref_params
    if var_names is None:
        var_names = list(sbc.kept_simulation_params.var_names)

    n_sims = len(sbc.posteriors)
    alpha_lo = (1 - ci_prob) / 2
    alpha_hi = 1 - alpha_lo

    # ---- compute summaries -------------------------------------------------
    summaries = {}
    for name in var_names:
        true_list, est_list, lo_list, hi_list = [], [], [], []

        for idx in range(n_sims):
            posterior = sbc.posteriors[idx]
            draws = _extract_draws(posterior, name)
            transformed_draws = np.array([transform(name, d) for d in draws])

            if point_estimate == "mean":
                est = np.mean(transformed_draws, axis=0)
            else:
                est = np.median(transformed_draws, axis=0)

            ci_lo = np.quantile(transformed_draws, alpha_lo, axis=0)
            ci_hi = np.quantile(transformed_draws, alpha_hi, axis=0)

            true_val = _extract_true_value(ref_params, name, idx)
            transformed_true = np.asarray(transform(name, true_val), dtype=float)

            true_list.append(np.atleast_1d(transformed_true))
            est_list.append(np.atleast_1d(est))
            lo_list.append(np.atleast_1d(ci_lo))
            hi_list.append(np.atleast_1d(ci_hi))

        summaries[name] = {
            "true_values": np.array(true_list),
            "point_estimates": np.array(est_list),
            "ci_lower": np.array(lo_list),
            "ci_upper": np.array(hi_list),
        }

    # ---- determine subplot layout ------------------------------------------
    subplot_specs = []  # (param_name, column_index, display_label)
    for name in var_names:
        n_elements = summaries[name]["true_values"].shape[1]
        if n_elements == 1:
            subplot_specs.append((name, 0, name))
        else:
            for j in range(n_elements):
                subplot_specs.append((name, j, f"{name}[{j}]"))

    n_subplots = len(subplot_specs)

    # ---- create or validate axes -------------------------------------------
    created_figure = axes is None
    if created_figure:
        n_cols = min(3, n_subplots)
        n_rows = -(-n_subplots // n_cols)  # ceil division
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes_grid.flatten()
        for k in range(n_subplots, len(axes_flat)):
            axes_flat[k].set_visible(False)
    else:
        axes_flat = np.atleast_1d(axes).flatten()
        if len(axes_flat) < n_subplots:
            raise ValueError(f"Need {n_subplots} axes but only {len(axes_flat)} provided.")
        fig = axes_flat[0].get_figure()

    # ---- plot each subplot -------------------------------------------------
    binomial_se = np.sqrt(ci_prob * (1 - ci_prob) / n_sims)

    for i, (name, col, label) in enumerate(subplot_specs):
        ax = axes_flat[i]
        s = summaries[name]

        true_vals = s["true_values"][:, col]
        estimates = s["point_estimates"][:, col]
        ci_lo = s["ci_lower"][:, col]
        ci_hi = s["ci_upper"][:, col]

        covered = (ci_lo <= true_vals) & (true_vals <= ci_hi)
        empirical_cov = covered.mean()

        yerr = np.array(
            [
                np.maximum(estimates - ci_lo, 0),
                np.maximum(ci_hi - estimates, 0),
            ]
        )

        # Plot covered and missed points separately for the legend
        mask_cov = covered
        mask_miss = ~covered

        if mask_cov.any():
            ax.errorbar(
                true_vals[mask_cov],
                estimates[mask_cov],
                yerr=yerr[:, mask_cov],
                fmt="o",
                color="C0",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                capsize=2,
                label="Covered",
            )
        if mask_miss.any():
            ax.errorbar(
                true_vals[mask_miss],
                estimates[mask_miss],
                yerr=yerr[:, mask_miss],
                fmt="o",
                color="C3",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                capsize=2,
                label="Missed",
            )

        # y = x reference line
        all_vals = np.concatenate([true_vals, estimates, ci_lo, ci_hi])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05 if vmax > vmin else 0.5
        ax.plot(
            [vmin - margin, vmax + margin],
            [vmin - margin, vmax + margin],
            "k--",
            alpha=0.4,
            linewidth=1,
            zorder=0,
        )

        ax.set_xlabel("True value")
        ax.set_ylabel(f"Posterior {point_estimate}")
        ax.set_title(
            f"{label} — coverage: {empirical_cov:.1%} "
            f"(expected: {ci_prob:.1%} ± {2 * binomial_se:.1%})",
            fontsize=9,
            wrap=True,
        )
        if mask_miss.any():
            ax.legend(fontsize="small")

    if created_figure:
        fig.tight_layout()

    if if_show:
        plt.show()

    return fig


def plot_ecdf(sbc, if_show=True):
    if not if_show:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    if sbc.method == "posterior":
        fig = plot_ecdf_pit(
            sbc.simulations,
            group="posterior_sbc",
            visuals={"xlabel": False},
        )
    else:
        fig = plot_ecdf_pit(
            sbc.simulations,
            visuals={"xlabel": False},
        )

    if if_show:
        plt.show()

    return fig
