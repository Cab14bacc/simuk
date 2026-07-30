"""Parameter recovery plotting for simulation-based calibration."""

from importlib import import_module

import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.validate import validate_dict_argument, validate_or_use_rcparam
from arviz_plots import plot_ecdf_pit, style
from arviz_plots.plot_collection import PlotCollection, backend_from_object
from arviz_plots.plots.utils import get_visual_kwargs

style.use("arviz-variat")


def plot_ecdf(sbc, if_show=True):
    """Plot the empirical CDF of the SBC simulations."""
    if sbc.method == "posterior":
        fig = plot_ecdf_pit(
            sbc.simulations,
            group="posterior_sbc",
            visuals={"xlabel": False},
        )
    else:
        fig = plot_ecdf_pit(
            sbc.simulations,
            group="prior_sbc",
            visuals={"xlabel": False},
        )

    if if_show:
        fig.show()

    return fig


def _build_recovery_dataset(sbc, ci_prob, point_estimate, transform, var_names):
    """Build an xarray.Dataset suitable for arviz_plots faceting.

    The dataset contains a single variable ``recovery`` with dimensions
    ``(simulation, parameter, quantity)``.  The ``quantity`` coordinate has
    the values ``true``, ``estimate``, ``ci_low`` and ``ci_high``.  A
    companion boolean variable ``covered`` indicates whether each
    simulation's true value falls inside the credible interval.

    Using a flat ``parameter`` dimension avoids naming conflicts between
    data variables and their dimension coordinates when parameters are
    scalar.
    """
    ref_params = sbc.kept_simulation_params.ref_params
    n_sims = len(sbc.posteriors)
    alpha_lo = (1 - ci_prob) / 2
    alpha_hi = 1 - alpha_lo

    param_labels = []
    summaries_list = []
    covered_list = []

    for name in var_names:
        true_list, est_list, lo_list, hi_list = [], [], [], []

        for idx in range(n_sims):
            posterior = sbc.posteriors[idx]
            draws = posterior[name].transpose("sample", ...).values

            transformed_draws = np.array([transform(name, d) for d in draws])
            if point_estimate == "mean":
                est = np.mean(transformed_draws, axis=0)
            else:
                est = np.median(transformed_draws, axis=0)

            ci_lo = np.quantile(transformed_draws, alpha_lo, axis=0)
            ci_hi = np.quantile(transformed_draws, alpha_hi, axis=0)

            true_val = ref_params[name].isel(sample=idx).values
            transformed_true = np.asarray(transform(name, true_val), dtype=float)

            true_list.append(np.atleast_1d(transformed_true))
            est_list.append(np.atleast_1d(est))
            lo_list.append(np.atleast_1d(ci_lo))
            hi_list.append(np.atleast_1d(ci_hi))

        true_arr = np.array(true_list)
        est_arr = np.array(est_list)
        lo_arr = np.array(lo_list)
        hi_arr = np.array(hi_list)

        summaries = np.stack([true_arr, est_arr, lo_arr, hi_arr], axis=-1)
        parameter_dims = true_arr.shape[1:]
        n_elements = np.sum(parameter_dims, dtype=int)
        if n_elements == 1:
            labels = [name]
        elif len(parameter_dims) == 1:
            labels = [f"{name}[{i}]" for i in range(n_elements)]
        else:
            labels = [f"{name}[{str(idx).strip(')(')}]" for idx in np.ndindex(parameter_dims)]

        param_labels.extend(labels)
        summaries_list.append(summaries.reshape(n_sims, n_elements, 4))
        covered_list.append(
            ((lo_arr <= true_arr) & (true_arr <= hi_arr)).reshape(n_sims, n_elements)
        )

    recovery = xr.DataArray(
        np.concatenate(summaries_list, axis=1),
        dims=["simulation", "parameter", "quantity"],
        coords={
            "parameter": param_labels,
            "quantity": ["true", "estimate", "ci_low", "ci_high"],
        },
    )
    covered = xr.DataArray(
        np.concatenate(covered_list, axis=1),
        dims=["simulation", "parameter"],
        coords={"parameter": param_labels},
    )

    return xr.Dataset({"recovery": recovery, "covered": covered})


def plot_parameter_recovery(
    sbc,
    ci_prob=None,
    point_estimate=None,
    var_names=None,
    col_wrap=4,
    figsize=None,
    title_size="medium",
    backend=None,
    plot_collection=None,
    visuals=None,
    pc_kwargs=None,
    if_show=True,
):
    """Create a parameter recovery plot for an SBC object.

    For each parameter (scalar or flattened vector element), the plot shows the
    posterior point estimate against the true value, a vertical credible interval,
    and a 45-degree reference line.  Subplot titles report the observed coverage
    and its expected value (plus/minus two binomial standard errors).

    Parameters
    ----------
    sbc : simuk.SBC
        Fitted SBC object with posteriors and kept simulation parameters.
    ci_prob : float, optional
        Credible interval probability.  Defaults to ``rcParams["stats.ci_prob"]``
        (0.89).
    point_estimate : {"mean", "median"}, optional
        Point estimate to plot on the y-axis.  Defaults to
        ``rcParams["stats.point_estimate"]`` ("mean").
    var_names : str or list of str, optional
        Variable names to include in the plot.  If None, all parameters in
        ``sbc.kept_simulation_params.var_names`` are used.
    col_wrap : int, optional
        Maximum number of columns before wrapping to the next row.  Default is 4.
    figsize : tuple of float, optional
        Base figure size in inches for a single subplot.  Default is (6, 6).
    title_size : str or float, optional
        Font size for subplot titles.  Default is "medium".
    backend : str, optional
        Plotting backend.  Defaults to ``rcParams["plot.backend"]``.
    plot_collection : arviz_plots.PlotCollection, optional
        Existing plot collection to add to.  If None, a new one is created.
    visuals : dict, optional
        Visuals configuration.  Keys are visual names, values are kwargs dicts.
        Use ``False`` to disable a visual.  Default is ``{}``.

        Supported visuals
        -----------------
        - ``"labels"``: kwargs for ``labelled_x`` / ``labelled_y``
        - ``"reference_line"``: kwargs for the 45-degree ``dline``
        - ``"errorbar"``: kwargs for the per-simulation scatter + CI line
        - ``"title"``: kwargs for the subplot title
        - ``"legend"``: kwargs for the figure-level legend
    pc_kwargs : dict, optional
        Additional kwargs for ``PlotCollection.wrap``.  ``figure_kwargs`` is
        merged with the computed figsize.
    if_show : bool, optional
        Whether to display the figure.  Default is True.

    Returns
    -------
    arviz_plots.PlotCollection
        The PlotCollection wrapping the recovery plot.
    """
    if not getattr(sbc, "keep_fits", False):
        raise ValueError(
            "plot_parameter_recovery requires keep_fits=True. Re-run SBC with keep_fits=True."
        )

    if not sbc.posteriors:
        raise ValueError(
            "No posteriors found. Run sbc.run_simulations() with keep_fits=True first."
        )

    # Validate and set defaults
    ci_prob = validate_or_use_rcparam(ci_prob, "stats.ci_prob")
    point_estimate = validate_or_use_rcparam(point_estimate, "stats.point_estimate")
    if figsize is None:
        figsize = (6, 6)
    if backend is None:
        backend = rcParams["plot.backend"]
    pc_kwargs = pc_kwargs or {}
    visuals = validate_dict_argument(
        visuals, valid_keys=["labels", "reference_line", "errorbar", "title", "legend"]
    )

    # Resolve variable names
    all_var_names = sbc.kept_simulation_params.var_names
    if var_names is None:
        resolved_var_names = all_var_names
    elif isinstance(var_names, str):
        resolved_var_names = [var_names]
    else:
        resolved_var_names = list(var_names)
    # Validate that all requested names exist
    missing = set(resolved_var_names) - set(all_var_names)
    if missing:
        raise ValueError(
            f"var_names contains unknown variables: {sorted(missing)}. "
            f"Available: {list(all_var_names)}"
        )

    # Get backend module
    plot_bknd = import_module(f"arviz_plots.backend.{backend}")

    # Build recovery dataset
    ds = _build_recovery_dataset(
        sbc,
        ci_prob=ci_prob,
        point_estimate=point_estimate,
        transform=sbc._transform,
        var_names=resolved_var_names,
    )

    # Setup plot collection
    n_params = len(ds.coords["parameter"])
    n_cols = min(col_wrap, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    # figsize is per-subplot; compute total and scale to dots
    total_figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
    scaled_figsize = plot_bknd.scale_fig_size(
        figsize=total_figsize,
        figsize_units="inches",
    )

    # Merge figsize into pc_kwargs
    pc_kwargs = pc_kwargs.copy()
    figure_kwargs = pc_kwargs.setdefault("figure_kwargs", {})
    figure_kwargs["figsize"] = scaled_figsize
    figure_kwargs["figsize_units"] = "dots"

    if plot_collection is None:
        pc = PlotCollection.wrap(
            ds, cols=["parameter"], col_wrap=col_wrap, backend=backend, **pc_kwargs
        )
    else:
        pc = plot_collection

    # Apply visuals
    # Axis labels
    visuals_labels = get_visual_kwargs(visuals, "labels", {})
    if visuals_labels is not False:
        pc.map(azp.visuals.labelled_x, text="True Value", **visuals_labels)
        pc.map(azp.visuals.labelled_y, text="Posterior Mean", **visuals_labels)

    # Reference line
    visuals_ref_line = get_visual_kwargs(visuals, "reference_line", {})
    if visuals_ref_line is not False:

        def _facet_reference_line(da, target, **kwargs):
            """Draws a 45-degree line scaled exactly to this facet's specific data range."""
            plot_backend = backend_from_object(target)

            true_vals = da.sel(quantity="true")
            ci_low = da.sel(quantity="ci_low")
            ci_hi = da.sel(quantity="ci_high")

            # Now this min/max is local to the specific parameter in this subplot
            line_min = np.min([true_vals.min().item(), ci_low.min().item()])
            line_max = np.max([true_vals.max().item(), ci_hi.max().item()])

            # Using the backend to draw the line
            plot_backend.line(
                [line_min, line_max],
                [line_min, line_max],
                target,
                color="gray",
                linestyle="--",
                **kwargs,
            )

        pc.map(
            _facet_reference_line,
            data=ds["recovery"],
            **visuals_ref_line,
        )

    # Errorbars
    visuals_errorbar = get_visual_kwargs(visuals, "errorbar", {})
    if visuals_errorbar is not False:

        def _recovery_errorbar(da, target, covered, **kwargs):
            """Point estimate + vertical CI for one parameter facet."""
            plot_backend = backend_from_object(target)
            for sim in da.simulation.values:
                true_v = da.sel(simulation=sim, quantity="true").item()
                est_v = da.sel(simulation=sim, quantity="estimate").item()
                lo_v = da.sel(simulation=sim, quantity="ci_low").item()
                hi_v = da.sel(simulation=sim, quantity="ci_high").item()
                cov = covered.sel(simulation=sim).item()

                color = "teal" if cov else "purple"
                plot_backend.scatter([true_v], [est_v], target, color=color, **kwargs)
                plot_backend.line([true_v, true_v], [lo_v, hi_v], target, color=color, **kwargs)

        pc.map(
            _recovery_errorbar,
            data=ds["recovery"],
            covered=ds["covered"],
            **visuals_errorbar,
        )

    # Title
    visuals_title = get_visual_kwargs(visuals, "title", {})
    if visuals_title is not False:

        def _recovery_title(da, target, ci_prob, size, **kwargs):
            """Add a title with coverage statistics to a parameter facet."""
            plot_backend = backend_from_object(target)
            label = da.coords["parameter"].item()
            n_sims = da.sizes["simulation"]
            coverage = da.mean().item()
            se = np.sqrt(ci_prob * (1 - ci_prob) / n_sims)
            margin = 2 * se
            text = f"{label}\ncoverage: {coverage:.1%}\n(expected: {ci_prob:.1%} ± {margin:.1%})"
            return plot_backend.title(text, target, size=size, **kwargs)

        pc.map(
            _recovery_title,
            data=ds["covered"],
            ci_prob=ci_prob,
            size=title_size,
            **visuals_title,
        )

    # Legend (figure-level)
    visuals_legend = get_visual_kwargs(visuals, "legend", {})
    if visuals_legend is not False:
        plot_bknd.legend(
            pc,
            kwarg_list=[
                {"color": "teal", "linestyle": "-", "width": 2},
                {"color": "purple", "linestyle": "-", "width": 2},
            ],
            label_list=["Covered", "Not covered"],
            title="Credible interval",
            **visuals_legend,
        )

    if if_show:
        pc.show()

    return pc
