import arviz_plots as azp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pymc as pm
import pytest
from numpyro.infer import NUTS

import simuk

@pytest.fixture(scope="module")
def sbc_with_fits(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_with_fits_numpyro(numpyro_eight_schools_cauchy_prior, numpyro_eight_schools_cauchy_prior_data):
    sbc = simuk.SBC(
        NUTS(numpyro_eight_schools_cauchy_prior),
        data_dir=numpyro_eight_schools_cauchy_prior_data,
        num_simulations=10,
        sample_kwargs={"num_warmup": 10, "num_samples": 10},
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_no_fits(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        keep_fits=False,
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_posterior_with_fits(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=10,
        seed=42,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_posterior_no_fits(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
        seed=42,
        keep_fits=False,
    )
    sbc.run_simulations()
    return sbc


def test_ppr_requires_keep_fits(sbc_no_fits):
    with pytest.raises(ValueError, match="keep_fits"):
        simuk.plot_parameter_recovery(sbc_no_fits, if_show=False)


def test_ppr_requires_keep_fits_posterior(sbc_posterior_no_fits):
    with pytest.raises(ValueError, match="keep_fits"):
        simuk.plot_parameter_recovery(sbc_posterior_no_fits, if_show=False)


def test_ppr_requires_completed_simulations(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc, if_show=False)


def test_ppr_requires_completed_simulations_posterior(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc, if_show=False)


def test_ppr_basic(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_plot_ppr_basic_numpyro(sbc_with_fits_numpyro):
    fig = simuk.plot_parameter_recovery(sbc_with_fits_numpyro, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_basic_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_var_names_filter(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, var_names=["mu"], if_show=False)

    used_params = len(fig.data.coords["parameter"])
    assert used_params == 1


def test_ppr_var_names_filter_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, var_names=["mu"], if_show=False)
    used_params = len(fig.data.coords["parameter"])
    assert used_params == 1


def test_ppr_custom_ci_prob(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, ci_prob=0.5, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_custom_ci_prob_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, ci_prob=0.5, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_with_preexisting_plot_collection(sbc_with_fits):
    from simuk.plots import _build_recovery_dataset  # noqa: PLC0415

    ds = _build_recovery_dataset(
        sbc_with_fits,
        ci_prob=0.89,
        point_estimate="mean",
        transform=sbc_with_fits._transform,
        var_names=sbc_with_fits.kept_simulation_params.var_names,
    )

    pc = azp.plot_collection.PlotCollection.wrap(
        ds,
        cols=["parameter"],
        col_wrap=4,
        backend="matplotlib",
    )
    # mu(1) + tau(1) + theta(8) = 10 subplots needed
    returned_fig = simuk.plot_parameter_recovery(sbc_with_fits, plot_collection=pc, if_show=False)
    assert returned_fig is pc


def test_ppr_median_point_estimate(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, point_estimate="median", if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_median_point_estimate_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(
        sbc_posterior_with_fits, point_estimate="median", if_show=False
    )
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ppr_invalid_point_estimate(sbc_with_fits):
    with pytest.raises(ValueError, match="is not one of"):
        simuk.plot_parameter_recovery(sbc_with_fits, point_estimate="modddd", if_show=False)


def test_ppr_invalid_point_estimate_posterior(sbc_posterior_with_fits):
    with pytest.raises(ValueError, match="is not one of"):
        simuk.plot_parameter_recovery(
            sbc_posterior_with_fits, point_estimate="modddd", if_show=False
        )


def test_ppr_show_branch(monkeypatch, sbc_with_fits):
    called = {"value": False}

    def fake_show(*args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(azp.plot_collection.PlotCollection, "show", fake_show)

    fig = simuk.plot_parameter_recovery(sbc_with_fits, if_show=True)
    assert isinstance(fig, azp.plot_collection.PlotCollection)
    assert called["value"]


def test_plot_ecdf_basic(sbc_with_fits, sbc_no_fits):
    fig = simuk.plot_ecdf(sbc_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)

    fig = simuk.plot_ecdf(sbc_no_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_plot_ecdf_posterior(sbc_posterior_with_fits, sbc_posterior_no_fits):
    fig = simuk.plot_ecdf(sbc_posterior_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)

    fig = simuk.plot_ecdf(sbc_posterior_no_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_ecdf_show_branch(monkeypatch, sbc_with_fits):
    called = {"value": False}

    def fake_show(*args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(azp.plot_collection.PlotCollection, "show", fake_show)

    fig = simuk.plot_ecdf(sbc_with_fits, if_show=True)
    assert isinstance(fig, azp.plot_collection.PlotCollection)
    assert called["value"]
