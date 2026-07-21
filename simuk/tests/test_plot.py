import arviz_plots as azp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest

import simuk

matplotlib.use("Agg")

# Test data (same as test_prior_sbc.py)
data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model() as centered_eight:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=data)


@pytest.fixture(scope="module")
def sbc_with_fits():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_no_fits():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        keep_fits=False,
        seed=42,
    )
    sbc.run_simulations()
    return sbc


# Test data (same as test_posterior_sbc.py)
default_rng = np.random.default_rng(1234)
obs_data = default_rng.normal(2.0, 1.0, size=20)
x_obs = np.linspace(0, 1, 20)
y_obs_reg = 1.5 * x_obs + default_rng.normal(0, 0.5, size=20)


with pm.Model() as simple_posterior_model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=2)
    y_data = pm.Data("y_data", obs_data)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

with simple_posterior_model:
    trace_simple = pm.sample(
        draws=30,
        tune=30,
        chains=1,
        random_seed=123,
        progressbar=False,
        compute_convergence_checks=False,
    )


@pytest.fixture(scope="module")
def sbc_posterior_with_fits():
    sbc = simuk.SBC(
        simple_posterior_model,
        trace=trace_simple,
        method="posterior",
        num_simulations=10,
        seed=42,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_posterior_no_fits():
    sbc = simuk.SBC(
        simple_posterior_model,
        trace=trace_simple,
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


def test_ppr_requires_completed_simulations():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc, if_show=False)


def test_ppr_requires_completed_simulations_posterior():
    sbc = simuk.SBC(
        simple_posterior_model,
        trace=trace_simple,
        method="posterior",
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc, if_show=False)


def test_ppr_basic(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, if_show=False)
    assert isinstance(fig, plt.Figure)


def test_ppr_basic_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, if_show=False)
    assert isinstance(fig, plt.Figure)


def test_ppr_var_names_filter(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, var_names=["mu"], if_show=False)
    visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert len(visible_axes) == 1


def test_ppr_var_names_filter_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, var_names=["mu"], if_show=False)
    visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert len(visible_axes) == 1


def test_ppr_with_transform(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(
        sbc_with_fits, transform=lambda name, val: np.mean(val), if_show=False
    )
    assert isinstance(fig, plt.Figure)
    # The mean transform reduces theta (8,) to scalar → 3 subplots total
    visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert len(visible_axes) == 3  # mu, tau, theta (all scalar after transform)

def test_ppr_with_bad_transform(sbc_with_fits):
    with pytest.raises(ValueError, match="`transform` should be a function or None"):
        simuk.plot_parameter_recovery(
            sbc_with_fits, transform="bad transform", if_show=False
        )


def test_ppr_with_transform_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(
        sbc_posterior_with_fits, transform=lambda name, val: np.mean(val), if_show=False
    )
    assert isinstance(fig, plt.Figure)
    visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert len(visible_axes) == 2  # mu, sigma (all scalar after transform)


def test_ppr_custom_ci_prob(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, ci_prob=0.5)
    plt.close(fig)
    assert isinstance(fig, plt.Figure)



def test_ppr_custom_ci_prob_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, ci_prob=0.5)
    plt.close(fig)
    assert isinstance(fig, plt.Figure)


def test_ppr_with_preexisting_axes(sbc_with_fits):
    # mu(1) + tau(1) + theta(8) = 10 subplots needed
    fig, axes = plt.subplots(2, 5)
    returned_fig = simuk.plot_parameter_recovery(sbc_with_fits, axes=axes, if_show=False)
    assert returned_fig is fig

def test_ppr_with_insufficient_preexisting_axes(sbc_with_fits):
    # mu(1) + tau(1) + theta(8) = 10 subplots needed
    with pytest.raises(ValueError, match="axes but only"):
        _, axes = plt.subplots(2, 4)
        simuk.plot_parameter_recovery(sbc_with_fits, axes=axes, if_show=False)


def test_ppr_with_preexisting_axes_posterior(sbc_posterior_with_fits):
    # mu(1) + sigma(1) + theta(8) = 10 subplots needed
    fig, axes = plt.subplots(2, 5)
    returned_fig = simuk.plot_parameter_recovery(sbc_posterior_with_fits, axes=axes, if_show=False)
    assert returned_fig is fig


def test_ppr_median_point_estimate(sbc_with_fits):
    fig = simuk.plot_parameter_recovery(sbc_with_fits, point_estimate="median", if_show=False)
    assert isinstance(fig, plt.Figure)


def test_ppr_median_point_estimate_posterior(sbc_posterior_with_fits):
    fig = simuk.plot_parameter_recovery(
        sbc_posterior_with_fits, point_estimate="median", if_show=False
    )
    assert isinstance(fig, plt.Figure)


def test_ppr_invalid_point_estimate(sbc_with_fits):
    with pytest.raises(ValueError, match="point_estimate"):
        simuk.plot_parameter_recovery(sbc_with_fits, point_estimate="mode", if_show=False)


def test_ppr_invalid_point_estimate_posterior(sbc_posterior_with_fits):
    with pytest.raises(ValueError, match="point_estimate"):
        simuk.plot_parameter_recovery(sbc_posterior_with_fits, point_estimate="mode", if_show=False)

def test_plot_ecdf_basic(sbc_with_fits, sbc_no_fits):
    fig = simuk.plot_ecdf(sbc_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)

    fig = simuk.plot_ecdf(sbc_no_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)


def test_plot_ecdf_posterior(sbc_posterior_with_fits, sbc_posterior_no_fits):
    fig = simuk.plot_ecdf(sbc_posterior_with_fits, if_show=False)
    assert isinstance(fig, azp.plot_collection.PlotCollection)

    plt.ion()
    fig = simuk.plot_ecdf(sbc_posterior_no_fits)
    assert isinstance(fig, azp.plot_collection.PlotCollection)
    plt.close("all")
