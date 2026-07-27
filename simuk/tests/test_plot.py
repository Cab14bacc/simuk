import arviz_plots as azp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pymc as pm
import pytest
from numpyro.infer import NUTS

import simuk

# Test data (same as test_prior_sbc.py)
plot_data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma_eight_schools = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model() as centered_eight:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    y_obs = pm.Normal("y", mu=theta, sigma=sigma_eight_schools, observed=plot_data)


def eight_schools_cauchy_prior(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
    numpyro.sample("y", dist.Normal(theta, sigma), obs=y)


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
def sbc_with_fits_numpyro():
    sbc = simuk.SBC(
        NUTS(eight_schools_cauchy_prior),
        data_dir={"J": 8, "sigma": sigma_eight_schools, "y": plot_data},
        num_simulations=10,
        sample_kwargs={"num_warmup": 10, "num_samples": 10},
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
    sigma_pymc = pm.HalfNormal("sigma", sigma=2)
    y_data = pm.Data("y_data", obs_data)
    pm.Normal("y", mu=mu, sigma=sigma_pymc, observed=y_data)

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
