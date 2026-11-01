"""The model with a matrix of covariates z [P, K] (b_p, b_nu as K-vectors), the
guides, and the standardization check."""
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from hsmodel import model as M
from hsmodel import nuts
from hsmodel.batching import ParentBatcher
from hsmodel.likelihood import group_by_count
from hsmodel.simulate import simulate_dataset, DEFAULT_CHROM_LEN

from conftest import draw_sites


@pytest.fixture(scope="module")
def sim60():
	return simulate_dataset(P=60, seed=4)


def test_simulator_single_covariate_draws(sim60):
	"""K = 1: z has shape [P, 1] and holds the same draws as the single-covariate simulator."""
	assert sim60["z"].shape == (60, 1)
	assert np.array_equal(sim60["z"][:, 0], np.random.default_rng(4).standard_normal(60))


def logdens(fn, args, params):
	"""Compiled numpyro log density of fn(*args) at params."""
	return float(jax.jit(lambda prm: log_density(fn, args, {}, prm)[0])(params))


def test_one_covariate_as_vector_or_matrix(sim60):
	"""A 1-D z and the same z as a [P, 1] matrix give identical log densities
	for the SVI model, the mean-field guide and the NUTS model."""
	d = sim60
	P = d["P"]
	bt = ParentBatcher(d["X"], d["NXO"], d["chrom"], d["par"], P, d["chrom_len"], batch=16)
	ix, data = next(bt.stream())
	Lc = bt.Lc
	z2, z1 = jnp.asarray(d["z"]), jnp.asarray(d["z"][:, 0])
	sites = draw_sites(M.model, data, Lc, ix, z2, P, seed=3)
	assert sites["b_p"].shape == (1,)
	lp2 = logdens(M.model, (data, Lc, ix, z2, P), sites)
	lp1 = logdens(M.model, (data, Lc, ix, z1, P), sites)
	assert np.isfinite(lp2) and abs(lp1 - lp2) < 1e-10
	gp = {f"{n}_loc": (jnp.array([0.3]) if n in M.COEFS else jnp.array(0.1)) for n in M.GLOBALS}
	gp.update({f"{n}_scale": (jnp.array([0.2]) if n in M.COEFS else jnp.array(0.2)) for n in M.GLOBALS})
	gl2 = logdens(handlers.substitute(M.guide, gp), (data, Lc, ix, z2, P), sites)
	gl1 = logdens(handlers.substitute(M.guide, gp), (data, Lc, ix, z1, P), sites)
	assert np.isfinite(gl2) and abs(gl1 - gl2) < 1e-10
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"], d["NXO"], d["chrom"], 22, d["par"], P))
	ns = draw_sites(nuts.model, cd, Lc, z2, seed=5)
	ln2 = logdens(nuts.model, (cd, Lc, z2), ns)
	ln1 = logdens(nuts.model, (cd, Lc, z1), ns)
	assert np.isfinite(ln2) and abs(ln1 - ln2) < 1e-8


def test_lowrank_marginals_match_guide_draws():
	"""K = 3: the analytic marginal SDs of the low-rank guide agree with 20,000 guide draws."""
	rng = np.random.default_rng(0)
	P, C, K, r = 12, 22, 3, 4
	G = 6 + 2 * K
	D = G + 2 * C
	params = {"glob_loc": jnp.asarray(rng.standard_normal(D)),
	          "glob_cov_factor": jnp.asarray(0.3 * rng.standard_normal((D, r))),
	          "glob_cov_diag": jnp.asarray(rng.uniform(0.01, 0.2, D)),
	          "local_loc": jnp.asarray(rng.standard_normal((P, 2))),
	          "local_scale_tril": jnp.asarray(np.tril(rng.uniform(0.2, 0.8, (P, 2, 2)))),
	          "local_coupling": jnp.asarray(0.3 * rng.standard_normal((P, 2, G)))}
	draws = Predictive(M.guide_lowrank, params=params, num_samples=20000)(
		jax.random.PRNGKey(1), None, jnp.asarray(DEFAULT_CHROM_LEN), jnp.arange(P), jnp.zeros((P, K)), P)
	m = M.lowrank_marginals({k: np.asarray(v) for k, v in params.items()}, K=K)
	assert draws["b_p"].shape == (20000, K)
	worst = max(np.max(np.abs(np.asarray(draws[n]).std(0) / np.atleast_1d(m[f"{n}_scale"]) - 1))
	            for n in M.GLOBALS + ["c_p", "c_nu", "e_p", "e_nu"])
	assert worst < 0.04


def test_check_standardized():
	rng = np.random.default_rng(0)
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		assert M.check_standardized(rng.standard_normal((2000, 2)))
		assert not w
		assert not M.check_standardized(np.c_[rng.standard_normal(500), rng.integers(0, 2, 500)], ["age", "sex"])
	assert len(w) == 1 and "sex" in str(w[0].message) and "age" not in str(w[0].message)


def test_global_entries_name_coefficients():
	entries = M.global_entries(M.globals_used(False), 2, ["age", "sex"])
	keys = [e[0] for e in entries]
	assert "b_p[age]" in keys and "b_nu[sex]" in keys and not any(k.startswith("log_t") for k in keys)


# ---------------------------------------------------------------------------
# Without individual-specific (parent) effects
# ---------------------------------------------------------------------------
def test_globals_used():
	assert M.globals_used() == M.GLOBALS
	assert M.globals_used(parent_effects=False) == ["a_p", "b_p", "log_t_p", "a_nu", "b_nu", "log_t_nu"]
	assert M.globals_used(False, False) == ["a_p", "b_p", "a_nu", "b_nu"]


def test_models_without_parent_effects(sim60):
	"""No e_p, e_nu, log_s_* sites anywhere; with the parent effects fixed at
	0 the full model's likelihood term equals the reduced model's."""
	d = sim60
	P = d["P"]
	bt = ParentBatcher(d["X"], d["NXO"], d["chrom"], d["par"], P, d["chrom_len"], batch=16)
	ix, data = next(bt.stream())
	Lc, z = bt.Lc, jnp.asarray(d["z"])
	sites = draw_sites(M.model, data, Lc, ix, z, P, False, False, seed=1)
	assert set(sites) == {"a_p", "b_p", "a_nu", "b_nu"}
	reduced = logdens(lambda *a: M.model(*a, chrom_effects=False, parent_effects=False), (data, Lc, ix, z, P), sites)
	full_sites = dict(sites, log_s_p=jnp.array(0.0), log_s_nu=jnp.array(0.0), e_p=jnp.zeros(16), e_nu=jnp.zeros(16))
	full = logdens(lambda *a: M.model(*a, chrom_effects=False), (data, Lc, ix, z, P), full_sites)
	prior_extra = float(sum(dist_logp for dist_logp in (
		jax.scipy.stats.norm.logpdf(0.0, *M.PRIORS["log_s_p"]), jax.scipy.stats.norm.logpdf(0.0, *M.PRIORS["log_s_nu"]),
		(P / 16) * 32 * jax.scipy.stats.norm.logpdf(0.0))))
	assert np.isfinite(reduced) and abs(full - reduced - prior_extra) < 1e-6
	# guides emit exactly the model's sites
	for g in (M.guide, M.guide_lowrank):
		tr = handlers.trace(handlers.seed(lambda *a: g(*a, chrom_effects=True, parent_effects=False), 0)).get_trace(
			data, Lc, ix, z, P)
		model_sites = {k for k, v in tr.items() if v["type"] == "sample" and not v["infer"].get("is_auxiliary")}
		assert model_sites == {"a_p", "b_p", "log_t_p", "a_nu", "b_nu", "log_t_nu", "c_p", "c_nu"}
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"], d["NXO"], d["chrom"], 22, d["par"], P))
	ns = draw_sites(nuts.model, cd, Lc, z, True, False, False, seed=2)
	assert "e_p" not in ns and "log_s_p" not in ns and "c_p_decentered" in ns
	assert np.isfinite(logdens(lambda *a: nuts.model(*a, parent_effects=False), (cd, Lc, z), ns))


def test_lowrank_marginals_without_parent_effects():
	rng = np.random.default_rng(1)
	K, C, r = 2, 5, 3
	G = 4 + 2 * K                                           # globals without log_s_p, log_s_nu
	D = G + 2 * C
	params = {"glob_loc": rng.standard_normal(D), "glob_cov_factor": 0.3 * rng.standard_normal((D, r)),
	          "glob_cov_diag": rng.uniform(0.01, 0.2, D)}
	m = M.lowrank_marginals(params, K=K, parent_effects=False)
	assert "e_p_loc" not in m and "log_s_p_loc" not in m and m["b_p_loc"].shape == (K,) and m["c_nu_loc"].shape == (C,)
