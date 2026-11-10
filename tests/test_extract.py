"""`hsmodel extract`: implied nu and p per individual (and per chromosome) from
draws of an SVI guide.  The fits are built by hand with the keys `infer`
writes, so no fitting is needed."""
import subprocess
import sys

import numpy as np
import pytest

from hsmodel.data import save_dataset
from hsmodel.extract import GuideDraws, individual_table
from hsmodel.simulate import simulate_dataset, DEFAULT_TRUTH
from hsmodel.summaries import GLOBALS

from conftest import run_cli

P, C, K = 12, 4, 2
LOC = {"a_p": -2.8, "log_s_p": -1.2, "log_t_p": -1.5, "a_nu": 2.3, "log_s_nu": -1.9, "log_t_nu": -1.6}
B_P, B_NU = np.array([0.3, -0.2]), np.array([-0.1, 0.25])
sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def meanfield(z, scale=0.1, e_scale=0.8, chrom_effects=True, parent_effects=True, store_z=True):
	"""A mean-field fit as `hsmodel infer` writes it; every global and chromosome
	effect has posterior SD `scale`, every individual effect `e_scale`."""
	rng = np.random.default_rng(0)
	f = {}
	for n in GLOBALS:
		if (n.startswith("log_t") and not chrom_effects) or (n.startswith("log_s") and not parent_effects):
			continue
		loc = {"b_p": B_P, "b_nu": B_NU}.get(n, LOC.get(n))
		f[f"{n}_loc"] = np.asarray(loc, dtype=float)
		f[f"{n}_scale"] = np.full(np.shape(loc), scale)
	if chrom_effects:
		for n in ("c_p", "c_nu"):
			f[f"{n}_loc"], f[f"{n}_scale"] = rng.normal(0, 0.3, C), np.full(C, scale)
	if parent_effects:
		for n in ("e_p", "e_nu"):
			f[f"{n}_loc"], f[f"{n}_scale"] = rng.normal(0, 1.0, P), np.full(P, e_scale)
	f.update(individual_ids=np.array([str(i) for i in range(P)]), covariate_names=np.array(["z1", "z2"]),
	         chrom_names=np.array([f"chr{c + 1}" for c in range(C)]), covariate_means=np.zeros(K),
	         covariate_sds=np.ones(K), standardized=False, losses=np.linspace(3000, 2500, 50))
	if store_z:
		f["covariate_values"] = z
	return f


def lowrank(z, rank=2):
	"""A low-rank fit (keys of guide_lowrank) with sizeable correlations and coupling."""
	rng = np.random.default_rng(1)
	G = 6 + 2 * K
	D = G + 2 * C
	mu = np.concatenate([[LOC["a_p"]], B_P, [LOC["log_s_p"], LOC["log_t_p"], LOC["a_nu"]], B_NU,
	                     [LOC["log_s_nu"], LOC["log_t_nu"]], rng.normal(0, 0.3, 2 * C)])
	f = dict(meanfield(z, parent_effects=True))
	f.update(glob_loc=mu, glob_cov_factor=rng.normal(0, 0.1, (D, rank)), glob_cov_diag=np.full(D, 0.01),
	         local_loc=rng.normal(0, 1, (P, 2)),
	         local_scale_tril=np.tile(np.array([[0.5, 0.0], [0.3, 0.4]]), (P, 1, 1)),
	         local_coupling=rng.normal(0, 0.5, (P, 2, G)))
	return f


@pytest.fixture(scope="module")
def z():
	return np.random.default_rng(5).standard_normal((P, K))


def as_array(rows):
	return np.array([v for _, _, v in rows])          # [rows, 8]: nu mean/median/lo/hi, p mean/median/lo/hi


def test_point_mass_gives_the_linear_predictors(z):
	"""With (almost) no posterior uncertainty every summary equals the
	back-transformed linear predictor, per individual and per chromosome."""
	f = meanfield(z, scale=1e-10, e_scale=1e-10)
	lp = LOC["a_p"] + z @ B_P + np.exp(LOC["log_s_p"]) * f["e_p_loc"]
	ln = LOC["a_nu"] + z @ B_NU + np.exp(LOC["log_s_nu"]) * f["e_nu_loc"]
	v = as_array(individual_table(f, z, 50, np.random.default_rng(0)))
	np.testing.assert_allclose(v[:, :4], np.repeat(1 + np.exp(ln)[:, None], 4, 1), rtol=1e-8)
	np.testing.assert_allclose(v[:, 4:], np.repeat(sig(lp)[:, None], 4, 1), rtol=1e-8)
	rows = individual_table(f, z, 50, np.random.default_rng(0), [str(c) for c in range(C)])
	assert [(i, c) for i, c, _ in rows] == [(i, c) for i in range(P) for c in range(C)]
	v = as_array(rows).reshape(P, C, 8)
	np.testing.assert_allclose(v[..., 1], 1 + np.exp(ln[:, None] + f["c_nu_loc"][None, :]), rtol=1e-8)
	np.testing.assert_allclose(v[..., 5], sig(lp[:, None] + f["c_p_loc"][None, :]), rtol=1e-8)


def test_one_uncertain_parameter(z):
	"""Only a_nu uncertain: nu - 1 is lognormal, so the median, mean and interval
	are known in closed form."""
	f = meanfield(z, scale=1e-10, e_scale=1e-10, parent_effects=False, chrom_effects=False)
	s = 0.3
	f["a_nu_scale"] = np.array(s)
	ln = LOC["a_nu"] + z @ B_NU
	v = as_array(individual_table(f, z, 200_000, np.random.default_rng(0)))
	np.testing.assert_allclose(v[:, 0], 1 + np.exp(ln + s * s / 2), rtol=3e-3)
	np.testing.assert_allclose(v[:, 1], 1 + np.exp(ln), rtol=3e-3)
	np.testing.assert_allclose(v[:, 2], 1 + np.exp(ln - 1.96 * s), rtol=5e-3)
	np.testing.assert_allclose(v[:, 3], 1 + np.exp(ln + 1.96 * s), rtol=5e-3)


def test_lowrank_draws_have_the_guide_covariance(z):
	"""Draws of the low-rank guide: globals with covariance diag(d) + W W^T, and
	each individual's effects with covariance L L^T + B Sigma_gg B^T."""
	f = lowrank(z)
	S = 200_000
	q = GuideDraws(f, K, S, np.random.default_rng(0))
	W, d = f["glob_cov_factor"], f["glob_cov_diag"]
	Sig = np.diag(d) + W @ W.T
	G = q.G
	assert G == 6 + 2 * K
	theta = np.column_stack([q.g["a_p"], q.g["b_p"], q.g["log_s_p"], q.g["log_t_p"], q.g["a_nu"], q.g["b_nu"],
	                         q.g["log_s_nu"], q.g["log_t_nu"], q.g["c_p"], q.g["c_nu"]])
	np.testing.assert_allclose(theta.mean(0), f["glob_loc"], atol=0.01)
	np.testing.assert_allclose(np.cov(theta.T), Sig, atol=0.003)
	e_p, e_nu = q.individual_effects(np.array([0, 5]))
	L, Bc = f["local_scale_tril"], f["local_coupling"]
	for j, i in enumerate((0, 5)):
		cov = L[i] @ L[i].T + Bc[i] @ Sig[:G, :G] @ Bc[i].T
		e = np.column_stack([e_p[:, j], e_nu[:, j]])
		np.testing.assert_allclose(e.mean(0), f["local_loc"][i], atol=0.01)
		np.testing.assert_allclose(np.cov(e.T), cov, atol=0.01)
		# coupling: Cov(e_i, globals) = B_i Sigma_gg
		cross = (e - e.mean(0)).T @ (theta[:, :G] - theta[:, :G].mean(0)) / S
		np.testing.assert_allclose(cross, Bc[i] @ Sig[:G, :G], atol=0.005)


@pytest.fixture(scope="module")
def files(tmp_path_factory, z):
	tmp = tmp_path_factory.mktemp("extract")
	truth = dict(DEFAULT_TRUTH, b_p=B_P, b_nu=B_NU)
	d = simulate_dataset(P=P, meioses=1, chrom_len=np.linspace(0.6, 2.0, C), truth=truth, seed=3)
	save_dataset(tmp / "sim.npz", d, truth, 3, True)
	zs = np.asarray(d["z"], dtype=float)
	np.savez(tmp / "fit_params.npz", **meanfield(zs))
	np.savez(tmp / "old_params.npz", **meanfield(zs, store_z=False))
	np.savez(tmp / "lowrank_params.npz", **lowrank(zs))
	np.savez(tmp / "plain_params.npz", **meanfield(zs, chrom_effects=False, parent_effects=False))
	return tmp


def read_tsv(text):
	lines = text.strip().split("\n")
	return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


@pytest.mark.parametrize("name", ["fit", "lowrank", "plain"])
def test_cli_tables(files, name):
	res = run_cli(["extract", files / f"{name}_params.npz"])
	assert res.returncode == 0, res.stderr
	head, rows = read_tsv(res.stdout)
	assert head == ["iid"] + [f"{q}_{s}" for q in ("nu", "p") for s in ("mean", "median", "lower95", "upper95")]
	assert [r[0] for r in rows] == [str(i) for i in range(P)]
	v = np.array([r[1:] for r in rows], dtype=float)
	for o in (0, 4):                                   # lower <= median <= upper, lower <= mean <= upper
		assert np.all((v[:, o + 2] <= v[:, o + 1]) & (v[:, o + 1] <= v[:, o + 3]))
		assert np.all((v[:, o + 2] <= v[:, o]) & (v[:, o] <= v[:, o + 3]))
	assert np.all(v[:, :4] > 1) and np.all((v[:, 4:] > 0) & (v[:, 4:] < 1))
	res = run_cli(["extract", files / f"{name}_params.npz", "--per-chromosome"])
	assert res.returncode == 0, res.stderr
	head, rows = read_tsv(res.stdout)
	assert head[:2] == ["iid", "chromosome"] and len(rows) == P * C
	assert [r[1] for r in rows[:C]] == [f"chr{c + 1}" for c in range(C)]
	if name == "plain":
		assert "no chromosome effects" in res.stderr


def test_old_fit_reads_covariates_from_the_data(files):
	"""A fit without stored covariate values needs --data, and then gives the same result."""
	res = run_cli(["extract", files / "old_params.npz"])
	assert res.returncode == 1 and "--data" in res.stderr
	new = run_cli(["extract", files / "fit_params.npz"])
	old = run_cli(["extract", files / "old_params.npz", "--data", files / "sim.npz"])
	assert old.returncode == 0, old.stderr
	assert old.stdout == new.stdout
	res = run_cli(["extract", files / "fit_params.npz", "--data", files / "sim.npz"])
	assert res.returncode == 0 and "ignored" in res.stderr


@pytest.mark.parametrize("args, code, message", [
	(["sim.npz"], 1, "not an SVI fit"),
	(["fit_params.npz", "--draws", "1"], 2, "--draws"),
	(["fit_params.npz", "--covariates", "x.txt"], 2, "--covariates needs --data"),
])
def test_errors(files, args, code, message):
	res = run_cli(["extract"] + [files / a if a.endswith(".npz") else a for a in args])
	assert res.returncode == code and message in res.stderr


def test_no_jax(files):
	code = ("import sys; from hsmodel.cli import main; rc = main(['extract', sys.argv[1]]); "
	        "assert rc == 0; assert 'jax' not in sys.modules, 'jax imported'")
	res = subprocess.run([sys.executable, "-c", code, str(files / "fit_params.npz")], capture_output=True, text=True)
	assert res.returncode == 0, res.stderr
