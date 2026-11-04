"""`hsmodel inspect`: recognizing datasets, SVI fits and NUTS fits; their
summaries; errors; and that datasets and SVI fits are read without JAX.
The fit files are built by hand with the keys `infer` and `nuts` write, so no
fitting is needed."""
import subprocess
import sys

import numpy as np
import pytest

from hsmodel.data import save_dataset
from hsmodel.errors import InputError
from hsmodel.inspect_command import classify, open_npz
from hsmodel.simulate import simulate_dataset, DEFAULT_TRUTH
from hsmodel.summaries import GLOBALS

from conftest import run_cli

P, C, K = 12, 4, 2
TRUTH = dict(DEFAULT_TRUTH, b_p=np.array([0.4, -0.3]), b_nu=np.array([0.0, 0.2]))
CHROMS = np.array(["chrA", "chrB", "chrC", "chrD"])
IDS = np.array([f"ind_{i}" for i in range(P)])
META = dict(individual_ids=IDS, covariate_names=np.array(["age", "parity"]), chrom_names=CHROMS,
            covariate_means=np.array([30.0, 1.5]), covariate_sds=np.array([5.0, 1.0]), standardized=True)
LOC = {"a_p": -2.8, "log_s_p": -1.2, "log_t_p": -1.5, "a_nu": 2.3, "log_s_nu": -1.9, "log_t_nu": -1.6}


def svi_params(chrom_effects=True, parent_effects=True, meta=True):
	"""A mean-field `hsmodel infer` output, as written by infer.run."""
	rng = np.random.default_rng(0)
	f = {}
	for n in GLOBALS:
		if (n.startswith("log_t") and not chrom_effects) or (n.startswith("log_s") and not parent_effects):
			continue
		if n in ("b_p", "b_nu"):
			f[f"{n}_loc"], f[f"{n}_scale"] = np.array([0.3, -0.2]), np.array([0.1, 0.12])
		else:
			f[f"{n}_loc"], f[f"{n}_scale"] = np.array(LOC[n]), np.array(0.1)
	if chrom_effects:
		for n in ("c_p", "c_nu"):
			f[f"{n}_loc"], f[f"{n}_scale"] = rng.normal(0, 0.2, C), np.full(C, 0.1)
	if parent_effects:
		for n in ("e_p", "e_nu"):
			f[f"{n}_loc"], f[f"{n}_scale"] = rng.normal(0, 0.5, P), np.full(P, 0.8)
	if meta:
		f.update(META, losses=np.linspace(3000, 2500, 400) + rng.normal(0, 5, 400), likelihood_exact=False,
		         likelihood_K=20)
	return f


def nuts_samples(chains=2, draws=50, divergent=0, bad=False):
	"""An `hsmodel nuts` output: draws [chains, draws, ...] plus metadata."""
	rng = np.random.default_rng(1)
	f = {}
	for n in GLOBALS:
		shape = (chains, draws, K) if n in ("b_p", "b_nu") else (chains, draws)
		f[n] = rng.normal(0, 0.1, shape) + (LOC.get(n, 0.0))
	if bad:                                  # chains centred in different places: large R-hat
		f["a_nu"][1] += 1.0
	for n, size in (("c_p", C), ("c_nu", C), ("e_p", P), ("e_nu", P)):
		f[n] = rng.normal(0, 0.3, (chains, draws, size))
	f.update(META, n_divergent=divergent, num_steps=np.full(chains * draws, 15),
	         accept_prob=np.full(chains * draws, 0.9))
	return f


@pytest.fixture(scope="module")
def files(tmp_path_factory):
	tmp = tmp_path_factory.mktemp("inspect")
	d = simulate_dataset(P=P, meioses=2, chrom_len=np.linspace(0.6, 2.0, C), truth=TRUTH, seed=3,
	                     chrom_names=list(CHROMS))
	save_dataset(tmp / "sim.npz", d, TRUTH, 3, True)
	# the same records as empirical data, via hsmodel preprocess
	with open(tmp / "map.tsv", "w") as fh:
		for c in range(C):
			fh.write(f"F\t{CHROMS[c]}\t{d['chrom_len'][c]:.12g}\nM\t{CHROMS[c]}\t{1.2 * d['chrom_len'][c]:.12g}\n")
	with open(tmp / "xo.tsv", "w") as fh:
		for r in range(len(d["NXO"])):
			i = d["par"][r]
			fh.write(f"{IDS[i]}\t{'F' if i % 2 else 'M'}\t{CHROMS[d['chrom'][r]]}\t"
			         f"[{','.join(f'{v:.12g}' for v in d['X'][r, :d['NXO'][r]])}]\n")
	res = run_cli(["preprocess", "-i", tmp / "xo.tsv", "-m", tmp / "map.tsv", "-o", tmp / "emp.npz"])
	assert res.returncode == 0, res.stderr
	np.savez(tmp / "svi_params.npz", **svi_params())
	np.savez(tmp / "svi_plain_params.npz", **svi_params(chrom_effects=False, parent_effects=False))
	np.savez(tmp / "nuts_svi_params.npz", **svi_params(meta=False))
	np.savez(tmp / "nuts_samples.npz", **nuts_samples())
	np.savez(tmp / "nuts_bad_samples.npz", **nuts_samples(divergent=3, bad=True))
	np.savez(tmp / "other.npz", a=np.arange(3))
	(tmp / "text.npz").write_text("not an archive\n")
	return tmp


@pytest.mark.parametrize("name, kind", [("sim", "dataset"), ("emp", "dataset"), ("svi_params", "svi"),
                                        ("svi_plain_params", "svi"), ("nuts_svi_params", "svi"),
                                        ("nuts_samples", "nuts")])
def test_classify(files, name, kind):
	assert open_npz(str(files / f"{name}.npz"))[0] == kind


def test_classify_rejects_other_npz():
	with pytest.raises(InputError, match="not a dataset or a model fit"):
		classify({"a": np.zeros(2)})


def test_simulated_dataset(files):
	res = run_cli(["inspect", files / "sim.npz"])
	assert res.returncode == 0, res.stderr
	out = res.stdout
	assert "simulated with `hsmodel simulate` (seed 3)" in out
	assert "Simulation settings (true values):" in out
	assert "b_p[z2]" in out and "-0.3" in out and "chromosome effects simulated:" in out
	assert "COs per meiotic product" in out
	for c in CHROMS:
		assert f"\n  {c} " in out                             # one row per chromosome


def test_empirical_dataset(files):
	res = run_cli(["inspect", files / "emp.npz"])
	assert res.returncode == 0, res.stderr
	out = res.stdout
	assert "read by `hsmodel preprocess`" in out
	assert "Simulation settings" not in out
	assert "male length (cM)" in out and "female length (cM)" in out
	assert "ind_0, ind_1" in out


def test_svi_fit(files):
	res = run_cli(["inspect", files / "svi_params.npz"])
	assert res.returncode == 0, res.stderr
	out = res.stdout
	assert "SVI fit from `hsmodel infer`" in out
	assert "mean-field" in out and "by quadrature" in out and "K = 20" in out
	assert "400 steps" in out
	assert "age had mean 30, SD 5" in out
	assert "b_p[age]" in out and "b_nu[parity]" in out
	# natural scale: the guide's median of p is sigmoid(a_p_loc)
	p_row = next(ln for ln in out.splitlines() if ln.strip().startswith("p "))
	assert float(p_row.split()[1]) == pytest.approx(1 / (1 + np.exp(2.8)), rel=1e-3)
	assert "Chromosome effects" in out and "chrD" in out
	assert "Individual effects" in out and "ind_" in out


def test_svi_fit_without_random_effects(files):
	res = run_cli(["inspect", files / "svi_plain_params.npz"])
	assert res.returncode == 0, res.stderr
	out = res.stdout
	assert "chromosome effects:                     not included" in out
	assert "individual effects:                     not included" in out
	assert "log_s_p" not in out and "log_t_nu" not in out
	assert "Chromosome effects (" not in out and "Individual effects (" not in out


def test_svi_init_file(files):
	res = run_cli(["inspect", files / "nuts_svi_params.npz"])
	assert res.returncode == 0, res.stderr
	assert "--svi-init" in res.stdout
	assert "b_p[z1]" in res.stdout                            # no stored names: z1, z2


def test_nuts_fit(files):
	res = run_cli(["inspect", files / "nuts_samples.npz"])
	assert res.returncode == 0, res.stderr
	out = res.stdout
	assert "NUTS fit from `hsmodel nuts`" in out
	assert "2 chain(s) x 50 draws" in out
	assert "n_eff" in out and "r_hat" in out
	assert "Chromosome effects" in out and "Individual effects" in out
	assert "R-hat <= 1.01" in out or "effective sample size" in out


def test_nuts_fit_problems(files):
	res = run_cli(["inspect", files / "nuts_bad_samples.npz"])
	assert res.returncode == 0, res.stderr
	assert "warning: 3 divergent transition(s)" in res.stdout
	assert "largest R-hat among the globals" in res.stdout and "(a_nu)" in res.stdout


@pytest.mark.parametrize("name, message", [("missing.npz", "no such file"), ("text.npz", "not an .npz archive"),
                                           ("other.npz", "not a dataset or a model fit")])
def test_errors(files, name, message):
	res = run_cli(["inspect", files / name])
	assert res.returncode == 1
	assert message in res.stderr


def test_no_file_is_usage_error():
	assert run_cli(["inspect"]).returncode == 2


@pytest.mark.parametrize("name", ["sim", "emp", "svi_params"])
def test_no_jax(files, name):
	"""Datasets and SVI fits are summarized without importing JAX."""
	code = ("import sys; from hsmodel.cli import main; rc = main(['inspect', sys.argv[1]]); "
	        "assert rc == 0; assert 'jax' not in sys.modules, 'jax imported'")
	res = subprocess.run([sys.executable, "-c", code, str(files / f"{name}.npz")], capture_output=True, text=True)
	assert res.returncode == 0, res.stderr
