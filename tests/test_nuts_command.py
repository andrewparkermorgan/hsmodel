"""`hsmodel nuts`: argument and input errors, and short runs on simulated and
preprocessed data (convergence is not tested; these runs are far too short)."""
import numpy as np
import pytest

from conftest import run_cli


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
	tmp = tmp_path_factory.mktemp("nuts")
	path = tmp / "sim.npz"
	r = run_cli(["simulate", "--parents", "20", "--meioses", "2", "--b-p", "0.4,-0.3", "--seed", "8", "--out", path])
	assert r.returncode == 0, r.stderr
	return tmp, path


@pytest.mark.parametrize("args, message", [
	(["--chains", "0"], "--chains must be at least 1"),
	(["--target-accept", "1.5"], "--target-accept must be between 0 and 1"),
	(["--parents", "1"], "--parents must be at least 2"),
	(["--warmup", "-1"], "--warmup must be non-negative"),
	(["--chain-method", "magic"], "invalid choice"),
])
def test_argument_errors(sim, args, message):
	tmp, path = sim
	r = run_cli(["nuts", "--data", path] + args)
	assert r.returncode == 2 and message in r.stderr, r.stderr


def test_input_errors(sim, tmp_path):
	tmp, path = sim
	r = run_cli(["nuts", "--data", path, "--coefficients", "z3"])
	assert r.returncode == 1 and "'z3' is neither" in r.stderr
	r = run_cli(["nuts", "--data", tmp_path / "missing.npz"])
	assert r.returncode == 1 and "error:" in r.stderr


def test_short_run_simulated(sim):
	"""First 6 individuals, no chromosome effects, one short chain."""
	tmp, path = sim
	out = tmp / "n1"
	r = run_cli(["nuts", "--data", path, "--parents", "6", "--no-chrom-effects", "--chains", "1", "--warmup", "15",
	             "--samples", "15", "--out", out])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "the first 6 of 20" in r.stderr and "start: prior medians" in r.stderr
	lines = [l.split("\t") for l in r.stdout.splitlines()]
	head = next(l for l in lines if l[0] == "parameter")
	assert head == ["parameter", "mean", "sd", "lower95", "upper95", "n_eff", "r_hat", "truth"]
	rows = {l[0]: l for l in lines if len(l) == len(head) and l[0] != "parameter"}
	assert set(rows) == {"a_p", "b_p[z1]", "b_p[z2]", "log_s_p", "a_nu", "b_nu[z1]", "b_nu[z2]", "log_s_nu"}
	assert all(np.isfinite(float(v)) for row in rows.values() for v in row[1:])
	assert not any(l[0] == "chromosome" for l in lines)
	s = np.load(f"{out}_samples.npz")
	assert s["a_p"].shape == (1, 15) and s["b_p"].shape == (1, 15, 2) and s["e_p"].shape == (1, 15, 6)
	assert list(s["covariate_names"]) == ["z1", "z2"] and len(s["individual_ids"]) == 6
	assert (tmp / "n1_summary.png").exists()


def test_short_run_without_parent_effects(sim):
	tmp, path = sim
	out = tmp / "n3"
	r = run_cli(["nuts", "--data", path, "--parents", "6", "--no-parent-effects", "--no-chrom-effects",
	             "--chains", "1", "--warmup", "15", "--samples", "15", "--out", out])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "individual-specific effects:      not included" in r.stderr
	rows = [l.split("\t")[0] for l in r.stdout.splitlines() if "\t" in l and not l.startswith("parameter")]
	assert rows == ["a_p", "b_p[z1]", "b_p[z2]", "a_nu", "b_nu[z1]", "b_nu[z2]"]
	s = np.load(f"{out}_samples.npz")
	assert "e_p" not in s.files and "log_s_p" not in s.files and s["b_nu"].shape == (1, 15, 2)


@pytest.mark.slow
def test_short_run_preprocessed_with_svi_init(tmp_path):
	"""Empirical-format data with sex-specific maps and a covariate file;
	chromosome effects; SVI starting point."""
	from hsmodel.simulate import simulate_dataset
	d = simulate_dataset(P=12, meioses=2, seed=2)
	C = len(d["chrom_len"])
	with open(tmp_path / "map.tsv", "w") as fh:
		for c in range(C):
			fh.write(f"F\tchr{c + 1}\t{d['chrom_len'][c]:.17g}\nM\tchr{c + 1}\t{1.3 * d['chrom_len'][c]:.17g}\n")
	with open(tmp_path / "xo.tsv", "w") as fh:
		for r_ in range(len(d["NXO"])):
			i = d["par"][r_]
			fh.write(f"id{i}\t{'F' if i % 2 else 'M'}\tchr{d['chrom'][r_] + 1}\t"
			         f"[{','.join(f'{v:.17g}' for v in d['X'][r_, :d['NXO'][r_]])}]\n")
	with open(tmp_path / "cov.txt", "w") as fh:
		fh.write("#iid\tage\n" + "".join(f"id{i}\t{30 + 5 * d['z'][i, 0]:.6g}\n" for i in range(12)))
	r = run_cli(["preprocess", "-i", tmp_path / "xo.tsv", "-m", tmp_path / "map.tsv", "-o", tmp_path / "e.npz"])
	assert r.returncode == 0, r.stderr
	out = tmp_path / "n2"
	r = run_cli(["nuts", "--data", tmp_path / "e.npz", "--covariates", tmp_path / "cov.txt", "--standardize",
	             "--svi-init", "--svi-steps", "40", "--chains", "1", "--warmup", "20", "--samples", "20", "--out", out])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "start: SVI fit (40 steps)" in r.stderr and "from " in r.stderr
	assert "truth" not in r.stdout
	chrom_rows = [l for l in r.stdout.splitlines() if l.startswith("chr") and not l.startswith("chromosome")]
	assert len(chrom_rows) == C
	s = np.load(f"{out}_samples.npz")
	assert s["c_p"].shape == (1, 20, C) and bool(s["standardized"]) and list(s["individual_ids"][:2]) == ["id0", "id1"]
	assert (tmp_path / "n2_svi_params.npz").exists() and (tmp_path / "n2_summary.png").exists()
