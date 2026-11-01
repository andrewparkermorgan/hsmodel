"""`hsmodel infer` end to end: simulated data -> text files -> `hsmodel
preprocess` -> `hsmodel infer` with a covariate file; input errors; options."""
import numpy as np
import pytest

from hsmodel.inputs import read_covariates, select_coefficients, match_individuals
from hsmodel.errors import InputError
from hsmodel.simulate import simulate_dataset, DEFAULT_TRUTH

from conftest import run_cli

P = 80
TRUTH = dict(DEFAULT_TRUTH, b_p=np.array([0.4, -0.3]), b_nu=np.array([0.0, 0.2]))
HEADER = "#iid\tage\tsexcov\tunused"


def write_cov(path, rows, header=HEADER):
	with open(path, "w") as fh:
		if header:
			fh.write(header + "\n")
		for r in rows:
			fh.write("\t".join(r) + "\n")
	return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
	tmp = tmp_path_factory.mktemp("infer")
	d = simulate_dataset(P=P, truth=TRUTH, seed=13)
	C = len(d["chrom_len"])
	iid = [f"ind_{i:03d}" for i in range(P)]
	sex = np.where(np.arange(P) % 2 == 0, 1, 2)
	# crossover and map files (both sexes, same map), then hsmodel preprocess
	with open(tmp / "map.tsv", "w") as fh:
		for c in range(C):
			for sx in (1, 2):
				fh.write(f"{sx}\tchr{c + 1}\t{d['chrom_len'][c]:.17g}\n")
	with open(tmp / "xo.tsv", "w") as fh:
		for r in range(len(d["NXO"])):
			i = d["par"][r]
			fh.write(f"{iid[i]}\t{sex[i]}\tchr{d['chrom'][r] + 1}\t"
			         f"[{','.join(f'{v:.17g}' for v in d['X'][r, :d['NXO'][r]])}]\n")
	data = tmp / "xo.npz"
	res = run_cli(["preprocess", "-i", tmp / "xo.tsv", "-m", tmp / "map.tsv", "-o", data])
	assert res.returncode == 0, res.stderr
	# covariate file: raw (unstandardized) covariates in shuffled order, plus an unused column
	rng = np.random.default_rng(1)
	age = 40 + 10 * d["z"][:, 0]                      # standardizing recovers z[:, 0] up to sampling
	sexcov = 5 + 2 * d["z"][:, 1]
	unused = rng.normal(size=P)
	order = rng.permutation(P)
	rows = [[iid[i], f"{age[i]:.10g}", f"{sexcov[i]:.10g}", f"{unused[i]:.10g}"] for i in order]
	return dict(tmp=tmp, data=data, rows=rows, cov=write_cov(tmp / "cov.txt", rows))


def run_infer(setup, args):
	return run_cli(["infer", "--data", setup["data"]] + args)


# ---------------------------------------------------------------------------
# Input errors (none of these may start a fit)
# ---------------------------------------------------------------------------
ERROR_CASES = {
	"individual missing from covariates": (lambda r: r[1:], HEADER, [], "1 in the data but not the covariates"),
	"extra individual in covariates": (lambda r: r + [["stranger", "1", "2", "3"]], HEADER, [],
	                                   "1 in the covariates but not the data: stranger"),
	"duplicate individual": (lambda r: r + [r[0]], HEADER, [], "listed more than once"),
	"header first column not iid": (lambda r: r, "#id\tage\tsexcov\tunused", [], "must be named 'iid'"),
	"non-numeric value": (lambda r: [r[0][:1] + ["abc"] + r[0][2:]] + r[1:], HEADER, [], "'abc' is not a number"),
	"missing value in a selected covariate": (lambda r: [r[0][:1] + ["NA"] + r[0][2:]] + r[1:], HEADER,
	                                          ["--coefficients", "age"], "missing covariate value"),
	"wrong number of columns": (lambda r: [r[0][:3]] + r[1:], HEADER, [], "expected 4 columns"),
	"unknown coefficient name": (lambda r: r, HEADER, ["--coefficients", "height"], "'height' is neither"),
	"coefficient index out of range": (lambda r: r, HEADER, ["--coefficients", "4"], "'4' is neither"),
	"coefficient selected twice": (lambda r: r, HEADER, ["--coefficients", "age,1"], "selected twice"),
}


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_input_errors(setup, case):
	edit, header, extra, message = ERROR_CASES[case]
	path = write_cov(setup["tmp"] / "bad.txt", edit(setup["rows"]), header)
	r = run_infer(setup, ["--covariates", path, "--steps", "5", "--out", setup["tmp"] / "x"] + extra)
	assert r.returncode == 1, r.stderr[-500:]
	assert message in r.stderr, r.stderr
	assert "individuals with data" not in r.stderr          # stopped before the fit


@pytest.mark.parametrize("args, message", [
	(["--steps", "0"], "--steps must be positive"),
	(["--lr", "0"], "--lr must be positive"),
	(["--rank", "0"], "--rank must be at least 1"),
	(["--guide-type", "full"], "invalid choice"),
])
def test_argument_errors(setup, args, message):
	r = run_infer(setup, ["--covariates", setup["cov"]] + args)
	assert r.returncode == 2 and message in r.stderr, r.stderr


def test_covariate_helpers(tmp_path):
	f = tmp_path / "c.txt"
	f.write_text("#iid a b\nx 1 2\ny 3 NA\n\nz 5 6\n")
	iids, names, vals = read_covariates(str(f))
	assert list(iids) == ["x", "y", "z"] and names == ["a", "b"] and np.isnan(vals[1, 1])
	assert select_coefficients(None, names) == [0, 1]
	assert select_coefficients("b,1", names) == [1, 0]
	assert list(match_individuals(["z", "x", "y"], list(iids))) == [2, 0, 1]
	with pytest.raises(InputError):
		match_individuals(["x", "y"], list(iids))
	f.write_text("x 1 2\ny 3 4\n")
	assert read_covariates(str(f))[1] == ["z1", "z2"]


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_fit_recovers_coefficients(setup):
	"""Covariates by name and index, standardized, default number of steps
	(400 x 80 / 32 = 1,000); a missing value in an unselected column is fine."""
	rows_na = [setup["rows"][0][:3] + ["NA"]] + setup["rows"][1:]
	cov_na = write_cov(setup["tmp"] / "cov_na.txt", rows_na)
	out = setup["tmp"] / "fit"
	r = run_infer(setup, ["--covariates", cov_na, "--coefficients", "age,2", "--standardize", "--out", out,
	                      "--seed", "3"])
	assert r.returncode == 0, r.stderr[-2000:]
	for line in ("individuals with data:            80", "meiotic products:                 5,280",
	             "model coefficients to estimate:   4", "chromosome-specific effects:      included",
	             "1,000 steps of 32 individuals"):
		assert line in r.stderr, r.stderr
	table = {}
	for line in r.stdout.splitlines():
		f = line.split("\t")
		if len(f) == 5 and f[0] != "parameter":
			table[f[0]] = [float(v) for v in f[1:]]
	for site in ("b_p", "b_nu"):
		for j, nm in enumerate(["age", "sexcov"]):
			m, sd, lo, hi = table[f"{site}[{nm}]"]
			assert lo <= TRUTH[site][j] <= hi, (site, nm, m, lo, hi)
	p = np.load(f"{out}_params.npz")
	assert list(p["covariate_names"]) == ["age", "sexcov"] and bool(p["standardized"])
	assert len(p["individual_ids"]) == P and len(p["losses"]) == 1000
	assert (setup["tmp"] / "fit_summary.png").exists()
	assert sum(1 for l in r.stdout.splitlines() if l.split("\t")[0][3:].isdigit() and l.startswith("chr")) == 22


def test_options_lowrank_no_chrom_effects_no_header(setup):
	"""Low-rank guide, no chromosome effects, headerless file (default names), explicit steps."""
	cov_nohead = write_cov(setup["tmp"] / "cov_nohead.txt", setup["rows"], header=None)
	r = run_infer(setup, ["--covariates", cov_nohead, "--coefficients", "1", "--guide-type", "lowrank",
	                      "--rank", "2", "--no-chrom-effects", "--steps", "30", "--batch", "16",
	                      "--out", setup["tmp"] / "fit2"])
	assert r.returncode == 0, r.stderr[-2000:]
	params = [l.split("\t")[0] for l in r.stdout.splitlines() if "\t" in l and not l.startswith("parameter")]
	assert "b_p[z1]" in params and not any(x.startswith("log_t") for x in params)
	assert "do not look standardized" in r.stderr
	assert "not included" in r.stderr and "lowrank (rank 2)" in r.stderr


# ---------------------------------------------------------------------------
# Simulated data: covariates taken from the data file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
	tmp = tmp_path_factory.mktemp("sim")
	path = tmp / "sim.npz"
	r = run_cli(["simulate", "--parents", "40", "--b-p", "0.4,-0.3", "--b-nu", "0,0.2", "--seed", "3", "--out", path])
	assert r.returncode == 0, r.stderr
	return tmp, path


def test_simulated_covariates_used_without_file(simulated):
	tmp, path = simulated
	r = run_cli(["infer", "--data", path, "--coefficients", "z2", "--steps", "30", "--out", tmp / "s1"])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "simulated values from the data file" in r.stderr and "1 covariate(s): z2" in r.stderr
	lines = [l.split("\t") for l in r.stdout.splitlines()]
	head = next(l for l in lines if l[0] == "parameter")
	assert head[-1] == "truth"
	rows = {l[0]: l for l in lines if len(l) == len(head) and l[0] != "parameter"}
	assert float(rows["b_p[z2]"][-1]) == pytest.approx(-0.3) and float(rows["b_nu[z2]"][-1]) == pytest.approx(0.2)
	assert "b_p[z1]" not in rows
	chrom_head = next(l for l in lines if l[0] == "chromosome")
	assert chrom_head[-1] == "true_effect_on_log_nu_minus_1"
	p = np.load(tmp / "s1_params.npz")
	assert list(p["individual_ids"][:3]) == ["0", "1", "2"]


def test_simulated_covariates_override_file(simulated, setup):
	tmp, path = simulated
	r = run_cli(["infer", "--data", path, "--covariates", setup["cov"], "--steps", "5", "--no-chrom-effects",
	             "--standardize", "--out", tmp / "s2"])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "is ignored" in r.stderr and "2 covariate(s): z1, z2" in r.stderr
	assert "truth" not in r.stdout.splitlines()[2]          # re-standardized: different scale, no truth


def test_covariate_file_required_for_empirical_data(setup):
	r = run_infer(setup, ["--steps", "5"])
	assert r.returncode == 1 and "give a covariate file with --covariates" in r.stderr


# ---------------------------------------------------------------------------
# Without individual-specific effects
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_no_parent_effects_recovers_truth(tmp_path):
	"""Data simulated with no between-individual variation (s_p = s_nu = 0),
	fitted without parent effects: no log_s_* or e_* parameters, and the
	intervals of the remaining globals cover the truth."""
	path = tmp_path / "sim0.npz"
	r = run_cli(["simulate", "--parents", "60", "--s-p", "0", "--s-nu", "0", "--b-p", "0.4", "--b-nu", "0.2",
	             "--seed", "5", "--out", path])
	assert r.returncode == 0, r.stderr
	out = tmp_path / "f0"
	r = run_cli(["infer", "--data", path, "--no-parent-effects", "--steps", "1500", "--out", out])
	assert r.returncode == 0, r.stderr[-2000:]
	assert "individual-specific effects:      not included" in r.stderr
	lines = [l.split("\t") for l in r.stdout.splitlines()]
	head = next(l for l in lines if l[0] == "parameter")
	rows = {l[0]: [float(v) for v in l[1:]] for l in lines if len(l) == len(head) and l[0] != "parameter"}
	assert set(rows) == {"a_p", "b_p[z1]", "log_t_p", "a_nu", "b_nu[z1]", "log_t_nu"}
	for key in ("a_p", "b_p[z1]", "a_nu", "b_nu[z1]", "log_t_nu"):
		m, sd, lo, hi, truth = rows[key]
		assert lo <= truth <= hi, (key, rows[key])
	p = np.load(f"{out}_params.npz")
	assert not any(k.startswith(("e_p", "e_nu", "log_s")) for k in p.files)


def test_no_parent_effects_lowrank(setup):
	r = run_infer(setup, ["--covariates", setup["cov"], "--standardize", "--no-parent-effects", "--guide-type",
	                      "lowrank", "--rank", "2", "--steps", "20", "--out", setup["tmp"] / "f3"])
	assert r.returncode == 0, r.stderr[-2000:]
	p = np.load(setup["tmp"] / "f3_params.npz")
	assert "local_loc" not in p.files and "e_p_loc" not in p.files and "glob_loc" in p.files
	assert "log_s_p" not in r.stdout and "log_t_p" in r.stdout
