"""`hsmodel simulate`: data, file round trip, summary, chromosome-length files,
covariate coefficients and argument errors."""
import numpy as np
import pytest

from hsmodel.data import load_dataset, stem_and_leaf, summarize_dataset
from hsmodel.simulate import simulate_dataset, parse_chrom_lengths, DEFAULT_TRUTH

from conftest import run_cli


def test_dataset_structure():
	d = simulate_dataset(P=7, meioses=2, seed=2)
	C = len(d["chrom_len"])
	assert len(d["NXO"]) == 7 * 2 * C                     # one record per meiotic product
	assert np.bincount(d["par"] * C + d["chrom"]).tolist() == [2] * (7 * C)
	assert np.allclose(d["LEN"], d["chrom_len"][d["chrom"]])
	assert all(np.all(np.diff(d["X"][r, :n]) >= 0) and np.all(d["X"][r, :n] <= d["LEN"][r])
	           for r, n in enumerate(d["NXO"]))
	assert not np.allclose(d["c_p"], 0)


def test_no_chrom_effects_keeps_crossover_stream():
	"""Chromosome effects come from their own random stream: switching them off
	sets them to 0 and leaves the parent draws unchanged."""
	a = simulate_dataset(P=5, seed=9)
	b = simulate_dataset(P=5, seed=9, chrom_effects=False)
	assert np.array_equal(a["z"], b["z"]) and np.array_equal(a["p_i"], b["p_i"])
	assert np.all(b["c_p"] == 0) and np.all(b["c_nu"] == 0)


def test_cli_matches_library_and_round_trips(tmp_path):
	out = tmp_path / "s.npz"
	res = run_cli(["simulate", "--parents", "12", "--meioses", "2", "--seed", "4", "--out", out])
	assert res.returncode == 0, res.stderr
	s = load_dataset(out)
	d = simulate_dataset(P=12, meioses=2, seed=4)
	for k in ("X", "NXO", "LEN", "par", "replicate", "chrom", "z", "c_p", "c_nu", "p_ic", "nu_ic"):
		assert np.array_equal(s[k], d[k]), k
	assert s["P"] == 12 and s["meioses"] == 2 and s["seed"] == 4 and s["chrom_effects"] is True
	assert s["truth"]["a_p"] == pytest.approx(DEFAULT_TRUTH["a_p"])
	# summary on stdout: counts per meiotic product (one chromosome in one individual)
	n_rec = len(d["NXO"])
	assert f"{n_rec:,}" in res.stdout and f"{int(d['NXO'].sum()):,}" in res.stdout
	assert f"{d['NXO'].mean():.3f}" in res.stdout
	assert "COs per meiotic product" in res.stdout and "Distance between consecutive COs" in res.stdout
	assert res.stdout == summarize_dataset(d) + f"\n\nwritten to {out}\n"


def test_chrom_lengths_file(tmp_path):
	f = tmp_path / "chroms.txt"
	f.write_text('# name, length in Morgans\n(chr1, 1.20)\nchr2, 0.80\n\nchr3 1.50\nchrX\t1.80\n("chr4", 0.95)\n')
	names, lens = parse_chrom_lengths(str(f))
	assert names == ["chr1", "chr2", "chr3", "chrX", "chr4"]
	assert np.allclose(lens, [1.2, 0.8, 1.5, 1.8, 0.95])
	out = tmp_path / "s.npz"
	res = run_cli(["simulate", "--parents", "3", "--chrom-lengths", f, "--out", out])
	assert res.returncode == 0, res.stderr
	s = load_dataset(out)
	assert list(s["chrom_names"]) == names and np.allclose(s["chrom_len"], lens)


def test_chrom_lengths_comma_list():
	names, lens = parse_chrom_lengths("0.5,1.5")
	assert names == ["1", "2"] and np.allclose(lens, [0.5, 1.5])


@pytest.mark.parametrize("content, message", [
	("chr1 1.2\nchr2\n", "expected (name, length)"),
	("chr1 abc\n", "not a number"),
	("chr1 -0.5\n", "must be positive"),
	("# only a comment\n", "no chromosomes found"),
])
def test_bad_chrom_lengths_file(tmp_path, content, message):
	f = tmp_path / "bad.txt"
	f.write_text(content)
	res = run_cli(["simulate", "--parents", "3", "--chrom-lengths", f, "--out", tmp_path / "s.npz"])
	assert res.returncode == 2 and message in res.stderr, res.stderr


def test_covariate_coefficients(tmp_path):
	out = tmp_path / "s.npz"
	res = run_cli(["simulate", "--parents", "10", "--b-p", "0.4,-0.3,0", "--b-nu", "0.1", "--out", out])
	assert res.returncode == 0, res.stderr
	s = load_dataset(out)
	assert s["z"].shape == (10, 3)
	assert np.allclose(s["truth"]["b_p"], [0.4, -0.3, 0]) and np.allclose(s["truth"]["b_nu"], [0.1, 0.1, 0.1])
	assert "parent covariates:" in res.stdout


@pytest.mark.parametrize("args, message", [
	(["--b-p", "0.4,0.1", "--b-nu", "1,2,3"], "b_p has 2 coefficients but b_nu has 3"),
	(["--parents", "0"], "must be at least 1"),
	(["--s-p", "-1"], "--s-p must be non-negative"),
])
def test_argument_errors(tmp_path, args, message):
	res = run_cli(["simulate", "--parents", "5", "--out", tmp_path / "s.npz"] + args)
	assert res.returncode == 2 and message in res.stderr, res.stderr


def test_stem_and_leaf():
	txt = stem_and_leaf([0, 1, 1, 2, 2, 2, 3, 12], "CO")
	assert "leaf unit = 1 CO" in txt
	rows = [l for l in txt.splitlines()[1:] if "|" in l]
	assert rows[0].split("|")[1].split()[0] == "011"
	assert sum(int(l.rsplit("[", 1)[1].rstrip("]")) for l in rows) == 8
	many = stem_and_leaf(np.random.default_rng(0).poisson(3, 5000), "CO")
	assert "each leaf represents about" in many
	assert stem_and_leaf([]) == "  (no values)"
