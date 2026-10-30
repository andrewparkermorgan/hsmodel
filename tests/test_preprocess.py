"""`hsmodel preprocess`: round trip from simulated data, units, stdin,
single-sex fitting compatibility, and input errors."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hsmodel.data import load_dataset
from hsmodel.likelihood import group_by_count, count_loglik_per_parent, batch_loglik_per
from hsmodel.preprocess import select_sex
from hsmodel.simulate import simulate_dataset

from conftest import run_cli

P = 40
MALE_SCALE = 1.25                                    # male map 25% longer than the female map
SEX = np.where(np.arange(P) < P // 2, 2, 1)          # first half female, second half male
SEX_CODES = {1: ["1", "m", "M"], 2: ["2", "f", "F"]}


@pytest.fixture(scope="module")
def sim():
	d = simulate_dataset(P=P, meioses=2, seed=5)
	d["names"] = [f"chr{c + 1}" for c in range(len(d["chrom_len"]))]
	return d


def write_inputs(d, folder, scale=1.0, fmt="{:.12g}", pars=range(P), tag=""):
	cross, mp = folder / f"xo{tag}.tsv", folder / f"map{tag}.tsv"
	names = d["names"]
	with open(mp, "w") as fh:
		fh.write("# sex\tchromosome\tlength\n")
		for c in range(len(names)):
			fh.write(f"F\t{names[c]}\t{fmt.format(d['chrom_len'][c] * scale)}\n")
			fh.write(f"M\t{names[c]}\t{fmt.format(d['chrom_len'][c] * MALE_SCALE * scale)}\n")
	with open(cross, "w") as fh:
		for r in range(len(d["NXO"])):
			i = d["par"][r]
			if i not in pars:
				continue
			pos = d["X"][r, :d["NXO"][r]] * scale
			code = SEX_CODES[SEX[i]][r % 3]               # every accepted spelling of sex is used
			fh.write(f"ind_{i:03d}\t{code}\t{names[d['chrom'][r]]}\t[{','.join(fmt.format(v) for v in pos)}]\n")
	return cross, mp


@pytest.fixture(scope="module")
def loaded(sim, tmp_path_factory):
	folder = tmp_path_factory.mktemp("pre")
	cross, mp = write_inputs(sim, folder)
	out = folder / "out.npz"
	res = run_cli(["preprocess", "-i", cross, "-m", mp, "-o", out])
	assert res.returncode == 0, res.stderr
	return dict(folder=folder, cross=cross, map=mp, res=res, e=load_dataset(out))


def test_round_trip(sim, loaded):
	"""Morgans, both sexes, text IDs, all sex spellings."""
	d, e = sim, loaded["e"]
	assert np.array_equal(e["NXO"], d["NXO"]) and np.allclose(e["X"], d["X"], atol=1e-12)
	assert np.array_equal(e["par"], d["par"]) and np.array_equal(e["chrom"], d["chrom"])
	assert np.array_equal(e["replicate"], d["replicate"])
	assert np.allclose(e["LEN"], d["LEN"] * np.where(SEX[d["par"]] == 1, MALE_SCALE, 1.0))
	assert list(e["chrom_names"]) == d["names"]
	assert list(e["parent_names"]) == [f"ind_{i:03d}" for i in range(P)]
	assert np.array_equal(e["sex"], SEX) and np.all(np.isnan(e["chrom_len"]))
	assert "z" not in e and "truth" not in e


def test_summary_printed(sim, loaded):
	out = loaded["res"].stdout
	assert out.startswith("Empirical crossover data")
	assert "(20 male, 20 female)" in out and f"{len(sim['NXO']):,}" in out
	assert "male map" in out and "female map" in out
	assert "chrom_len is NaN" in out


@pytest.mark.parametrize("args", [["-i", "-"], []], ids=["-i -", "no -i"])
def test_centimorgans_from_stdin(sim, loaded, args):
	folder = loaded["folder"]
	cross_cm, mp_cm = write_inputs(sim, folder, scale=100.0, tag="_cm")
	out = folder / "out_cm.npz"
	res = run_cli(["preprocess"] + args + ["-m", mp_cm, "-o", out, "--cM"], stdin=cross_cm.read_text())
	assert res.returncode == 0, res.stderr
	e2 = load_dataset(out)
	assert np.allclose(e2["X"], loaded["e"]["X"], atol=1e-12) and np.allclose(e2["LEN"], loaded["e"]["LEN"])


def check_fitting_likelihood(dd, C):
	"""The fitting code's likelihood (one length per chromosome) equals the per-record one."""
	k = dd["P"]
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(dd["X"], dd["NXO"], dd["chrom"], C, dd["par"], k))
	nu, p = jnp.full(k, 11.0), jnp.full(k, 0.06)
	a = np.asarray(jax.jit(count_loglik_per_parent)(cd, jnp.asarray(dd["chrom_len"]), nu, p))
	b = np.bincount(dd["par"], weights=np.asarray(jax.jit(batch_loglik_per)(
		jnp.asarray(dd["X"]), jnp.asarray(dd["NXO"]), jnp.asarray(dd["LEN"]), nu[dd["par"]], p[dd["par"]])),
		minlength=k)
	assert np.max(np.abs(a - b)) < 1e-9


def test_single_sex_file(sim, loaded):
	folder = loaded["folder"]
	cross_f, _ = write_inputs(sim, folder, pars=range(P // 2), tag="_f")
	out = folder / "out_f.npz"
	res = run_cli(["preprocess", "-i", cross_f, "-m", loaded["map"], "-o", out])
	assert res.returncode == 0, res.stderr
	ef = load_dataset(out)
	assert np.allclose(ef["chrom_len"], sim["chrom_len"])
	check_fitting_likelihood(ef, len(sim["chrom_len"]))


@pytest.mark.parametrize("sex", ["m", "f"])
def test_select_sex(sim, loaded, sex):
	dd = select_sex(loaded["e"], sex)
	assert dd["P"] == P // 2 and np.all(dd["sex"] == (1 if sex == "m" else 2))
	check_fitting_likelihood(dd, len(sim["chrom_len"]))


@pytest.mark.parametrize("content, message", [
	("ind1\tX\tchr1\t[0.5]\n", "sex must be"),
	("ind1\tF\tchr99\t[0.5]\n", "no female map length"),
	("ind1\tF\tchr1\t[{beyond}]\n", "outside the chromosome"),
	("ind1\tF\tchr1\t0.5,0.7\n", "bracketed list"),
	("ind1\tF\tchr1\t[0.5,abc]\n", "could not read"),
	("ind1\tF\tchr1\t[]\nind1\tM\tchr2\t[]\n", "is male here but female earlier"),
	("ind1\tF\tchr1\n", "expected 4 tab-separated fields"),
	("individual\tsex\tchromosome\tpositions\n", "sex must be"),
	("", "no meiotic products"),
], ids=["bad sex", "chromosome not in map", "position beyond chromosome", "malformed list",
        "non-numeric position", "inconsistent sex", "wrong field count", "header line", "empty"])
def test_input_errors(sim, loaded, content, message):
	content = content.replace("{beyond}", str(sim["chrom_len"][0] + 0.1))
	res = run_cli(["preprocess", "-m", loaded["map"], "-o", loaded["folder"] / "bad.npz"], stdin=content)
	assert res.returncode == 1 and message in res.stderr, res.stderr
	assert "line" in res.stderr or message == "no meiotic products"


@pytest.mark.parametrize("content, message", [
	("F\tchr1\n", "expected 3 tab-separated fields"),
	("F\tchr1\tabc\n", "is not a number"),
	("F\tchr1\t-1\n", "must be positive"),
	("F\tchr1\t1.0\nF\tchr1\t1.1\n", "second length"),
])
def test_map_errors(loaded, content, message):
	mp = loaded["folder"] / "badmap.tsv"
	mp.write_text(content)
	res = run_cli(["preprocess", "-m", mp, "-o", loaded["folder"] / "bad.npz"], stdin="ind1\tF\tchr1\t[]\n")
	assert res.returncode == 1 and message in res.stderr, res.stderr


def test_unsorted_and_coincident_positions(loaded):
	res = run_cli(["preprocess", "-m", loaded["map"], "-o", loaded["folder"] / "o3.npz"],
	              stdin="ind1\tF\tchr22\t[0.9, 0.2]\nind1\tF\tchr22\t[0.4,0.4]\n")
	assert res.returncode == 0, res.stderr
	e3 = load_dataset(loaded["folder"] / "o3.npz")
	assert np.allclose(e3["X"][0, :2], [0.2, 0.9]) and list(e3["replicate"]) == [0, 1]
	notes = [l for l in res.stdout.splitlines() if l.startswith("note")]
	assert any("sorted" in n for n in notes) and any("identical position" in n for n in notes)
