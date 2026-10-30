"""Chromosome lengths looked up by sex as well as chromosome."""
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

from hsmodel import nuts, svi
from hsmodel.batching import ParentBatcher
from hsmodel.data import pad_records, load_dataset
from hsmodel.likelihood import group_by_count, count_loglik_per_parent, batch_loglik_per, parent_chrom_lengths
from hsmodel.simulate import simulate_product

from conftest import run_cli, draw_sites


@pytest.fixture(scope="module")
def sexdata():
	rng = np.random.default_rng(21)
	P, C = 36, 23
	female = np.linspace(0.6, 2.8, C)
	male = female * rng.uniform(0.6, 1.0, C)             # a different shape, not just a scale
	male[-1] = np.nan                                    # last chromosome: female map only (an X-like case)
	by_sex = np.vstack([male, female])                   # row 0 male, row 1 female
	sex = np.where(np.arange(P) % 2 == 0, 1, 2)
	nu_i, p_i = rng.uniform(3.0, 25.0, P), rng.uniform(0.01, 0.25, P)
	xo, lens, par, chrom = [], [], [], []
	for i in range(P):
		for _ in range(rng.integers(1, 4)):
			for c in range(C):
				L = by_sex[sex[i] - 1, c]
				if np.isnan(L) or rng.random() < 0.1:
					continue
				xo.append(simulate_product(nu_i[i], p_i[i], L, rng)); lens.append(L); par.append(i); chrom.append(c)
	par, chrom = np.array(par), np.array(chrom)
	X, NXO, LEN = pad_records(xo, lens)
	Lpar = parent_chrom_lengths(by_sex, sex)
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(X, NXO, chrom, C, par, P, parent_lengths=Lpar))
	ref_fn = jax.jit(lambda a, b: jnp.bincount(jnp.asarray(par), weights=batch_loglik_per(
		jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN), a, b), length=P))
	return dict(rng=rng, P=P, C=C, female=female, male=male, by_sex=by_sex, sex=sex, nu_i=nu_i, p_i=p_i,
	            X=X, NXO=NXO, LEN=LEN, par=par, chrom=chrom, Lpar=Lpar, cd=cd, ref_fn=ref_fn,
	            Lc_dummy=jnp.asarray(np.nan_to_num(female)))          # ignored when data["L"] is present


def test_parent_lengths_match_records(sexdata):
	s = sexdata
	assert np.allclose(s["Lpar"][s["par"], s["chrom"]], s["LEN"])


@pytest.mark.parametrize("per_chrom", [False, True], ids=["per parent", "per parent and chromosome"])
def test_likelihood_with_parent_lengths(sexdata, per_chrom):
	s = sexdata
	P, C, par, chrom = s["P"], s["C"], s["par"], s["chrom"]
	if per_chrom:
		rng = np.random.default_rng(3)
		nu = jnp.asarray(s["nu_i"][:, None] * np.exp(0.2 * rng.standard_normal((P, C))))
		p = jnp.asarray(np.clip(s["p_i"][:, None] * np.exp(0.3 * rng.standard_normal((P, C))), 1e-3, 0.5))
		idx = lambda a: a[par, chrom]
	else:
		nu, p = jnp.asarray(s["nu_i"]), jnp.asarray(s["p_i"])
		idx = lambda a: a[par]
	got = np.asarray(jax.jit(count_loglik_per_parent)(s["cd"], s["Lc_dummy"], nu, p))
	ref = np.asarray(s["ref_fn"](idx(nu), idx(p)))
	g = jax.jit(jax.grad(lambda a, b: jnp.sum(count_loglik_per_parent(s["cd"], s["Lc_dummy"], a, b)),
	                     argnums=(0, 1)))(nu, p)
	gr = jax.jit(jax.grad(lambda a, b: jnp.sum(s["ref_fn"](idx(a), idx(b))), argnums=(0, 1)))(nu, p)
	assert np.all(np.isfinite(got))
	assert np.max(np.abs(got - ref)) < 1e-9
	assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(g, gr)) < 1e-8


def test_one_map_for_everyone_is_wrong(sexdata):
	"""The sex lookup matters: the female map for everyone gives a different likelihood."""
	s = sexdata
	nu, p = jnp.asarray(s["nu_i"]), jnp.asarray(s["p_i"])
	wrong = np.asarray(jax.jit(count_loglik_per_parent)(
		jax.tree_util.tree_map(jnp.asarray, group_by_count(s["X"], s["NXO"], s["chrom"], s["C"], s["par"], s["P"])),
		s["Lc_dummy"], nu, p))
	right = np.asarray(s["ref_fn"](nu[s["par"]], p[s["par"]]))
	assert abs(np.sum(wrong) - np.sum(right)) > 1.0


def test_batches_carry_parent_lengths(sexdata):
	s = sexdata
	bt = ParentBatcher(s["X"], s["NXO"], s["chrom"], s["par"], s["P"], np.full(s["C"], np.nan), batch=8,
	                   parent_lengths=s["Lpar"])
	right = np.asarray(s["ref_fn"](jnp.asarray(s["nu_i"])[s["par"]], jnp.asarray(s["p_i"])[s["par"]]))
	f = jax.jit(lambda data, a, b: count_loglik_per_parent(data, bt.Lc, a, b))
	worst = 0.0
	for ix, data in bt.epoch():
		ix = np.asarray(ix)
		got = np.asarray(f(data, jnp.asarray(s["nu_i"][ix]), jnp.asarray(s["p_i"][ix])))
		worst = max(worst, np.max(np.abs(got - right[ix])))
	assert worst < 1e-9


def test_nan_lengths_need_parent_lengths(sexdata):
	s = sexdata
	with pytest.raises(ValueError, match="parent_lengths"):
		ParentBatcher(s["X"], s["NXO"], s["chrom"], s["par"], s["P"], np.full(s["C"], np.nan), batch=8)


def test_models_finite_with_parent_lengths(sexdata):
	"""NUTS log density is finite; a short SVI fit (chromosome effects) has finite losses."""
	s = sexdata
	z = jnp.asarray(s["rng"].standard_normal(s["P"]))
	prm = draw_sites(nuts.model, s["cd"], s["Lc_dummy"], z)
	lp = float(jax.jit(lambda q: log_density(nuts.model, (s["cd"], s["Lc_dummy"], z), {}, q)[0])(prm))
	assert np.isfinite(lp)
	bt = ParentBatcher(s["X"], s["NXO"], s["chrom"], s["par"], s["P"], np.full(s["C"], np.nan), batch=8,
	                   parent_lengths=s["Lpar"])
	_, losses = svi.fit(bt, np.asarray(z), steps=60, lr=0.02, log_every=0)
	assert np.all(np.isfinite(losses))


def test_preprocess_output_gives_parent_lengths(sexdata, tmp_path):
	s = sexdata
	with open(tmp_path / "map.tsv", "w") as fh:
		for c in range(s["C"]):
			for sx, row in ((1, s["male"]), (2, s["female"])):
				if np.isfinite(row[c]):
					fh.write(f"{sx}\tchr{c + 1}\t{row[c]:.17g}\n")
	with open(tmp_path / "xo.tsv", "w") as fh:
		for r in range(len(s["NXO"])):
			pos = ",".join(f"{v:.17g}" for v in s["X"][r, :s["NXO"][r]])
			fh.write(f"ind{s['par'][r]}\t{s['sex'][s['par'][r]]}\tchr{s['chrom'][r] + 1}\t[{pos}]\n")
	res = run_cli(["preprocess", "-i", tmp_path / "xo.tsv", "-m", tmp_path / "map.tsv", "-o", tmp_path / "out.npz"])
	assert res.returncode == 0, res.stderr
	d = load_dataset(os.path.join(tmp_path, "out.npz"))
	Ld = parent_chrom_lengths(d["chrom_len_by_sex"], d["sex"])
	assert np.allclose(Ld[d["par"], d["chrom"]], d["LEN"])
	assert int(np.sum(np.isnan(d["chrom_len_by_sex"][0]))) == 1          # male map missing for one chromosome


def test_parent_lengths_shape_checked(sexdata):
	s = sexdata
	with pytest.raises(ValueError, match="shape"):
		group_by_count(s["X"], s["NXO"], s["chrom"], s["C"], s["par"], s["P"], parent_lengths=s["Lpar"][:, :-1])
