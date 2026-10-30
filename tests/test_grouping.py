"""The grouped likelihoods (by chromosome, by chromosome width, by crossover
count) must equal the per-record likelihood, values and gradients, including
for ragged data (unequal meioses per parent, missing chromosomes)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hsmodel.likelihood import (group_by_chromosome, batch_loglik, batch_loglik_per, grouped_loglik,
                                parent_grouped_loglik, group_by_chromosome_width, bucketed_loglik,
                                parent_bucketed_loglik, group_by_count, count_loglik_per_parent, count_loglik)


@pytest.fixture(scope="module")
def ref(ragged_data):
	"""Per-record reference: per-parent log-likelihoods and their gradients."""
	d = ragged_data
	X, NXO, LEN, par = map(jnp.asarray, (d["X"], d["NXO"], d["LEN"], d["par"]))
	nu, p = jnp.asarray(d["nu_i"]), jnp.asarray(d["p_i"])
	ll_rec = np.asarray(jax.jit(batch_loglik_per)(X, NXO, LEN, nu[par], p[par]))
	per_parent = np.bincount(d["par"], weights=ll_rec, minlength=d["P"])

	def tot_rec(nu, p):
		return jnp.sum(batch_loglik_per(X, NXO, LEN, nu[par], p[par]))

	grad = jax.jit(jax.grad(tot_rec, argnums=(0, 1)))(nu, p)
	shared = float(jnp.sum(jax.jit(batch_loglik)(X, NXO, LEN, 11.0, 0.06)))
	return dict(per_parent=per_parent, grad=grad, shared=shared, nu=nu, p=p)


def max_diff(a, b):
	return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b))


def test_per_parent_by_chromosome(ragged_data, ref):
	d = ragged_data
	Xg, Ng, Mg, Lc = map(jnp.asarray, group_by_chromosome(d["X"], d["NXO"], d["chrom"], d["chrom_len"],
	                                                      d["par"], d["P"]))
	got = np.asarray(jax.jit(parent_grouped_loglik)(Xg, Ng, Mg, Lc, ref["nu"], ref["p"]))
	assert np.max(np.abs(got - ref["per_parent"])) < 1e-9
	g = jax.jit(jax.grad(lambda nu, p: jnp.sum(parent_grouped_loglik(Xg, Ng, Mg, Lc, nu, p)),
	                     argnums=(0, 1)))(ref["nu"], ref["p"])
	assert all(bool(jnp.all(jnp.isfinite(a))) for a in g)
	assert max_diff(g, ref["grad"]) < 1e-8


@pytest.mark.parametrize("nu, p", [(2.0, 0.2), (11.0, 0.06), (40.0, 0.01)])
def test_shared_parameters_by_chromosome(ragged_data, nu, p):
	"""One (nu, p) for every record: G* once per chromosome."""
	d = ragged_data
	Xs, Ns, Ms, Lc = map(jnp.asarray, group_by_chromosome(d["X"], d["NXO"], d["chrom"], d["chrom_len"]))
	a = float(jax.jit(grouped_loglik)(Xs, Ns, Ms, Lc, nu, p))
	b = float(jnp.sum(jax.jit(batch_loglik)(jnp.asarray(d["X"]), jnp.asarray(d["NXO"]),
	                                        jnp.asarray(d["LEN"]), nu, p)))
	assert abs(a - b) < 1e-8


def test_chromosome_specific_width(ragged_data, ref):
	d = ragged_data
	buckets, chroms = group_by_chromosome_width(d["X"], d["NXO"], d["chrom"], d["chrom_len"], d["par"], d["P"])
	buckets = [tuple(map(jnp.asarray, b)) for b in buckets]
	assert sorted(np.concatenate(chroms)) == list(range(22)), "every chromosome in exactly one bucket"
	for b, cs in zip(buckets, chroms):                   # width = max crossovers on those chromosomes
		assert b[0].shape[-1] == max(1, d["NXO"][np.isin(d["chrom"], cs)].max())
	got = np.asarray(jax.jit(parent_bucketed_loglik)(buckets, ref["nu"], ref["p"]))
	assert np.max(np.abs(got - ref["per_parent"])) < 1e-9
	g = jax.jit(jax.grad(lambda nu, p: jnp.sum(parent_bucketed_loglik(buckets, nu, p)), argnums=(0, 1)))(
		ref["nu"], ref["p"])
	assert max_diff(g, ref["grad"]) < 1e-8
	sb, _ = group_by_chromosome_width(d["X"], d["NXO"], d["chrom"], d["chrom_len"])
	sb = [tuple(map(jnp.asarray, b)) for b in sb]
	assert abs(float(jax.jit(bucketed_loglik)(sb, 11.0, 0.06)) - ref["shared"]) < 1e-8


def test_grouped_by_count(ragged_data, ref):
	"""Records grouped by crossover count (no padding)."""
	d = ragged_data
	Lc = jnp.asarray(d["chrom_len"])
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"], d["NXO"], d["chrom"], 22, d["par"], d["P"]))
	assert int(cd["count0"].sum()) + sum(g[0].shape[0] for g in cd["groups"]) == len(d["NXO"]), \
		"every record used once"
	got = np.asarray(jax.jit(count_loglik_per_parent)(cd, Lc, ref["nu"], ref["p"]))
	assert np.max(np.abs(got - ref["per_parent"])) < 1e-9
	g = jax.jit(jax.grad(lambda nu, p: jnp.sum(count_loglik_per_parent(cd, Lc, nu, p)), argnums=(0, 1)))(
		ref["nu"], ref["p"])
	assert max_diff(g, ref["grad"]) < 1e-8
	sc = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"], d["NXO"], d["chrom"], 22))
	assert abs(float(jax.jit(count_loglik)(sc, Lc, 11.0, 0.06)) - ref["shared"]) < 1e-8


def test_per_parent_and_chromosome_parameters(ragged_data, ref):
	"""Separate (nu, p) for every parent and chromosome; [P, C] arrays with
	equal columns reproduce the [P] case."""
	d = ragged_data
	P, par, chrom = d["P"], d["par"], d["chrom"]
	rng = np.random.default_rng(12)
	Lc = jnp.asarray(d["chrom_len"])
	cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"], d["NXO"], chrom, 22, par, P))
	nu_pc = d["nu_i"][:, None] * np.exp(0.2 * rng.standard_normal((P, 22)))
	p_pc = np.clip(d["p_i"][:, None] * np.exp(0.3 * rng.standard_normal((P, 22))), 1e-4, 0.6)
	X, NXO, LEN = map(jnp.asarray, (d["X"], d["NXO"], d["LEN"]))
	ll_rec = np.asarray(jax.jit(batch_loglik_per)(X, NXO, LEN, jnp.asarray(nu_pc[par, chrom]),
	                                              jnp.asarray(p_pc[par, chrom])))
	expect = np.bincount(par, weights=ll_rec, minlength=P)
	got = np.asarray(jax.jit(count_loglik_per_parent)(cd, Lc, jnp.asarray(nu_pc), jnp.asarray(p_pc)))
	assert np.max(np.abs(got - expect)) < 1e-9
	same = np.asarray(jax.jit(count_loglik_per_parent)(cd, Lc, jnp.asarray(np.repeat(d["nu_i"][:, None], 22, 1)),
	                                                   jnp.asarray(np.repeat(d["p_i"][:, None], 22, 1))))
	assert np.max(np.abs(same - ref["per_parent"])) < 1e-9
	g = jax.jit(jax.grad(lambda a, b: jnp.sum(count_loglik_per_parent(cd, Lc, a, b)), argnums=(0, 1)))(
		jnp.asarray(nu_pc), jnp.asarray(p_pc))
	ip, ic = jnp.asarray(par), jnp.asarray(chrom)
	gr = jax.jit(jax.grad(lambda a, b: jnp.sum(batch_loglik_per(X, NXO, LEN, a[ip, ic], b[ip, ic])),
	                      argnums=(0, 1)))(jnp.asarray(nu_pc), jnp.asarray(p_pc))
	assert max_diff(g, gr) < 1e-8
