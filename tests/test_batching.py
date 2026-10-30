"""ParentBatcher must pack exactly the right records with fixed shapes."""
import numpy as np
import jax
import jax.numpy as jnp

from hsmodel.batching import ParentBatcher
from hsmodel.data import pad_records
from hsmodel.likelihood import batch_loglik_per, count_loglik_per_parent
from hsmodel.simulate import simulate_product


def test_batches_reproduce_per_parent_likelihood():
	rng = np.random.default_rng(5)
	P, B = 70, 16                                          # 70 = 4 batches of 16 + 6 left over
	chrom_len = np.linspace(0.6, 2.8, 22)
	nu_i, p_i = rng.uniform(2.0, 30.0, P), rng.uniform(0.005, 0.3, P)
	xo, lens, par, chrom = [], [], [], []
	for i in range(P):
		for _ in range(rng.integers(1, 5)):
			for c, L in enumerate(chrom_len):
				if rng.random() < 0.15:
					continue
				xo.append(simulate_product(nu_i[i], p_i[i], L, rng)); lens.append(L); par.append(i); chrom.append(c)
	par = np.array(par)
	X, NXO, LEN = pad_records(xo, lens)
	ll_rec = np.asarray(jax.jit(batch_loglik_per)(jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN),
	                                              jnp.asarray(nu_i[par]), jnp.asarray(p_i[par])))
	ref = np.bincount(par, weights=ll_rec, minlength=P)    # per-parent reference

	bt = ParentBatcher(X, NXO, chrom, par, P, chrom_len, batch=B, seed=1)
	summary = bt.padding_summary()
	assert all(cap >= real for real, cap in summary.values())
	f = jax.jit(lambda data, nu, p: count_loglik_per_parent(data, bt.Lc, nu, p))
	shapes, worst, left_out = None, 0.0, []
	for _ in range(5):
		seen = []
		for ix, data in bt.epoch():
			ix = np.asarray(ix)
			s = [a.shape for a in jax.tree_util.tree_leaves(data)]
			assert shapes is None or s == shapes, "batch shapes changed"
			shapes = s
			got = np.asarray(f(data, jnp.asarray(nu_i[ix]), jnp.asarray(p_i[ix])))
			worst = max(worst, np.max(np.abs(got - ref[ix])))
			seen.extend(ix.tolist())
		assert len(seen) == len(set(seen)) == (P // B) * B, "each parent at most once per epoch"
		left_out.append(frozenset(set(range(P)) - set(seen)))
	assert worst < 1e-9
	assert len(set(left_out)) > 1, "a different set of parents should sit out each epoch"
