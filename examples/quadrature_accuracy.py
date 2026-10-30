"""
Accuracy and cost of the fast likelihood as a function of the number of
Gauss-Legendre nodes (N_QUAD) and mixture terms (K), to choose cheaper settings
for NUTS, where every leapfrog step needs a full-data gradient.

Error: per-record log-likelihood minus the closed-form reference (incomplete
gamma F*, G* with K = 200), over nu in {2, 8, 16, 64}, p in {0.01, 0.1},
L in {0.6, 1.5, 2.8} Morgans, 100 simulated products each.
Cost: log-likelihood + gradient for the first 50 parents of the demo dataset.
"""
import itertools, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from hsmodel import likelihood
from hsmodel.data import pad_records
from hsmodel.likelihood import record_loglik, group_by_count, count_loglik_per_parent
from hsmodel.simulate import simulate_product as simulate, simulate_dataset

NODES = [8, 12, 16, 24, 32, 64]
KS = [8, 12, 20]
NUS, PS, LS = [2, 8, 16, 64], [0.01, 0.1], [0.6, 1.5, 2.8]

rng = np.random.default_rng(3)
cells = []
for nu, p, L in itertools.product(NUS, PS, LS):
	recs = [simulate(nu, p, L, rng) for _ in range(100)]
	cells.append((nu, p, L, recs))
width = max(1, max(len(r) for c in cells for r in c[3]))
padded = []
for nu, p, L, recs in cells:
	X, NXO, LEN = pad_records(recs, [L] * len(recs), max_n=width)
	padded.append((nu, p, jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN)))

ref_fn = jax.jit(jax.vmap(partial(record_loglik, K=200, exact=True), in_axes=(0, 0, 0, None, None)))
refs = [np.asarray(ref_fn(X, N, L, float(nu), p)) for nu, p, X, N, L in padded]

d = simulate_dataset()
k = 50
sel = d["par"] < k
cd = jax.tree_util.tree_map(jnp.asarray, group_by_count(d["X"][sel], d["NXO"][sel], d["chrom"][sel], 22, d["par"][sel], k))
Lc = jnp.asarray(d["chrom_len"])
nu_b, p_b = jnp.asarray(d["nu_i"][:k]), jnp.asarray(d["p_i"][:k])

print(f"{'nodes':>5} {'K':>3} {'max |error|':>12} {'worst cell (nu, p, L)':>24} {'grad ms (50 parents)':>21}")
rows = []
for nodes, K in itertools.product(NODES, KS):
	likelihood.set_quadrature(nodes, K)                 # fresh jit objects below pick up the new globals
	fast = jax.jit(jax.vmap(record_loglik, in_axes=(0, 0, 0, None, None)))
	worst, where = 0.0, None
	for (nu, p, X, N, L), ref in zip(padded, refs):
		e = np.max(np.abs(np.asarray(fast(X, N, L, float(nu), p)) - ref))
		if e > worst:
			worst, where = e, (nu, p, float(L[0]))
	g = jax.jit(jax.value_and_grad(lambda a, b: jnp.sum(count_loglik_per_parent(cd, Lc, a, b)), argnums=(0, 1)))
	jax.block_until_ready(g(nu_b, p_b)); t = time.time()
	for _ in range(5):
		jax.block_until_ready(g(nu_b, p_b))
	ms = (time.time() - t) / 5 * 1e3
	rows.append((nodes, K, worst, where, ms))
	print(f"{nodes:5d} {K:3d} {worst:12.2e} {str(where):>24} {ms:21.0f}", flush=True)
likelihood.set_quadrature(64, 20)
np.savetxt("quadrature_accuracy.csv", np.array([(r[0], r[1], r[2], r[4]) for r in rows]),
           delimiter=",", header="nodes,K,max_abs_error,grad_ms_50_parents", comments="", fmt="%g")
