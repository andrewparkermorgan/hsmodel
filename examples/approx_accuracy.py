"""
Accuracy of the likelihood approximations in hsmodel.likelihood.

Two approximations are studied:
  1. Gauss-Legendre quadrature (N_QUAD nodes) for 1-F* and 1-G*, versus the
     closed form in terms of the regularized incomplete gamma function.
  2. Truncation of the thinning mixture  f* = sum_k 2^-k Gamma(k nu, 2nu(1-p))
     after K terms (used by both versions).

For each (nu, p, L) we simulate 100 meiotic products (one chromosome each) and
score the SAME records with every K, so differences across K reflect only K.
Columns per record:
  ll_fast   quadrature F*, G*, K terms         (what the fitting code uses)
  ll_exact  incomplete-gamma F*, G*, K terms
  diff      ll_fast - ll_exact
  ll_ref    incomplete-gamma F*, G*, K = 200   (truncation error < 2^-200)
  err_fast  ll_fast - ll_ref ;  err_exact  ll_exact - ll_ref
"""
import itertools, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from hsmodel.data import pad_records
from hsmodel.likelihood import record_loglik, N_QUAD
from hsmodel.simulate import simulate_product as simulate

NUS = [2, 8, 16, 64]
PS = [0.001, 0.01, 0.1, 0.2]
LS = [0.5, 1.0, 2.0]
KS = [20, 60, 100]
K_REF = 200
N_PRODUCTS = 100
SEED = 2026

_scorers = {}
def scorer(K, exact):
	"""Compiled per-record log-likelihood for a batch; nu, p scalar."""
	if (K, exact) not in _scorers:
		f = partial(record_loglik, K=K, exact=exact)
		_scorers[(K, exact)] = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, None, None)))
	return _scorers[(K, exact)]


def main(out_csv="approx_accuracy.csv"):
	rng = np.random.default_rng(SEED)
	# simulate everything first so all groups share one padded width (one compile per scorer)
	sims = {}
	for nu, p, L in itertools.product(NUS, PS, LS):
		sims[(nu, p, L)] = [simulate(nu, p, L, rng) for _ in range(N_PRODUCTS)]
	width = max(1, max(len(s) for v in sims.values() for s in v))

	rows, t0 = [], time.time()
	for (nu, p, L), recs in sims.items():
		X, NXO, LEN = pad_records(recs, [L] * N_PRODUCTS, max_n=width)
		X, NXO, LEN = jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN)
		ll_ref = np.asarray(scorer(K_REF, True)(X, NXO, LEN, float(nu), p))
		for K in KS:
			ll_fast = np.asarray(scorer(K, False)(X, NXO, LEN, float(nu), p))
			ll_exact = np.asarray(scorer(K, True)(X, NXO, LEN, float(nu), p))
			for r in range(N_PRODUCTS):
				rows.append((nu, p, L, K, r + 1, int(NXO[r]), ll_fast[r], ll_exact[r],
							 ll_fast[r] - ll_exact[r], ll_ref[r],
							 ll_fast[r] - ll_ref[r], ll_exact[r] - ll_ref[r]))
	print(f"scored {len(rows):,} (record, K) pairs in {time.time()-t0:.0f}s; "
		  f"N_QUAD = {N_QUAD}, max crossovers/record = {width}")

	header = "nu,p,L,K,product,n_xo,ll_fast,ll_exact,diff_fast_minus_exact,ll_ref_K200,err_fast,err_exact"
	with open(out_csv, "w") as fh:
		fh.write(header + "\n")
		for row in rows:
			fh.write(",".join([f"{row[0]}", f"{row[1]}", f"{row[2]}", f"{row[3]}", f"{row[4]}",
							   f"{row[5]}"] + [f"{v:.17g}" for v in row[6:]]) + "\n")
	return np.array(rows, dtype=float)


if __name__ == "__main__":
	A = main()
	nu, p, L, K = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
	ef, ee, d = A[:, 10], A[:, 11], A[:, 8]
	print(f"\nnon-finite values: {np.sum(~np.isfinite(A[:, 6:]))}")
	print(f"overall max |fast - exact| = {np.max(np.abs(d)):.2e}")
	print(f"overall max |fast - ref|   = {np.max(np.abs(ef)):.2e}")
	print(f"overall max |exact - ref|  = {np.max(np.abs(ee)):.2e}")

	def tab(label, mask_fn, keys):
		print(f"\nmax |error| by {label}")
		print(f"  {'':>14} {'fast-exact':>11} {'fast-ref':>11} {'exact-ref':>11}")
		for k in keys:
			m = mask_fn(k)
			print(f"  {str(k):>14} {np.max(np.abs(d[m])):11.2e} {np.max(np.abs(ef[m])):11.2e}"
				  f" {np.max(np.abs(ee[m])):11.2e}")

	tab("K", lambda k: K == k, KS)
	tab("nu", lambda k: nu == k, NUS)
	tab("(nu, L)", lambda k: (nu == k[0]) & (L == k[1]), list(itertools.product(NUS, LS)))
	tab("p", lambda k: p == k, PS)
	tab("(nu, K)", lambda k: (nu == k[0]) & (K == k[1]), list(itertools.product(NUS, KS)))
