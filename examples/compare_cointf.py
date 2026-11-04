"""
Check hsmodel's record likelihood against the `cointf` package
(https://github.com/aabiddanda/cointf).

Simulates meiotic products (one chromosome in one gamete) on a chromosome of
length L with fixed nu and p (no chromosome- or individual-level variation),
then evaluates each product's log-likelihood with:

  hsmodel   quadrature   F*, G* by Gauss-Legendre quadrature, K = 20 terms in f*
                         (what `hsmodel infer` uses by default)
  hsmodel   exact        F*, G* in closed form (incomplete gamma), K = 20
  hsmodel   exact K=200  the same with 200 terms
  cointf    loglik_fast  windowed series, truncation error < e^-40
  cointf    loglik_naive 200-term series, one meiosis at a time

The two packages parameterize the same model differently: cointf scales
distances so that f* has unit rate, with interference gaps q f*(q x; nu) for
q = 1 - p, while hsmodel puts the rate 2 nu (1 - p) into the gamma terms of
f*.  These are identical, as are cointf's no-crossover term G(qL) and
hsmodel's 1 - G*(L), so the log-likelihoods should agree to rounding error.

Usage:
    python examples/compare_cointf.py [--n 100] [--L 1.0] [--nu 10] [--p 0.05] [--seed 1]
                                      [--csv compare_cointf.csv]
Requires cointf (pip install git+https://github.com/aabiddanda/cointf).
"""
import argparse
import sys
import time
from functools import partial

import numpy as np


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n", type=int, default=100, help="meiotic products to simulate (default 100)")
	ap.add_argument("--L", type=float, default=1.0, help="chromosome length in Morgans (default 1.0)")
	ap.add_argument("--nu", type=float, default=10.0, help="interference parameter (default 10)")
	ap.add_argument("--p", type=float, default=0.05, help="escape fraction (default 0.05)")
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--csv", default="compare_cointf.csv", help="per-product log-likelihoods (default "
	                                                             "compare_cointf.csv; '' for none)")
	a = ap.parse_args(argv)
	try:
		import cointf
		from cointf import loglik_fast, loglik_naive
	except ImportError:
		sys.exit("error: cointf is not installed (pip install git+https://github.com/aabiddanda/cointf)")
	import jax
	import jax.numpy as jnp
	from hsmodel.data import pad_records
	from hsmodel.likelihood import record_loglik
	from hsmodel.simulate import simulate_product

	# ---- simulate (hsmodel's simulator, independent of both likelihoods) --------
	rng = np.random.default_rng(a.seed)
	products = [simulate_product(a.nu, a.p, a.L, rng) for _ in range(a.n)]
	counts = np.array([len(x) for x in products])
	X, NXO, LEN = pad_records(products, [a.L] * a.n)
	print(f"{a.n} meiotic products on L = {a.L} M, nu = {a.nu}, p = {a.p} (seed {a.seed}): "
	      f"{counts.sum()} crossovers, {counts.min()}-{counts.max()} per product (mean {counts.mean():.2f})")
	print(f"hsmodel (this package) vs cointf {cointf.__version__}\n")

	# ---- evaluate ----------------------------------------------------------------
	def hs(**kw):
		f = jax.jit(jax.vmap(partial(record_loglik, **kw), in_axes=(0, 0, 0, None, None)))
		args = (jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN), a.nu, a.p)
		jax.block_until_ready(f(*args))                     # compile outside the timing
		t = time.perf_counter()
		out = np.asarray(jax.block_until_ready(f(*args)))
		return out, time.perf_counter() - t

	data = [list(x) for x in products]
	methods = {}
	methods["hsmodel quadrature (K=20)"] = hs()
	methods["hsmodel exact (K=20)"] = hs(exact=True)
	methods["hsmodel exact (K=200)"] = hs(exact=True, K=200)
	t = time.perf_counter()
	methods["cointf loglik_fast"] = (np.asarray(loglik_fast(data, a.L, nu=a.nu, p=a.p, per_meiosis=True)),
	                                 time.perf_counter() - t)
	t = time.perf_counter()
	methods["cointf loglik_naive"] = (np.array([loglik_naive([x], a.L, nu=a.nu, p=a.p) for x in data]),
	                                  time.perf_counter() - t)

	# ---- report ------------------------------------------------------------------
	ref_name = "cointf loglik_naive"
	ref = methods[ref_name][0]
	w = max(len(k) for k in methods)
	print(f"{'method':{w}}  {'total log-likelihood':>22}  {'total - naive':>14}  "
	      f"{'max |per-product - naive|':>26}  {'time':>9}")
	for name, (ll, secs) in methods.items():
		print(f"{name:{w}}  {ll.sum():22.12f}  {ll.sum() - ref.sum():14.2e}  "
		      f"{np.max(np.abs(ll - ref)):26.2e}  {secs * 1e3:7.1f} ms")
	print("\n(times: one evaluation of all products, after compilation for hsmodel)")

	names = list(methods)
	print("\nmax |per-product difference| between every pair of methods:")
	print(" " * (w + 2) + "".join(f"{i + 1:>10}" for i in range(len(names))))
	for i, ni in enumerate(names):
		row = "".join(f"{np.max(np.abs(methods[ni][0] - methods[nj][0])):10.1e}" if j != i else f"{'-':>10}"
		              for j, nj in enumerate(names))
		print(f"{i + 1} {ni:{w}}{row}")

	# per crossover count: where the differences come from
	print("\nmax |hsmodel - cointf loglik_naive| by crossovers per product:")
	print(f"  {'COs':>3} {'products':>8}" + "".join(f"  {n.replace('hsmodel ', ''):>20}" for n in names[:3]))
	for c in np.unique(counts):
		m = counts == c
		print(f"  {c:3d} {m.sum():8d}" + "".join(f"  {np.max(np.abs(methods[n][0][m] - ref[m])):20.1e}"
		                                       for n in names[:3]))

	if a.csv:
		cols = ["product", "n_crossovers", "positions"] + [n.replace(" ", "_").replace("(", "").replace(")", "")
		                                                   .replace("=", "") for n in names]
		with open(a.csv, "w") as fh:
			fh.write(",".join(cols) + "\n")
			for r in range(a.n):
				pos = ";".join(f"{v:.10g}" for v in products[r])
				fh.write(f"{r + 1},{counts[r]},{pos}," + ",".join(f"{methods[n][0][r]:.17g}" for n in names) + "\n")
		print(f"\nper-product log-likelihoods written to {a.csv}")
	return methods


if __name__ == "__main__":
	main()
