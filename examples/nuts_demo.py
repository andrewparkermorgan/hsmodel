"""
Full Bayesian fit of the hierarchical Housworth-Stahl model with NUTS
(hsmodel.nuts) on the default simulated dataset.

Same model, priors and simulated data as examples/svi_demo.py, with parent and
chromosome random effects:
    logit p_ic      = a_p  + b_p  z_i + s_p  * e_p_i  + c_p_c,    e_p_i  ~ N(0,1),  c_p_c  ~ N(0, t_p^2)
    log(nu_ic - 1)  = a_nu + b_nu z_i + s_nu * e_nu_i + c_nu_c,   e_nu_i ~ N(0,1),  c_nu_c ~ N(0, t_nu^2)
with priors from hsmodel.model.PRIORS and data from hsmodel.simulate.simulate_dataset().
--no-chrom-effects fits the earlier parent-only model.

The parent effects are non-centered (the e's are sampled), which avoids the
funnel geometry that troubles NUTS when each parent contributes few records.
The chromosome effects are non-centered by default too (c / t is sampled).
Measured on the first 50 demo parents (2 chains x (400 warmup + 400 draws)):
non-centered took 54 min at a median of 15 leapfrog steps per draw, with ESS
498 and 243 for log_t_p and log_t_nu; centered took 106 min at 31 steps, with
ESS 54 and 68 and slowly drifting traces (same posterior means).  With many
parents each chromosome effect is tightly informed, and --chrom-param centered
may then mix better; that case has not been measured.

Unlike SVI, every NUTS leapfrog step needs the log-likelihood gradient for the
whole dataset: ~2 ms per parent on a CPU with K = 12 (the demo has 66 records
per parent), i.e. ~3 s for all 1500 parents.  With the settings below NUTS
settles at ~15 leapfrog steps per iteration, so the full demo dataset needs a
GPU or a long run.  Measured on a 2-core CPU: 50 parents, 2 chains x (400
warmup + 400 draws) took 44 min and ~2.5 GB of memory (0 divergences, R-hat
<= 1.004, ESS 230-1190).  --parents k fits the first k parents of the same
simulated dataset.

Settings that matter (all defaults here):
  * start at the prior medians or an SVI fit, not numpyro's random start
    (random starts sent early warmup to the 255-step tree-depth cap);
  * dense mass matrix for the globals, diagonal for the parent and chromosome effects;
  * K = 12 mixture terms (examples/quadrature_accuracy.py: error < 1e-7, ~1.6x cheaper);
  * chains run sequentially on CPU: 'parallel' chains on XLA host devices
    grew by ~200 MB/min until they ran out of memory in testing.

Usage:
    python examples/nuts_demo.py                           # all 1500 parents
    python examples/nuts_demo.py --parents 50 --warmup 400 --samples 400 --chains 2 --compare-svi
"""
import argparse
import os
import time


def parse_args(argv=None):
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--parents", type=int, default=1500, help="use the first k parents (default: all 1500)")
	ap.add_argument("--chains", type=int, default=4)
	ap.add_argument("--warmup", type=int, default=1000)
	ap.add_argument("--samples", type=int, default=1000, help="post-warmup draws per chain")
	ap.add_argument("--target-accept", type=float, default=0.9)
	ap.add_argument("--max-tree-depth", type=int, default=8)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--k-terms", type=int, default=12,
	                help="mixture terms in f* (examples/quadrature_accuracy.py: K=12 with 64 nodes keeps the "
	                     "per-record log-likelihood error below 1e-7 and is ~1.6x cheaper than K=20)")
	ap.add_argument("--quad-nodes", type=int, default=64,
	                help="Gauss-Legendre nodes for F* and G*; fewer than 64 is inaccurate at strong interference")
	ap.add_argument("--chain-method", choices=["auto", "parallel", "sequential", "vectorized"], default="auto",
	                help="auto: sequential on CPU, vectorized on GPU. 'parallel' runs one chain per XLA "
	                     "host device; on CPU its memory grew by ~200 MB/min until it ran out (JAX 0.8, "
	                     "numpyro 0.20), so it is opt-in only")
	ap.add_argument("--no-chrom-effects", action="store_true",
	                help="fit the parent-only model (no chromosome effects)")
	ap.add_argument("--chrom-param", choices=["centered", "noncentered"], default="noncentered",
	                help="parameterization of the chromosome effects (see the notes above)")
	ap.add_argument("--compare-svi", action="store_true",
	                help="also fit SVI to the same parents and overlay its posterior")
	ap.add_argument("--svi-steps", type=int, default=2500)
	ap.add_argument("--out", default="nuts", help="prefix for output files")
	return ap.parse_args(argv)


args = parse_args()

# Parallel chains on CPU need one XLA host device per chain, and the device
# count must be set before JAX initializes (hsmodel.likelihood creates arrays at
# import), so the arguments are parsed before hsmodel is imported.
import numpyro
if args.chain_method == "parallel":
	numpyro.set_host_device_count(min(args.chains, os.cpu_count() or 1))

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.diagnostics import summary as mcmc_summary

from hsmodel import likelihood
from hsmodel.batching import ParentBatcher
from hsmodel.likelihood import group_by_count
from hsmodel.model import GLOBALS, CHROM_SITES, as_covariate_matrix, check_standardized, global_entries, \
	entry_value, expand_truth
from hsmodel.nuts import run_nuts, svi_init_values, global_summary, chrom_draw_summary, parent_draws
from hsmodel.plots import plot_nuts_summary
from hsmodel.simulate import simulate_dataset, TRUTH_SAMPLED
from hsmodel.svi import fit as svi_fit, chrom_summary, chrom_recovery


def main():
	likelihood.set_quadrature(args.quad_nodes, args.k_terms)   # before anything is compiled
	chrom_effects = not args.no_chrom_effects
	centered = args.chrom_param == "centered"
	print("simulating (same dataset as examples/svi_demo.py) ...", flush=True)
	d = simulate_dataset(chrom_effects=chrom_effects)
	k = min(args.parents, d["P"])
	sel = d["par"] < k
	X, NXO, chrom, par = d["X"][sel], d["NXO"][sel], d["chrom"][sel], d["par"][sel]
	z, Lc = as_covariate_matrix(d["z"])[:k], jnp.asarray(d["chrom_len"])
	check_standardized(z)
	data = jax.tree_util.tree_map(jnp.asarray, group_by_count(X, NXO, chrom, len(d["chrom_len"]), par, k))
	print(f"  first {k} of {d['P']} parents: {sel.sum():,} records, {NXO.sum():,} crossovers; "
	      f"chromosome effects {'on (' + args.chrom_param + ')' if chrom_effects else 'off'}")
	print(f"  {args.chains} chains x ({args.warmup} warmup + {args.samples} draws), "
	      f"chain method {args.chain_method}, backend {jax.default_backend()}", flush=True)

	svi_params = None
	if args.compare_svi:
		print("SVI on the same parents ...", flush=True)
		bt = ParentBatcher(X, NXO, chrom, par, k, d["chrom_len"], batch=min(32, max(2, k // 5)))
		svi_params, _ = svi_fit(bt, z, steps=args.svi_steps, lr=0.02, log_every=0, chrom_effects=chrom_effects)
		del bt
		jax.clear_caches()          # release the compiled SVI step before NUTS compiles

	print("NUTS ...", flush=True)
	t0 = time.time()
	mcmc = run_nuts(data, Lc, z, args.chains, args.warmup, args.samples, args.target_accept,
	                args.max_tree_depth, args.seed,
	                init_values=svi_init_values(svi_params, centered) if svi_params is not None else None,
	                chain_method=args.chain_method, chrom_effects=chrom_effects, chrom_centered=centered)
	jax.block_until_ready(mcmc.get_samples())          # JAX dispatches asynchronously
	elapsed = time.time() - t0
	extra = mcmc.get_extra_fields()
	steps = np.asarray(extra["num_steps"])
	n_div = int(np.asarray(extra["diverging"]).sum())
	print(f"  done in {elapsed/60:.1f} min; divergent transitions: {n_div}; "
	      f"leapfrog steps per draw: median {np.median(steps):.0f}, max {steps.max()} "
	      f"(tree-depth cap {2 ** args.max_tree_depth - 1}); mean accept prob {np.mean(extra['accept_prob']):.2f}")

	by_chain = {k_: np.asarray(v) for k_, v in mcmc.get_samples(group_by_chain=True).items()}
	summ = global_summary(by_chain)
	print("\nposterior (NUTS)      mean     sd    2.5%   97.5%   n_eff  r_hat   truth"
	      + ("     SVI mean (sd)" if svi_params is not None else ""))
	K = z.shape[1]
	truth_x = expand_truth(TRUTH_SAMPLED, K)
	for key, site, kk, _ in global_entries([n for n in GLOBALS if n in summ], K):
		s = {q: entry_value(summ[site][q], kk) for q in ("mean", "std", "2.5%", "97.5%", "n_eff", "r_hat")}
		line = (f"  {key:12s}{s['mean']:7.3f} {s['std']:6.3f} {s['2.5%']:7.3f} {s['97.5%']:7.3f} "
		        f"{s['n_eff']:7.0f} {s['r_hat']:6.3f} {truth_x[key]:7.3f}")
		if svi_params is not None:
			line += (f"   {entry_value(svi_params[f'{site}_loc'], kk):7.3f} "
			         f"({entry_value(svi_params[f'{site}_scale'], kk):.3f})")
		print(line)

	truth_chrom = {"c_p": d["c_p"], "c_nu": d["c_nu"]} if chrom_effects else None
	if chrom_effects:
		chrom_nuts = chrom_draw_summary(by_chain)
		chrom_svi = chrom_summary(svi_params) if svi_params is not None else {}
		rh = mcmc_summary({n: by_chain[n] for n in CHROM_SITES}, prob=0.95)
		print()
		for site in CHROM_SITES:
			r, rmse, cover = chrom_recovery(chrom_nuts[site], truth_chrom[site])
			line = (f"  {site:5s} NUTS vs truth: correlation {r:.2f}, RMSE {rmse:.3f}, 95% coverage {cover:.0%}; "
			        f"max R-hat {np.max(rh[site]['r_hat']):.3f}, min ESS {np.min(rh[site]['n_eff']):.0f}")
			if site in chrom_svi:
				r2, rmse2, cover2 = chrom_recovery(chrom_svi[site], truth_chrom[site])
				line += f"  |  SVI: correlation {r2:.2f}, RMSE {rmse2:.3f}, 95% coverage {cover2:.0%}"
			print(line)

	p_draws, nu_draws = parent_draws({k_: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:])
	                                  for k_, v in by_chain.items()}, z)
	cover_p = np.mean((np.percentile(p_draws, 2.5, 0) <= d["p_i"][:k]) & (d["p_i"][:k] <= np.percentile(p_draws, 97.5, 0)))
	cover_nu = np.mean((np.percentile(nu_draws, 2.5, 0) <= d["nu_i"][:k]) & (d["nu_i"][:k] <= np.percentile(nu_draws, 97.5, 0)))
	print(f"\n  parent-level 95% intervals covering the true value: p_i {cover_p:.0%}, nu_i {cover_nu:.0%}")

	np.savez(f"{args.out}_samples.npz", n_divergent=n_div, **by_chain)
	print(f"  samples: {args.out}_samples.npz")
	if svi_params is not None:
		np.savez(f"{args.out}_svi_params.npz", **svi_params)
		print(f"  SVI parameters: {args.out}_svi_params.npz")
	print(f"  summary plot: {plot_nuts_summary(by_chain, n_div, f'{args.out}_summary.png', TRUTH_SAMPLED, svi_params, truth_chrom)}")


if __name__ == "__main__":
	main()
