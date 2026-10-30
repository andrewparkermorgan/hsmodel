"""
Fit the hierarchical Housworth-Stahl model to crossover data with NUTS
(`hsmodel nuts`): full Bayesian inference, using the whole dataset at every
leapfrog step.

Same model, priors, inputs and covariate handling as `hsmodel infer` (SVI).
NUTS is much more expensive than SVI: each leapfrog step needs the
log-likelihood gradient for every individual (~2 ms per individual with 66
meiotic products on one CPU core), and NUTS typically takes ~15 leapfrog
steps per draw.  --parents K fits only the first K individuals.

Inputs:
  --data FILE.npz       crossover data from `hsmodel preprocess` or
                        `hsmodel simulate`
  --covariates FILE     parent-level covariates (not needed for simulated data)
  --coefficients LIST   covariates to include (default: all)

Settings that matter (the defaults):
  * chains start at the prior medians, or with --svi-init at the means of an
    SVI fit to the same data (the SVI posterior is then also overlaid on the
    plot); numpyro's random start sends early warmup to the tree-depth cap;
  * dense mass matrix for the globals, diagonal for the parent and chromosome
    effects; chromosome effects non-centered (--chrom-param);
  * K = 12 mixture terms with 64 quadrature nodes (per-record log-likelihood
    error < 1e-7, ~1.6x cheaper than K = 20);
  * chains run sequentially on CPU ('parallel' chains on XLA host devices grew
    by ~200 MB/min until they ran out of memory in testing), vectorized on GPU.

Outputs (prefix --out): OUT_samples.npz (draws grouped by chain, the number of
divergent transitions, and the names of the individuals, covariates and
chromosomes), OUT_summary.png (posteriors, chromosome effects and traces),
OUT_svi_params.npz with --svi-init, and tables of posterior summaries with
effective sample sizes and R-hat on stdout (with the true values for simulated
data).  The data summary, progress and sampler diagnostics go to stderr.

Examples:
    hsmodel nuts --data simulated_data.npz --parents 50 --chains 2 --warmup 400 --samples 400
    hsmodel nuts --data crossovers.npz --covariates covariates.txt --standardize --svi-init --out nuts1
"""
import math
import os
import sys
import time

import numpy as np

from .errors import CommandError, InputError
from .inputs import (COVARIATE_HELP, add_input_arguments, load_fit_inputs, subset_parents, data_summary,
                     parent_lengths_of, global_table, chrom_table)

__doc__ += "\n" + COVARIATE_HELP


def add_arguments(ap):
	add_input_arguments(ap)
	ap.add_argument("--parents", type=int, default=None, help="fit only the first K individuals (default: all)")
	ap.add_argument("--chains", type=int, default=4)
	ap.add_argument("--warmup", type=int, default=1000)
	ap.add_argument("--samples", type=int, default=1000, help="post-warmup draws per chain")
	ap.add_argument("--target-accept", type=float, default=0.9)
	ap.add_argument("--max-tree-depth", type=int, default=8)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--k-terms", type=int, default=12,
	                help="mixture terms in f* (default 12: per-record log-likelihood error below 1e-7 with 64 "
	                     "nodes)")
	ap.add_argument("--quad-nodes", type=int, default=64,
	                help="Gauss-Legendre nodes for F* and G*; fewer than 64 is inaccurate at strong interference")
	ap.add_argument("--chain-method", choices=["auto", "parallel", "sequential", "vectorized"], default="auto",
	                help="auto: sequential on CPU, vectorized on GPU. 'parallel' runs one chain per XLA host "
	                     "device; on CPU its memory grew without bound in testing, so it is opt-in only")
	ap.add_argument("--chrom-param", choices=["centered", "noncentered"], default="noncentered",
	                help="parameterization of the chromosome effects (default noncentered)")
	ap.add_argument("--svi-init", action="store_true",
	                help="start the chains at an SVI fit to the same data (and overlay it on the plot)")
	ap.add_argument("--svi-steps", type=int, default=None,
	                help="SVI steps for --svi-init (default: 400 x individuals / 32)")
	ap.add_argument("--out", default="nuts_fit", help="prefix for output files (default nuts_fit)")


def run(a):
	for name in ("chains", "samples", "max_tree_depth", "k_terms", "quad_nodes"):
		if getattr(a, name) < 1:
			raise CommandError(f"--{name.replace('_', '-')} must be at least 1")
	if a.warmup < 0:
		raise CommandError("--warmup must be non-negative")
	if not 0 < a.target_accept < 1:
		raise CommandError("--target-accept must be between 0 and 1")
	if a.parents is not None and a.parents < 2:
		raise CommandError("--parents must be at least 2")
	if a.svi_steps is not None and a.svi_steps < 1:
		raise CommandError("--svi-steps must be positive")
	try:
		f = load_fit_inputs(a)
		P_all = f.P
		f = subset_parents(f, a.parents)
	except (InputError, OSError, KeyError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1
	d, P, names = f.d, f.P, f.names
	n_prod = len(d["NXO"])
	svi_steps = a.svi_steps if a.svi_steps is not None else math.ceil(400 * P / min(32, P))
	print(data_summary(f, a.chrom_effects, a.standardize) + "\n"
	      + (f"individuals used:                 the first {P:,} of {P_all:,}\n" if P < P_all else "")
	      + f"sampler: NUTS, {a.chains} chain(s) x ({a.warmup} warmup + {a.samples} draws); "
	        f"target accept {a.target_accept}; max tree depth {a.max_tree_depth}; chromosome effects "
	        f"{a.chrom_param}; K = {a.k_terms}, {a.quad_nodes} quadrature nodes; start: "
	        + (f"SVI fit ({svi_steps:,} steps)" if a.svi_init else "prior medians") + f"; seed {a.seed}",
	      file=sys.stderr, flush=True)

	# One XLA host device per chain for 'parallel' must be requested before JAX
	# initializes, so the JAX-based modules are imported only now.
	import numpyro
	if a.chain_method == "parallel":
		numpyro.set_host_device_count(min(a.chains, os.cpu_count() or 1))
	import warnings
	import jax
	import jax.numpy as jnp
	from numpyro.diagnostics import summary as mcmc_summary
	from . import likelihood
	from .model import GLOBALS, CHROM_SITES, global_entries, entry_value, expand_truth, check_standardized
	from .nuts import run_nuts, svi_init_values, global_summary, chrom_draw_summary
	from .plots import plot_nuts_summary

	likelihood.set_quadrature(a.quad_nodes, a.k_terms)            # before anything is compiled
	plengths = parent_lengths_of(d)
	data = jax.tree_util.tree_map(jnp.asarray, likelihood.group_by_count(
		d["X"], d["NXO"], d["chrom"], len(d["chrom_len"]), d["par"], P, parent_lengths=plengths))
	Lc = jnp.asarray(np.nan_to_num(np.asarray(d["chrom_len"], dtype=float), nan=1.0))   # unused with data["L"]
	centered = a.chrom_param == "centered"
	with warnings.catch_warnings():
		warnings.simplefilter("always")
		warnings.showwarning = lambda msg, *args, **kw: print(f"warning: {msg}", file=sys.stderr)
		if not a.svi_init:                      # the SVI fit runs the same check
			check_standardized(f.z, names)

		svi_params = None
		if a.svi_init:
			from .batching import ParentBatcher
			from .svi import fit as svi_fit
			print("SVI fit for the starting point ...", file=sys.stderr, flush=True)
			bt = ParentBatcher(d["X"], d["NXO"], d["chrom"], d["par"], P, d["chrom_len"], batch=min(32, P),
			                   seed=a.seed, parent_lengths=plengths)
			svi_params, _ = svi_fit(bt, f.z, steps=svi_steps, lr=0.02, seed=a.seed,
			                        log_every=max(1, svi_steps // 5), chrom_effects=a.chrom_effects,
			                        covariate_names=names, log=sys.stderr)
			del bt
			jax.clear_caches()                  # release the compiled SVI step before NUTS compiles

	print("NUTS ...", file=sys.stderr, flush=True)
	t0 = time.time()
	mcmc = run_nuts(data, Lc, f.z, a.chains, a.warmup, a.samples, a.target_accept, a.max_tree_depth, a.seed,
	                init_values=svi_init_values(svi_params, centered) if svi_params is not None else None,
	                chain_method=a.chain_method, progress_bar=sys.stderr.isatty(),
	                chrom_effects=a.chrom_effects, chrom_centered=centered)
	jax.block_until_ready(mcmc.get_samples())          # JAX dispatches asynchronously
	elapsed = time.time() - t0
	extra = mcmc.get_extra_fields()
	steps = np.asarray(extra["num_steps"])
	n_div = int(np.asarray(extra["diverging"]).sum())
	print(f"done in {elapsed / 60:.1f} min; divergent transitions: {n_div}; leapfrog steps per draw: median "
	      f"{np.median(steps):.0f}, max {steps.max()} (tree-depth cap {2 ** a.max_tree_depth - 1}); mean "
	      f"acceptance probability {np.mean(extra['accept_prob']):.2f}", file=sys.stderr, flush=True)

	# ---- results ---------------------------------------------------------------
	by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
	summ = global_summary(by_chain)
	K = len(names)
	rows = {}
	for key, site, k, _ in global_entries([n for n in GLOBALS if n in summ], K, names):
		s = {q: entry_value(summ[site][q], k) for q in ("mean", "std", "2.5%", "97.5%", "n_eff", "r_hat")}
		rows[key] = (s["mean"], s["std"], s["2.5%"], s["97.5%"], f"{s['n_eff']:.0f}", f"{s['r_hat']:.4f}")
	worst_rhat = max(float(r[5]) for r in rows.values())
	print(f"# Housworth-Stahl NUTS fit: {P} individuals, {n_prod} meiotic products, "
	      f"{int(d['NXO'].sum())} crossovers; {a.chains} chains x {a.samples} draws; {n_div} divergent transitions")
	if a.standardize:
		print("# covariates standardized: " + "; ".join(f"{n} mean {m:.4g} SD {sd:.4g}"
		                                                  for n, m, sd in zip(names, f.means, f.sds)))
	print(global_table(rows, expand_truth(f.truth, K, names), extra_cols=("n_eff", "r_hat")))
	chrom = chrom_draw_summary(by_chain)
	if chrom:
		print("\n" + chrom_table(chrom, d["chrom_names"], f.truth_chrom))
		rh = mcmc_summary({n: by_chain[n] for n in CHROM_SITES}, prob=0.95)
		worst_rhat = max(worst_rhat, *(float(np.max(rh[n]["r_hat"])) for n in CHROM_SITES))
	problems = ([f"{n_div} divergent transition(s)"] if n_div else []) + \
		([f"largest R-hat {worst_rhat:.3f} (want <= 1.01)"] if worst_rhat > 1.01 else [])
	if problems:
		hint = "more warmup and draws, a higher --target-accept" + ("" if a.svi_init else ", or --svi-init")
		print(f"warning: {'; '.join(problems)}. Consider {hint}.", file=sys.stderr)

	meta = dict(n_divergent=n_div, individual_ids=np.array(f.iids), covariate_names=np.array(names),
	            chrom_names=np.asarray(d["chrom_names"]), covariate_means=f.means, covariate_sds=f.sds,
	            standardized=a.standardize, num_steps=steps, accept_prob=np.asarray(extra["accept_prob"]))
	np.savez(f"{a.out}_samples.npz", **by_chain, **meta)
	written = [f"{a.out}_samples.npz"]
	if svi_params is not None:
		np.savez(f"{a.out}_svi_params.npz", **svi_params)
		written.append(f"{a.out}_svi_params.npz")
	written.append(plot_nuts_summary(by_chain, n_div, f"{a.out}_summary.png", truth=f.truth, svi_params=svi_params,
	                                 truth_chrom=f.truth_chrom, covariate_names=names,
	                                 chrom_names=list(d["chrom_names"]),
	                                 title=f"Housworth-Stahl NUTS fit: {P} individuals, {n_prod:,} meiotic products"))
	print(f"\nwrote {', '.join(written)}", file=sys.stderr)
	return 0
