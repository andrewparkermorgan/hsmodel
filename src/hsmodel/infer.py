"""
Fit the hierarchical Housworth-Stahl model to crossover data with SVI
(`hsmodel infer`).

Inputs:
  --data FILE.npz       crossover data from `hsmodel preprocess` or
                        `hsmodel simulate`
  --covariates FILE     parent-level covariates (not needed for simulated data)
  --coefficients LIST   covariates to include (default: all)

Outputs (prefix --out): OUT_params.npz (guide parameters, plus the names of the
individuals, covariates and chromosomes, and any standardization used),
OUT_summary.png, and tables of posterior summaries on stdout (with the true
values for simulated data).  Progress and the data summary go to stderr.

Examples:
    hsmodel infer --data crossovers.npz --covariates covariates.txt \\
        --coefficients age,sex --standardize --guide-type meanfield --out fit1
    hsmodel infer --data simulated_data.npz --out sim_fit
"""
import math
import sys

import numpy as np

from .errors import CommandError, InputError
from .inputs import (COVARIATE_HELP, add_input_arguments, load_fit_inputs, data_summary, parent_lengths_of,
                     global_table, chrom_table)

__doc__ += "\n" + COVARIATE_HELP


def add_arguments(ap):
	add_input_arguments(ap)
	ap.add_argument("--steps", type=int, default=None,
	                help="optimization steps (default: 400 x individuals / batch, i.e. ~400 updates per "
	                     "individual; with --no-parent-effects there are no per-individual parameters and "
	                     "fewer steps usually suffice)")
	ap.add_argument("--guide-type", choices=["meanfield", "lowrank"], default="meanfield")
	ap.add_argument("--lr", type=float, default=0.02, help="learning rate (default 0.02)")
	ap.add_argument("--rank", type=int, default=4, help="rank of the low-rank guide's covariance (default 4)")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--batch", type=int, default=32, help="individuals per minibatch (default 32)")
	ap.add_argument("--exact", action="store_true",
	                help="evaluate F* and G* in the likelihood in closed form (regularized incomplete gamma "
	                     "function) instead of by Gauss-Legendre quadrature.  Gradients in nu are several times "
	                     "slower.  The closed forms drop the ~2^-K tail of the sum in f* that quadrature keeps, "
	                     "so consider a larger --truncate-at (e.g. 60)")
	ap.add_argument("--truncate-at", type=int, default=None, metavar="K",
	                help="number of terms kept in the infinite sum f*(d) = sum_k 2^-k Gamma(d; k nu, 2 nu (1-p)) "
	                     "(default 20; omitted terms have total weight 2^-K)")
	ap.add_argument("--out", default="svi_fit", help="prefix for output files (default svi_fit)")


def run(a):
	if a.steps is not None and a.steps < 1:
		raise CommandError("--steps must be positive")
	if a.lr <= 0:
		raise CommandError("--lr must be positive")
	if a.rank < 1:
		raise CommandError("--rank must be at least 1")
	if a.batch < 1:
		raise CommandError("--batch must be at least 1")
	if a.truncate_at is not None and a.truncate_at < 1:
		raise CommandError("--truncate-at must be at least 1")
	try:
		f = load_fit_inputs(a)
	except (InputError, OSError, KeyError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1

	d, P, names = f.d, f.P, f.names
	n_prod, n_xo = len(d["NXO"]), int(d["NXO"].sum())
	batch = min(a.batch, P)
	steps = a.steps if a.steps is not None else math.ceil(400 * P / batch)
	print(data_summary(f, a.chrom_effects, a.standardize, a.parent_effects) + "\n"
	      f"guide: {a.guide_type}{f' (rank {a.rank})' if a.guide_type == 'lowrank' else ''}; "
	      f"{steps:,} steps of {batch} individuals (~{steps * batch / P:.0f} updates per individual); "
	      f"lr {a.lr}; seed {a.seed}", file=sys.stderr, flush=True)

	# JAX-based modules are imported only once the inputs have been checked
	import warnings
	from . import likelihood
	K_terms = a.truncate_at if a.truncate_at is not None else likelihood.K_TERMS
	print(f"likelihood: F* and G* {'in closed form (incomplete gamma)' if a.exact else f'by quadrature ({likelihood.N_QUAD} Gauss-Legendre nodes)'}; "
	      f"sum in f* truncated at K = {K_terms} terms", file=sys.stderr, flush=True)
	from .batching import ParentBatcher
	from .model import expand_truth
	from .plots import plot_summary
	from .svi import fit, summarize, chrom_summary, intercept_nu, intercept_p

	bt = ParentBatcher(d["X"], d["NXO"], d["chrom"], d["par"], P, d["chrom_len"], batch=batch,
	                   seed=a.seed, parent_lengths=parent_lengths_of(d))
	with warnings.catch_warnings():
		warnings.simplefilter("always")
		warnings.showwarning = lambda msg, *args, **kw: print(f"warning: {msg}", file=sys.stderr)
		params, losses = fit(bt, f.z, steps=steps, lr=a.lr, seed=a.seed, log_every=max(1, steps // 10),
		                     chrom_effects=a.chrom_effects, guide_type=a.guide_type, rank=a.rank,
		                     covariate_names=names, log=sys.stderr, parent_effects=a.parent_effects,
		                     K_terms=K_terms, exact=a.exact)

	# ---- results ---------------------------------------------------------------
	truth_x = expand_truth(f.truth, len(names), names)
	print(f"# Housworth-Stahl SVI fit: {P} individuals, {n_prod} meiotic products, {n_xo} crossovers; "
	      f"guide {a.guide_type}; {steps} steps; likelihood {'exact' if a.exact else 'quadrature'}, K = {K_terms}")
	if a.standardize:
		print("# covariates standardized: " + "; ".join(f"{n} mean {m:.4g} SD {sd:.4g}"
		                                                  for n, m, sd in zip(names, f.means, f.sds)))
	print(global_table(summarize(params, names), truth_x))
	cs = chrom_summary(params)
	if cs:
		print("\n" + chrom_table(cs, d["chrom_names"], f.truth_chrom))
	np.savez(f"{a.out}_params.npz", **params, individual_ids=np.array(f.iids), covariate_names=np.array(names),
	         chrom_names=np.asarray(d["chrom_names"]), covariate_means=f.means, covariate_sds=f.sds,
	         standardized=a.standardize, losses=losses, likelihood_exact=a.exact, likelihood_K=K_terms)
	plot = plot_summary(params, losses, f"{a.out}_summary.png", truth=f.truth, truth_chrom=f.truth_chrom,
	                    covariate_names=names, chrom_names=list(d["chrom_names"]),
	                    title=f"Housworth-Stahl SVI fit ({a.guide_type} guide): {P} individuals, "
	                          f"{n_prod:,} meiotic products, {steps:,} steps")
	print(f"\nwrote {a.out}_params.npz and {plot}", file=sys.stderr)
	nu0 = intercept_nu(params)
	at = ("covariates at their means (standardized)" if a.standardize else "all covariates = 0")
	effects = [e for e, on in (("chromosome", a.chrom_effects), ("individual", a.parent_effects)) if on]
	if effects:
		at += f", {' and '.join(effects)} effects at their mean (0)"
	print(f"\nimplied nu at the intercept (nu = 1 + exp(a_nu); {at}):\n"
	      f"  posterior median {nu0['median']:.4g}, mean {nu0['mean']:.4g}, "
	      f"95% interval [{nu0['lower95']:.4g}, {nu0['upper95']:.4g}]", file=sys.stderr)
	p0 = intercept_p(params)
	print(f"implied p at the intercept (p = 1 / (1 + exp(-a_p)); same conditions):\n"
	      f"  posterior median {p0['median']:.4g}, mean {p0['mean']:.4g}, "
	      f"95% interval [{p0['lower95']:.4g}, {p0['upper95']:.4g}]", file=sys.stderr)
	return 0
