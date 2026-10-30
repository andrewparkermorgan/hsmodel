"""
Full Bayesian fit of the hierarchical Housworth-Stahl model with NUTS (no
subsampling: every leapfrog step uses the whole dataset).

Same model and priors as the SVI fit (hsmodel.model):
    logit p_ic      = a_p  + z_i . b_p  + s_p  * e_p_i  + c_p_c
    log(nu_ic - 1)  = a_nu + z_i . b_nu + s_nu * e_nu_i + c_nu_c

The parent effects are non-centered (the e's are sampled), which avoids the
funnel geometry that troubles NUTS when each parent contributes few records.
The chromosome effects are non-centered by default too (c / t is sampled).
Measured on the first 50 demo parents (2 chains x (400 warmup + 400 draws)):
non-centered took 54 min at a median of 15 leapfrog steps per draw, with ESS
498 and 243 for log_t_p and log_t_nu; centered took 106 min at 31 steps, with
ESS 54 and 68 and slowly drifting traces (same posterior means).

Every leapfrog step needs the log-likelihood gradient for the whole dataset:
~2 ms per parent on a CPU with K = 12 (66 records per parent).  Settings that
matter (the defaults of run_nuts and examples/nuts_demo.py):
  * start at the prior medians or an SVI fit (svi_init_values), not numpyro's
    random start (random starts sent early warmup to the tree-depth cap);
  * dense mass matrix for the globals, diagonal for the parent and chromosome effects;
  * K = 12 mixture terms (likelihood.set_quadrature(64, 12); error < 1e-7);
  * chains run sequentially on CPU: 'parallel' chains on XLA host devices
    grew by ~200 MB/min until they ran out of memory in testing.
"""
import sys
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value
from numpyro.infer.reparam import LocScaleReparam
from numpyro.diagnostics import summary as mcmc_summary

from .likelihood import count_loglik_per_parent
from .model import GLOBALS, CHROM_SITES, globals_used, sample_globals, as_covariate_matrix


# ---------------------------------------------------------------------------
# Model (full data, no subsampling)
# ---------------------------------------------------------------------------
def model(data, Lc, z, chrom_effects=True, chrom_centered=False):
	"""data: hsmodel.likelihood.group_by_count layout with one parent per index; Lc [C];
	z [P, K] standardized covariates (1-D: one covariate)."""
	z = z[:, None] if z.ndim == 1 else z
	P = z.shape[0]
	g = sample_globals(chrom_effects, z.shape[1])
	with numpyro.plate("parents", P):
		e_p = numpyro.sample("e_p", dist.Normal(0.0, 1.0))
		e_nu = numpyro.sample("e_nu", dist.Normal(0.0, 1.0))
	lp = g["a_p"] + z @ g["b_p"] + jnp.exp(g["log_s_p"]) * e_p                 # [P]
	ln = g["a_nu"] + z @ g["b_nu"] + jnp.exp(g["log_s_nu"]) * e_nu             # [P]
	if chrom_effects:
		# centered: sample c directly; noncentered: sample c / t (site "<name>_decentered")
		config = {} if chrom_centered else {n: LocScaleReparam(centered=0) for n in CHROM_SITES}
		with numpyro.handlers.reparam(config=config), numpyro.plate("chromosomes", Lc.shape[0]):
			c_p = numpyro.sample("c_p", dist.Normal(0.0, jnp.exp(g["log_t_p"])))
			c_nu = numpyro.sample("c_nu", dist.Normal(0.0, jnp.exp(g["log_t_nu"])))
		p = jax.nn.sigmoid(lp[:, None] + c_p[None, :])                            # [P, C]
		nu = 1.0 + jnp.exp(ln[:, None] + c_nu[None, :])
	else:
		p, nu = jax.nn.sigmoid(lp), 1.0 + jnp.exp(ln)
	numpyro.factor("obs", jnp.sum(count_loglik_per_parent(data, Lc, nu, p)))


def parent_draws(samples, z):
	"""Parent-level p_i and nu_i (at a chromosome effect of 0) for every draw: [draws, P]."""
	z = as_covariate_matrix(z)
	b_p = np.asarray(samples["b_p"]).reshape(len(samples["a_p"]), -1)             # [draws, K]
	b_nu = np.asarray(samples["b_nu"]).reshape(len(samples["a_nu"]), -1)
	p = 1 / (1 + np.exp(-(samples["a_p"][:, None] + b_p @ z.T
	                      + np.exp(samples["log_s_p"])[:, None] * samples["e_p"])))
	nu = 1 + np.exp(samples["a_nu"][:, None] + b_nu @ z.T
	                + np.exp(samples["log_s_nu"])[:, None] * samples["e_nu"])
	return p, nu


def svi_init_values(svi_params, chrom_centered=False):
	"""Starting point for NUTS from an SVI fit on the same parents (guide means)."""
	vals = {n: svi_params[f"{n}_loc"] for n in GLOBALS if f"{n}_loc" in svi_params}
	vals["e_p"] = svi_params["e_p_loc"]
	vals["e_nu"] = svi_params["e_nu_loc"]
	for n, t in (("c_p", "log_t_p"), ("c_nu", "log_t_nu")):
		if f"{n}_loc" in svi_params:
			c = np.asarray(svi_params[f"{n}_loc"])
			if chrom_centered:
				vals[n] = c
			else:
				vals[f"{n}_decentered"] = c / np.exp(float(svi_params[f"{t}_loc"]))
	return vals


def run_nuts(data, Lc, z, chains=4, warmup=1000, samples=1000, target_accept=0.9,
             max_tree_depth=8, seed=0, init_values=None, chain_method="auto", progress_bar=None,
             chrom_effects=True, chrom_centered=False):
	"""Returns the numpyro MCMC object (samples grouped by chain available).

	Chains start at init_values (e.g. svi_init_values(...)) if given, otherwise
	at the prior medians (globals at their prior means, all effects at 0);
	numpyro's default random start puts the chains far from the posterior, and
	the first warmup iterations then hit the tree-depth cap.  The mass matrix is
	dense for the globals (their posteriors are correlated) and diagonal for the
	parent and chromosome effects."""
	n_dev = jax.local_device_count()
	init = init_to_value(values=init_values) if init_values is not None else init_to_median(num_samples=15)
	kernel = NUTS(partial(model, chrom_effects=chrom_effects, chrom_centered=chrom_centered),
	              target_accept_prob=target_accept, max_tree_depth=max_tree_depth, init_strategy=init,
	              find_heuristic_step_size=True, dense_mass=[tuple(globals_used(chrom_effects))])
	if chain_method == "auto":
		# Parallel chains on CPU host devices leaked memory in testing; sequential is
		# safe and on a CPU about as fast as vectorized (which runs chains in lockstep).
		chain_method = "vectorized" if jax.default_backend() == "gpu" else "sequential"
	if chain_method == "parallel" and n_dev < chains:
		chain_method = "sequential"
	mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
	            chain_method=chain_method,
	            progress_bar=sys.stdout.isatty() if progress_bar is None else progress_bar)
	mcmc.run(jax.random.PRNGKey(seed), data, Lc, jnp.asarray(z),
	         extra_fields=("diverging", "num_steps", "accept_prob"))
	return mcmc


def global_summary(by_chain):
	"""numpyro summary (mean, sd, quantiles, n_eff, r_hat) for the globals present.
	by_chain: {name: [chains, draws, ...]}, e.g. mcmc.get_samples(group_by_chain=True)."""
	return mcmc_summary({k: np.asarray(by_chain[k]) for k in GLOBALS if k in by_chain}, prob=0.95)


def chrom_draw_summary(by_chain):
	"""{c_p / c_nu: (mean [C], 2.5% [C], 97.5% [C])} from the NUTS draws."""
	out = {}
	for n in CHROM_SITES:
		if n in by_chain:
			d = np.asarray(by_chain[n]).reshape(-1, np.asarray(by_chain[n]).shape[-1])
			out[n] = (d.mean(0), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0))
	return out
