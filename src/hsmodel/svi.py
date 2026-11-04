"""
Stochastic variational inference for the hierarchical Housworth-Stahl model
(hsmodel.model), and summaries of the fitted guide.

Guide: mean-field Gaussian (model.guide) or structured low-rank
(model.guide_lowrank).  Minibatches of parents come from
hsmodel.batching.ParentBatcher (a new random partition every epoch, padded to
fixed capacities so a single compiled step serves every batch).
"""
import sys
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
import optax

from .model import as_covariate_matrix, check_standardized, model, guide, guide_lowrank, lowrank_marginals
from .summaries import summarize, chrom_summary, chrom_recovery, intercept_nu, intercept_p

# The summaries of guide parameters live in hsmodel.summaries (no JAX) and are re-exported here.
__all__ = ["fit", "summarize", "chrom_summary", "chrom_recovery", "intercept_nu", "intercept_p"]


def fit(batcher, z, steps=3000, lr=0.01, seed=0, num_particles=1, log_every=500, chrom_effects=True,
        guide_type="meanfield", rank=4, covariate_names=None, log=None, parent_effects=True, K_terms=None,
        exact=False):
	"""batcher: ParentBatcher; z [P, K] standardized parent covariates (1-D: one).
	guide_type: "meanfield" (guide) or "lowrank" (guide_lowrank, with `rank`).
	For "lowrank" the returned params also hold the marginal {site}_loc and
	{site}_scale from lowrank_marginals, so downstream code is unchanged.

	Convergence: each parent's guide parameters are updated only when the parent
	is in a batch, i.e. about steps * B / P times.  With ~50 updates per parent
	the between-parent SDs (log_s_p, log_s_nu) collapse towards zero; with ~400
	they were recovered (200-parent test).  Choose steps ~ 400 * P / B, e.g.
	~19,000 steps for the 1500-parent demo with B = 32.  Without parent effects
	(parent_effects=False) there are no per-parent parameters, so far fewer
	steps suffice.

	Likelihood: K_terms terms of the sum over k in f* (None: likelihood.K_TERMS,
	i.e. 20); exact=True evaluates F* and G* in closed form with the
	incomplete gamma function instead of by quadrature (slower gradients in nu).

	Progress (every log_every steps; 0 for none) is written to `log`
	(default sys.stdout)."""
	z = as_covariate_matrix(z)
	check_standardized(z, covariate_names)
	K = z.shape[1]
	z = jnp.asarray(z)
	P, Lc = batcher.P, batcher.Lc
	if guide_type == "lowrank":
		guide_fn = partial(guide_lowrank, chrom_effects=chrom_effects, rank=rank, parent_effects=parent_effects)
	elif guide_type == "meanfield":
		guide_fn = partial(guide, chrom_effects=chrom_effects, parent_effects=parent_effects)
	else:
		raise ValueError(f"unknown guide_type {guide_type!r}")
	opt = optax.chain(optax.clip_by_global_norm(10.0),
	                  optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05)))
	svi = SVI(partial(model, chrom_effects=chrom_effects, parent_effects=parent_effects, K_terms=K_terms,
	                  exact=exact), guide_fn, opt,
	          loss=Trace_ELBO(num_particles=num_particles))
	batches = batcher.stream()
	ix, data = next(batches)
	state = svi.init(jax.random.PRNGKey(seed), data, Lc, ix, z, P)

	def _step(s, data, ix):
		s, loss = svi.update(s, data, Lc, ix, z, P)
		finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(v)) for v in
		                            jax.tree_util.tree_leaves(svi.optim.get_params(s.optim_state))]))
		return s, loss, finite

	update = jax.jit(_step)
	losses, t0 = [], time.time()
	for it in range(steps):
		if it > 0:
			ix, data = next(batches)
		state, loss, params_finite = update(state, data, ix)
		losses.append(float(loss))
		# svi.update returns the loss at the parameters *before* the step, so a nan
		# gradient shows up first as non-finite parameters, one step before a nan loss.
		if not np.isfinite(losses[-1]) or not bool(params_finite):
			raise FloatingPointError(
				f"non-finite {'loss' if not np.isfinite(losses[-1]) else 'parameters'} at SVI step "
				f"{it + 1} (loss {losses[-1]}). Re-run with jax.config.update('jax_debug_nans', True) "
				f"to locate the operation that produced it.")
		if log_every and (it + 1) % log_every == 0:
			print(f"  step {it+1:5d}  -ELBO (running mean) {np.mean(losses[-log_every:]):.1f}"
			      f"  [{time.time()-t0:.0f}s]", file=log or sys.stdout, flush=True)
	params = {k: np.asarray(v) for k, v in svi.get_params(state).items()}
	if guide_type == "lowrank":
		params.update(lowrank_marginals(params, chrom_effects, K, parent_effects))
	return params, np.array(losses)
