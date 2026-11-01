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

from .model import (GLOBALS, CHROM_SITES, as_covariate_matrix, check_standardized, global_entries, entry_value,
                    model, guide, guide_lowrank, lowrank_marginals)


def fit(batcher, z, steps=3000, lr=0.01, seed=0, num_particles=1, log_every=500, chrom_effects=True,
        guide_type="meanfield", rank=4, covariate_names=None, log=None, parent_effects=True):
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
	svi = SVI(partial(model, chrom_effects=chrom_effects, parent_effects=parent_effects), guide_fn, opt,
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


def summarize(params, covariate_names=None):
	"""{display key: (mean, sd, 2.5%, 97.5%)} for the globals present in params,
	one entry per coefficient for b_p and b_nu (keys like "b_p[age]")."""
	names = [n for n in GLOBALS if f"{n}_loc" in params]
	K = np.atleast_1d(params["b_p_loc"]).size if "b_p_loc" in params else 1
	out = {}
	for key, site, k, _ in global_entries(names, K, covariate_names):
		m, s = entry_value(params[f"{site}_loc"], k), entry_value(params[f"{site}_scale"], k)
		out[key] = (m, s, m - 1.96 * s, m + 1.96 * s)
	return out


def chrom_summary(params):
	"""{c_p / c_nu: (mean [C], 2.5% [C], 97.5% [C])} from the guide, if fitted."""
	out = {}
	for n in CHROM_SITES:
		if f"{n}_loc" in params:
			m, s = np.asarray(params[f"{n}_loc"]), np.asarray(params[f"{n}_scale"])
			out[n] = (m, m - 1.96 * s, m + 1.96 * s)
	return out


def chrom_recovery(est, truth):
	"""Agreement between estimated and true chromosome effects:
	(correlation, RMSE, coverage of the 95% intervals)."""
	m, lo, hi = est
	truth = np.asarray(truth)
	return (float(np.corrcoef(m, truth)[0, 1]), float(np.sqrt(np.mean((m - truth) ** 2))),
	        float(np.mean((lo <= truth) & (truth <= hi))))
