"""
Hierarchical Housworth-Stahl model with parent and chromosome random effects:
priors, the NumPyro model for minibatches of parents, and the variational
guides (mean-field and structured low-rank).

For parent i (e.g., a mother) with standardized covariates z_i and chromosome c:
    logit p_ic      = a_p  + z_i . b_p  + s_p  * e_p_i  + c_p_c,    e_p_i  ~ N(0,1),  c_p_c  ~ N(0, t_p^2)
    log(nu_ic - 1)  = a_nu + z_i . b_nu + s_nu * e_nu_i + c_nu_c,   e_nu_i ~ N(0,1),  c_nu_c ~ N(0, t_nu^2)
z [P, K] holds K parent-level covariates, assumed standardized (mean 0, SD 1;
check_standardized warns otherwise); a 1-D z is one covariate.  b_p and b_nu
are K-vectors, each coefficient with its own N(0, 1) prior.
The parent and chromosome effects are crossed and additive on the link scale.
chrom_effects=False drops the c terms (the parent-only model);
parent_effects=False drops the parent terms s * e_i (and with them s_p, s_nu).

Every meiotic product (one chromosome in one individual; one record) has the
exact marginal likelihood from hsmodel.likelihood.record_loglik (pathway labels
summed out).  Records are grouped by crossover count
(likelihood.count_loglik_per_parent): no padded crossover slots, crossover-free
records reduce to a closed form, and 1 - G*(L) is computed once per (parent,
chromosome).

Minibatches are sets of B parents drawn on the host (hsmodel.batching.
ParentBatcher).  Parent-level terms are scaled by P / B, so gradients are
unbiased for the full dataset; the chromosome effects are informed by every
batch and are sampled in full at every step, like the globals.
"""
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .likelihood import count_loglik_per_parent

GLOBALS = ["a_p", "b_p", "log_s_p", "log_t_p", "a_nu", "b_nu", "log_s_nu", "log_t_nu"]
CHROM_SITES = ["c_p", "c_nu"]
# Normal(loc, scale) priors on the globals, on the scale they are sampled.
# Used by the model, the guide's initialization and the plots, so the plots
# always show the priors fitted.
PRIORS = {
	"a_p": (-3.0, 1.5),
	"b_p": (0.0, 1.0),
	"log_s_p": (-1.0, 1.0),
	"log_t_p": (-1.5, 1.0),
	"a_nu": (float(np.log(9.0)), 1.0),
	"b_nu": (0.0, 1.0),
	"log_s_nu": (-1.0, 1.0),
	"log_t_nu": (-1.5, 1.0),
}
LABELS = {
	"a_p": "logit escape fraction at z = 0",
	"b_p": "effect of covariate {cov} on logit p",
	"log_s_p": "log between-parent SD of logit p",
	"log_t_p": "log between-chromosome SD of logit p",
	"a_nu": "log(nu - 1) at z = 0",
	"b_nu": "effect of covariate {cov} on log(nu - 1)",
	"log_s_nu": "log between-parent SD of log(nu - 1)",
	"log_t_nu": "log between-chromosome SD of log(nu - 1)",
}


COEFS = ("b_p", "b_nu")          # covariate coefficients: one per column of z


def globals_used(chrom_effects=True, parent_effects=True):
	"""The global parameters of the model with or without chromosome effects
	(log_t_p, log_t_nu) and parent effects (log_s_p, log_s_nu)."""
	return [g for g in GLOBALS if (chrom_effects or not g.startswith("log_t"))
	        and (parent_effects or not g.startswith("log_s"))]


def as_covariate_matrix(z):
	"""z as a [P, K] array (a 1-D z is one covariate)."""
	z = np.asarray(z, dtype=float)
	return z[:, None] if z.ndim == 1 else z


def default_covariate_names(K):
	return ["z"] if K == 1 else [f"z{k + 1}" for k in range(K)]


def check_standardized(z, covariate_names=None, tol_mean=0.15, tol_sd=0.25):
	"""Warn about covariates that do not look standardized (|mean| > tol_mean or
	|SD - 1| > tol_sd); the priors on b_p and b_nu assume standardized z."""
	z = as_covariate_matrix(z)
	names = covariate_names or default_covariate_names(z.shape[1])
	bad = [f"{n} (mean {m:.2f}, SD {sd:.2f})" for n, m, sd in zip(names, z.mean(0), z.std(0))
	       if abs(m) > tol_mean or abs(sd - 1) > tol_sd]
	if bad:
		warnings.warn("covariates do not look standardized: " + "; ".join(bad) +
		              ". The N(0, 1) priors on the coefficients assume mean 0 and SD 1.", stacklevel=2)
	return not bad


def global_entries(names, K, covariate_names=None):
	"""Scalar entries for display: [(key, site, k, label)], k = None for scalar
	sites and the coefficient index for b_p / b_nu (key e.g. "b_p[age]")."""
	cov = covariate_names or default_covariate_names(K)
	out = []
	for n in names:
		if n in COEFS:
			for k in range(K):
				out.append((f"{n}[{cov[k]}]", n, k, LABELS[n].format(cov=cov[k])))
		else:
			out.append((n, n, None, LABELS[n]))
	return out


def entry_value(value, k):
	"""float of a site's value (scalar or K-vector) for entry index k."""
	v = np.asarray(value, dtype=float)
	return float(v) if k is None or v.ndim == 0 else float(np.atleast_1d(v)[k])


def expand_truth(truth_sampled, K, covariate_names=None):
	"""{display key: value} from {site: scalar or K-vector} (e.g. TRUTH_SAMPLED)."""
	if truth_sampled is None:
		return None
	return {key: entry_value(truth_sampled[site], k) for key, site, k, _ in
	        global_entries([n for n in GLOBALS if n in truth_sampled], K, covariate_names)}


def sample_globals(chrom_effects, K, parent_effects=True):
	"""Sample the global sites; b_p and b_nu are K-vectors with independent priors."""
	g = {}
	for n in globals_used(chrom_effects, parent_effects):
		if n in COEFS:
			with numpyro.plate("covariates", K):
				g[n] = numpyro.sample(n, dist.Normal(*PRIORS[n]))
		else:
			g[n] = numpyro.sample(n, dist.Normal(*PRIORS[n]))
	return g


# ---------------------------------------------------------------------------
# Model and guides
# ---------------------------------------------------------------------------
def model(data, Lc, ix, z, n_parents, chrom_effects=True, parent_effects=True):
	"""data: one ParentBatcher batch; Lc [C] chromosome lengths; ix [B] global
	parent indices; z [P, K] covariates (1-D: one covariate).  Without parent
	effects the likelihood is still scaled by P / B per minibatch."""
	z = z[:, None] if z.ndim == 1 else z
	g = sample_globals(chrom_effects, z.shape[1], parent_effects)
	if chrom_effects:
		with numpyro.plate("chromosomes", Lc.shape[0]):
			c_p = numpyro.sample("c_p", dist.Normal(0.0, jnp.exp(g["log_t_p"])))
			c_nu = numpyro.sample("c_nu", dist.Normal(0.0, jnp.exp(g["log_t_nu"])))
	B = ix.shape[0]
	zb = z[ix]                                                                  # [B, K]
	with numpyro.handlers.scale(scale=n_parents / B), numpyro.plate("parents", B):
		lp = g["a_p"] + zb @ g["b_p"]                                           # [B]
		ln = g["a_nu"] + zb @ g["b_nu"]
		if parent_effects:
			e_p = numpyro.sample("e_p", dist.Normal(0.0, 1.0))
			e_nu = numpyro.sample("e_nu", dist.Normal(0.0, 1.0))
			lp = lp + jnp.exp(g["log_s_p"]) * e_p
			ln = ln + jnp.exp(g["log_s_nu"]) * e_nu
		if chrom_effects:                                                       # [B, C]
			p = jax.nn.sigmoid(lp[:, None] + c_p[None, :])
			nu = 1.0 + jnp.exp(ln[:, None] + c_nu[None, :])
		else:
			p, nu = jax.nn.sigmoid(lp), 1.0 + jnp.exp(ln)
		numpyro.factor("obs", count_loglik_per_parent(data, Lc, nu, p))


def guide(data, Lc, ix, z, n_parents, chrom_effects=True, parent_effects=True):
	K = 1 if z.ndim == 1 else z.shape[1]
	for name in globals_used(chrom_effects, parent_effects):
		if name in COEFS:
			loc = numpyro.param(f"{name}_loc", jnp.full(K, PRIORS[name][0]))
			scale = numpyro.param(f"{name}_scale", jnp.full(K, 0.1), constraint=dist.constraints.positive)
			with numpyro.plate("covariates", K):
				numpyro.sample(name, dist.Normal(loc, scale))
		else:
			loc = numpyro.param(f"{name}_loc", PRIORS[name][0])
			scale = numpyro.param(f"{name}_scale", 0.1, constraint=dist.constraints.positive)
			numpyro.sample(name, dist.Normal(loc, scale))
	if chrom_effects:
		C = Lc.shape[0]
		locs = {n: numpyro.param(f"{n}_loc", jnp.zeros(C)) for n in CHROM_SITES}
		scales = {n: numpyro.param(f"{n}_scale", jnp.full(C, 0.1), constraint=dist.constraints.positive)
		          for n in CHROM_SITES}
		with numpyro.plate("chromosomes", C):
			for n in CHROM_SITES:
				numpyro.sample(n, dist.Normal(locs[n], scales[n]))
	if not parent_effects:
		return
	ep_loc = numpyro.param("e_p_loc", jnp.zeros(n_parents))
	ep_sc = numpyro.param("e_p_scale", jnp.ones(n_parents), constraint=dist.constraints.positive)
	en_loc = numpyro.param("e_nu_loc", jnp.zeros(n_parents))
	en_sc = numpyro.param("e_nu_scale", jnp.ones(n_parents), constraint=dist.constraints.positive)
	B = ix.shape[0]
	with numpyro.handlers.scale(scale=n_parents / B), numpyro.plate("parents", B):
		numpyro.sample("e_p", dist.Normal(ep_loc[ix], ep_sc[ix]))
		numpyro.sample("e_nu", dist.Normal(en_loc[ix], en_sc[ix]))


def _global_layout(chrom_effects, K, parent_effects=True):
	"""[(site, slice into the global latent vector)] for guide_lowrank, and its length G."""
	out, start = [], 0
	for n in globals_used(chrom_effects, parent_effects):
		size = K if n in COEFS else 1
		out.append((n, slice(start, start + size)))
		start += size
	return out, start


def guide_lowrank(data, Lc, ix, z, n_parents, chrom_effects=True, rank=4, parent_effects=True):
	"""Structured guide that relaxes mean-field, compatible with minibatches of parents.

	Global block: theta = (globals incl. the K-vectors b_p, b_nu; c_p; c_nu),
	G + 2C dimensions, gets a low-rank-plus-diagonal Gaussian,
	theta ~ N(mu, diag(d) + W W^T),  W [D, rank].
	Parent block: (e_p_i, e_nu_i) | theta ~ N(m_i + B_i (g - mu_g), L_i L_i^T),
	where g are the G sampled globals, B_i [2, G] a per-parent coupling and
	L_i a 2x2 Cholesky factor (so e_p_i and e_nu_i may be correlated).

	The joint covariance is low-rank + block-diagonal, and the density factorizes
	as q(theta) prod_i q(e_i | theta), so subsampling parents stays exact: the
	parent block is scaled by P / B exactly as in the mean-field guide.  A full
	low-rank covariance across *all* parents would share factors between
	parents, and its density could not be evaluated one minibatch at a time.
	Without parent effects there is no parent block.

	The latent vectors are auxiliary sites; the model's sites are emitted as
	Deltas at their values (the same device numpyro's autoguides use)."""
	K = 1 if z.ndim == 1 else z.shape[1]
	layout, G = _global_layout(chrom_effects, K, parent_effects)
	C = Lc.shape[0] if chrom_effects else 0
	D = G + 2 * C
	mu0 = jnp.concatenate([jnp.full(sl.stop - sl.start, PRIORS[n][0]) for n, sl in layout] + [jnp.zeros(2 * C)])
	mu = numpyro.param("glob_loc", mu0)
	W = numpyro.param("glob_cov_factor", 0.01 * jax.random.normal(jax.random.PRNGKey(0), (D, rank)))
	d = numpyro.param("glob_cov_diag", jnp.full(D, 0.01), constraint=dist.constraints.positive)
	theta = numpyro.sample("_glob_latent", dist.LowRankMultivariateNormal(mu, W, d),
	                       infer={"is_auxiliary": True})
	for n, sl in layout:
		if n in COEFS:
			with numpyro.plate("covariates", K):
				numpyro.sample(n, dist.Delta(theta[sl]))
		else:
			numpyro.sample(n, dist.Delta(theta[sl.start]))
	if chrom_effects:
		with numpyro.plate("chromosomes", C):
			numpyro.sample("c_p", dist.Delta(theta[G:G + C]))
			numpyro.sample("c_nu", dist.Delta(theta[G + C:]))
	if not parent_effects:
		return
	m = numpyro.param("local_loc", jnp.zeros((n_parents, 2)))
	L = numpyro.param("local_scale_tril", jnp.tile(jnp.eye(2), (n_parents, 1, 1)),
	                  constraint=dist.constraints.lower_cholesky)
	Bc = numpyro.param("local_coupling", jnp.zeros((n_parents, 2, G)))
	dg = theta[:G] - mu[:G]                       # deviation of the sampled globals from their mean
	B = ix.shape[0]
	with numpyro.handlers.scale(scale=n_parents / B), numpyro.plate("parents", B):
		mean = m[ix] + jnp.einsum("bkg,g->bk", Bc[ix], dg)
		e = numpyro.sample("_local_latent", dist.MultivariateNormal(mean, scale_tril=L[ix]),
		                   infer={"is_auxiliary": True})
		numpyro.sample("e_p", dist.Delta(e[:, 0]))
		numpyro.sample("e_nu", dist.Delta(e[:, 1]))


def lowrank_marginals(params, chrom_effects=True, K=1, parent_effects=True):
	"""Marginal mean and SD of every site under guide_lowrank, in the mean-field
	naming ({site}_loc, {site}_scale; K-vectors for b_p and b_nu), so summarize,
	chrom_summary and the plots work unchanged.  Also
	returns the global block's correlation matrix as "glob_corr" (order:
	globals as in GLOBALS with b_p, b_nu expanded, then c_p, c_nu)."""
	layout, G = _global_layout(chrom_effects, K, parent_effects)
	mu = np.asarray(params["glob_loc"])
	W = np.asarray(params["glob_cov_factor"])
	Sig = np.diag(np.asarray(params["glob_cov_diag"])) + W @ W.T
	sd = np.sqrt(np.diag(Sig))
	out = {"glob_corr": Sig / np.outer(sd, sd)}
	for n, sl in layout:
		if n in COEFS:
			out[f"{n}_loc"], out[f"{n}_scale"] = mu[sl], sd[sl]
		else:
			out[f"{n}_loc"], out[f"{n}_scale"] = mu[sl.start], sd[sl.start]
	if chrom_effects:
		C = (len(mu) - G) // 2
		for j, n in enumerate(["c_p", "c_nu"]):
			sl = slice(G + j * C, G + (j + 1) * C)
			out[f"{n}_loc"], out[f"{n}_scale"] = mu[sl], sd[sl]
	if not parent_effects:
		return out
	# parents: Cov(e_i) = L_i L_i^T + B_i Sig_gg B_i^T
	L = np.asarray(params["local_scale_tril"])
	Bc = np.asarray(params["local_coupling"])
	cov = L @ np.swapaxes(L, 1, 2) + Bc @ Sig[:G, :G] @ np.swapaxes(Bc, 1, 2)
	loc = np.asarray(params["local_loc"])
	out["e_p_loc"], out["e_nu_loc"] = loc[:, 0], loc[:, 1]
	out["e_p_scale"], out["e_nu_scale"] = np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1])
	out["e_corr"] = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
	return out
