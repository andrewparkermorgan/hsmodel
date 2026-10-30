"""
Sketch: hierarchical Housworth-Stahl model with a polygenic (kinship-structured)
effect on interference.

Parent-level model (P parents):
    logit p_i      = a_p  + b_p  z_i + s_p  * e_p_i                     e_p ~ N(0, I)
    log(nu_i - 1)  = a_nu + b_nu z_i + u_i  [+ s_res * r_i]             r   ~ N(0, I)
    u ~ MVN(0, s_nu^2 K)       K: known P x P relationship matrix

K must be the additive genetic *relationship* matrix (diagonal 1 for
non-inbred individuals, 0.5 for parent-offspring and full sibs).  A matrix of
kinship *coefficients* (0.5 on the diagonal, 0.25 for sibs) is half of that,
so pass 2 * Phi.

Non-centered form used for sampling:  u = s_nu * F @ eta,  eta ~ N(0, I),
with F F^T = K precomputed once (kinship_factor).  The optional residual term
s_res * r keeps non-genetic between-parent variation from being forced into
s_nu; with it, s_nu^2 / (s_nu^2 + s_res^2) is the heritability of log(nu - 1).

The likelihood is unchanged: hsmodel.likelihood.count_loglik_per_parent(data, Lc, nu, p).
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hsmodel.likelihood import count_loglik_per_parent

PRIORS = {
	"a_p": (-3.0, 1.5), "b_p": (0.0, 1.0), "log_s_p": (-1.0, 1.0),
	"a_nu": (float(np.log(9.0)), 1.0), "b_nu": (0.0, 1.0),
	"log_s_nu": (-1.0, 1.0),        # genetic SD of log(nu - 1)
	"log_s_res": (-1.5, 1.0),       # residual (non-genetic) SD of log(nu - 1)
}


def kinship_factor(K, rel_tol=1e-8):
	"""F with F F^T = K, via the eigendecomposition, so a positive
	semi-definite K (e.g. identical twins, duplicated samples) works where a
	Cholesky factorization would fail.  Eigenvalues below rel_tol * max are
	dropped, so F may have fewer than P columns: F is [P, r]."""
	K = np.asarray(K, dtype=float)
	K = 0.5 * (K + K.T)
	lam, U = np.linalg.eigh(K)
	keep = lam > rel_tol * lam.max()
	return U[:, keep] * np.sqrt(lam[keep])


def _globals():
	g = {n: numpyro.sample(n, dist.Normal(*PRIORS[n])) for n in
	     ["a_p", "b_p", "log_s_p", "a_nu", "b_nu", "log_s_nu", "log_s_res"]}
	return g


def model_kinship(data, Lc, z, F, residual=True):
	"""Full-data model (NUTS).  data: hsmodel.likelihood.group_by_count layout, one index per
	parent; z [P]; F [P, r] from kinship_factor."""
	P, r = F.shape
	g = _globals()
	with numpyro.plate("parents", P):
		e_p = numpyro.sample("e_p", dist.Normal(0.0, 1.0))
		res = numpyro.sample("r", dist.Normal(0.0, 1.0)) if residual else jnp.zeros(P)
	with numpyro.plate("genetic_dims", r):
		eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
	u = jnp.exp(g["log_s_nu"]) * (F @ eta)                      # polygenic effects [P]
	p = jax.nn.sigmoid(g["a_p"] + g["b_p"] * z + jnp.exp(g["log_s_p"]) * e_p)
	nu = 1.0 + jnp.exp(g["a_nu"] + g["b_nu"] * z + u + jnp.exp(g["log_s_res"]) * res)
	numpyro.factor("obs", jnp.sum(count_loglik_per_parent(data, Lc, nu, p)))


def model_kinship_minibatch(data, Lc, ix, z, F, residual=True):
	"""SVI with minibatches of parents (data from hsmodel.batching.ParentBatcher, ix [B]).

	eta can no longer be subsampled with the parents: u_i = s_nu * F[i] @ eta
	depends on every eta_j.  So eta (and its prior) is always sampled in full,
	and only the likelihood is subsampled, scaled by P / B.  e_p and r are still
	independent per parent and are subsampled as before."""
	P, r = F.shape
	B = ix.shape[0]
	g = _globals()
	with numpyro.plate("genetic_dims", r):
		eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
	with numpyro.handlers.scale(scale=P / B), numpyro.plate("parents", B):
		e_p = numpyro.sample("e_p", dist.Normal(0.0, 1.0))
		res = numpyro.sample("r", dist.Normal(0.0, 1.0)) if residual else jnp.zeros(B)
		u = jnp.exp(g["log_s_nu"]) * (F[ix] @ eta)               # only the batch's rows of F
		p = jax.nn.sigmoid(g["a_p"] + g["b_p"] * z[ix] + jnp.exp(g["log_s_p"]) * e_p)
		nu = 1.0 + jnp.exp(g["a_nu"] + g["b_nu"] * z[ix] + u + jnp.exp(g["log_s_res"]) * res)
		numpyro.factor("obs", count_loglik_per_parent(data, Lc, nu, p))
	# Guide: a mean-field guide over eta ignores the posterior correlations that
	# relatives induce; numpyro.infer.autoguide.AutoLowRankMultivariateNormal (or
	# a hand-written guide with a low-rank-plus-diagonal covariance for eta) is
	# the minimum.  e_p and r would need a guide indexed by ix, as in hsmodel.model.guide.


def sibship_relationship_matrix(family_sizes):
	"""Toy K for testing: unrelated full-sib families (1 on the diagonal, 0.5
	between sibs), block diagonal."""
	P = int(np.sum(family_sizes))
	K = np.zeros((P, P))
	start = 0
	for n in family_sizes:
		K[start:start + n, start:start + n] = 0.5
		start += n
	np.fill_diagonal(K, 1.0)
	return K
