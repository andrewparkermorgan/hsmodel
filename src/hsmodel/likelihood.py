"""
Likelihood of the Housworth-Stahl two-pathway crossover interference model
(JAX).

Model (per gamete, per chromosome of genetic length L Morgans):
  * interfering chiasmata: stationary gamma renewal process,
      shape nu, rate 2*nu*(1-p)          (intensity 2(1-p) per Morgan)
  * escape chiasmata: Poisson process, intensity 2p per Morgan
  * each chiasma is seen in a given gamete with prob 1/2 (no chromatid interference)

The pathway label of every observed crossover is a discrete latent variable.
It is NOT given a variational distribution: it is summed out exactly with an
O(n^2) forward recursion over "index of the last interfering crossover", which
is possible because the Poisson pathway is memoryless.  That keeps the ELBO
gradients low-variance; VI is only used for the continuous parameters.

Notation (Housworth & Stahl 2003), all for the gamete-level interfering pathway:
  f*(d)  density of the distance between consecutive interfering crossovers
  F*(d)  cdf of f*
  g*(x)  density of the distance from the chromosome end to the first
         interfering crossover;  g*(x) = (1-p) [1 - F*(x)]  (stationarity)
  G*(x)  cdf of g*;  1 - G*(L) = P(no interfering crossover on [0, L])
Functions return logs, and the cdfs are returned as log(1 - F*) and
log(1 - G*) because those are the forms the likelihood uses.

Data layout (all arrays padded):
  X    [..., N]   sorted crossover positions in Morgans (pad with 0)
  NXO  [...]      number of real crossovers
  LEN  [...]      chromosome genetic length in Morgans
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln, gammaincc, gammainc, logsumexp

jax.config.update("jax_enable_x64", True)

K_TERMS = 20          # truncation of the thinning mixture; terms k>20 have weight <1e-6
					  # and mean gap >= 10 Morgans, so they add ~nothing on L <= ~5 M
_EPS = 1e-9
# Log-weight for masked graph edges and padded positions.  Finite on purpose:
# a logsumexp over all -inf inputs has value -inf but gradient inf*0 = nan,
# which poisons every parameter.  exp(_NEG - finite) is exactly 0 in float64,
# so masked terms still contribute nothing to the value or the gradient.
_NEG = -1e30
_LOG2 = jnp.log(2.0)


# ---------------------------------------------------------------------------
# Building blocks for the thinned (gamete-level) interfering process
# ---------------------------------------------------------------------------
def _mix(nu, p, K=None):
	# K=None everywhere means "use the current K_TERMS" (see set_quadrature)
	K = K_TERMS if K is None else K
	k = jnp.arange(1, K + 1, dtype=jnp.float64)
	return k, k * nu, 2.0 * nu * (1.0 - p)          # k, shapes a_k, rate r


def log_f_star(d, nu, p, K=None):
	"""log f*(d): density of the distance between consecutive interfering
	crossovers in one gamete,  f*(d) = sum_k 2^-k Gamma(d; k*nu, 2nu(1-p))."""
	k, a, r = _mix(nu, p, K)
	d = jnp.clip(d, _EPS)[..., None]
	lp = -k * _LOG2 + a * jnp.log(r) + (a - 1.0) * jnp.log(d) - r * d - gammaln(a)
	return logsumexp(lp, axis=-1)


# Gauss-Legendre nodes on (0, 1).  F* and G* are computed by quadrature of f*
# rather than with gammaincc, because JAX's gradient of the incomplete gamma
# function w.r.t. its shape argument is ~7x slower on CPU.  examples/approx_accuracy.py
# measures per-record log-likelihood error vs. a K=200 closed-form reference:
# <= 1.3e-12 for nu in [2, 64], p in [0.001, 0.2], L <= 2 Morgans, at any
# K >= 20 (the error grows with nu and L; re-check before going far outside
# that range).  Quadrature keeps the 2^-K tail mass as survival, so unlike the
# closed forms it does not need a large K.
N_QUAD = 64
_t, _w = __import__("numpy").polynomial.legendre.leggauss(N_QUAD)
_QT, _QW = jnp.asarray((_t + 1) / 2), jnp.asarray(_w / 2)


def set_quadrature(n_nodes=64, k_terms=None):
	"""Change the number of Gauss-Legendre nodes (and optionally K_TERMS) used
	by every likelihood function.  Call it before anything is jit-compiled:
	already-compiled functions keep the old values.  examples/quadrature_accuracy.py
	measures the error for each setting."""
	global N_QUAD, _QT, _QW, K_TERMS
	import numpy as _np
	t, w = _np.polynomial.legendre.leggauss(int(n_nodes))
	N_QUAD, _QT, _QW = int(n_nodes), jnp.asarray((t + 1) / 2), jnp.asarray(w / 2)
	if k_terms is not None:
		K_TERMS = int(k_terms)


def log1m_F_star(d, nu, p, K=None):
	"""log(1 - F*(d)),  F*(d) = int_0^d f*."""
	d = jnp.clip(d, 0.0)
	xs = d[..., None] * _QT
	F = d * jnp.sum(_QW * jnp.exp(log_f_star(xs, nu, p, K)), axis=-1)
	return jnp.log(jnp.clip(1.0 - F, 1e-300))


def log_g_star(x, nu, p, K=None):
	"""log g*(x): density of the distance from the chromosome end to the first
	interfering crossover,  g*(x) = (1-p) [1 - F*(x)]."""
	return jnp.log1p(-p) + log1m_F_star(x, nu, p, K)


def log1m_G_star(L, nu, p, K=None):
	"""log(1 - G*(L)) = log P(no interfering crossover on [0, L]),
	with G*(L) = (1-p) int_0^L (1 - F*(x)) dx
	           = (1-p) [L - int_0^L (L - x) f*(x) dx]."""
	xs = L[..., None] * _QT
	int_1mF = L - L * jnp.sum(_QW * (L[..., None] - xs) * jnp.exp(log_f_star(xs, nu, p, K)), axis=-1)
	return jnp.log(jnp.clip(1.0 - (1.0 - p) * int_1mF, 1e-300))


# Closed-form reference versions (exact; slow gradients in nu). Used in tests.
def log1m_F_star_exact(d, nu, p, K=None):
	k, a, r = _mix(nu, p, K)
	d = jnp.clip(d, 0.0)[..., None]
	q = jnp.clip(gammaincc(a, r * d), 1e-300)
	return logsumexp(-k * _LOG2 + jnp.log(q), axis=-1)


def log1m_G_star_exact(L, nu, p, K=None):
	k, a, r = _mix(nu, p, K)
	L_ = L[..., None]
	# int_0^L Q(a, r x) dx = L Q(a, rL) + (a/r) P(a+1, rL);  (1-p) a/r = k/2
	term = (1.0 - p) * L_ * gammaincc(a, r * L_) + 0.5 * k * gammainc(a + 1.0, r * L_)
	inside = 1.0 - jnp.sum(jnp.exp(-k * _LOG2) * term, axis=-1)
	return jnp.log(jnp.clip(inside, 1e-300))


# ---------------------------------------------------------------------------
# Labellings as paths through a graph
# ---------------------------------------------------------------------------
# Nodes: START, one node per crossover (0-based i = 0..n-1), END.  A labelling
# with interfering set {s_1 < ... < s_k} is the path START -> s_1 -> ... -> s_k
# -> END (k = 0: START -> END).  Escape crossovers are skipped over by an edge
# and each contributes a factor p.  Edge log-weights:
#   START -> i      i*logp           + log g*(x_i)
#   j -> i  (j<i)   (i-j-1)*logp     + log f*(x_i - x_j)
#   i -> END        (n-1-i)*logp     + log(1 - F*(L - x_i))
#   START -> END    n*logp           + log(1 - G*(L))
# Likelihood = exp(-pL) * (sum over all START -> END paths of path weight).
def path_log_weights(x, n, L, nu, p, K=None, exact=False, log1m_G=None):
	"""Edge log-weights of the labelling graph for one record.
	log1m_G: precomputed log(1 - G*(L)) for this (L, nu, p); computed here if None.
	Returns (w_start [N], w_edge [N, N] with w_edge[i, j] = edge j -> i,
	w_end [N], w_none scalar).  Edges touching padded positions get log-weight _NEG.
	K: terms kept in the thinning mixture.  exact=True uses the closed-form
	(incomplete gamma) F* and G* instead of quadrature (slow gradients in nu;
	for testing).  K and exact must be static under jit."""
	if log1m_G is None:
		log1m_G = (log1m_G_star_exact if exact else log1m_G_star)(L, nu, p, K)
	l1mF = log1m_F_star_exact if exact else log1m_F_star
	N = x.shape[0]
	idx = jnp.arange(N)
	logp = jnp.log(p)
	valid = idx < n
	lg_star = jnp.log1p(-p) + l1mF(x, nu, p, K)               # g* = (1-p)(1-F*)
	w_start = jnp.where(valid, idx * logp + lg_star, _NEG)
	w_edge = (idx[:, None] - idx[None, :] - 1) * logp + log_f_star(x[:, None] - x[None, :], nu, p, K)
	allowed = (idx[None, :] < idx[:, None]) & valid[:, None] & valid[None, :]
	w_edge = jnp.where(allowed, w_edge, _NEG)
	w_end = jnp.where(valid, (n - 1 - idx) * logp + l1mF(L - x, nu, p, K), _NEG)
	w_none = n * logp + log1m_G
	return w_start, w_edge, w_end, w_none


def _forward(w_start, w_edge):
	"""log b_i = log total weight of START -> i paths
	(all labellings of crossovers 0..i in which crossover i is interfering).
	b_i = w_start[i] + sum_{j<i} b_j w_edge[i, j]."""
	N = w_start.shape[0]

	def step(b, i):
		bi = logsumexp(jnp.concatenate([w_start[i][None], b + w_edge[i]]))
		return b.at[i].set(bi), None

	b, _ = lax.scan(step, jnp.full(N, _NEG), jnp.arange(N))
	return b


def _backward(w_edge, w_end):
	"""log c_i = log total weight of i -> END paths
	(all labellings of crossovers i+1..n-1, given crossover i is interfering).
	c_i = w_end[i] + sum_{k>i} w_edge[k, i] c_k."""
	N = w_end.shape[0]

	def step(c, i):
		ci = logsumexp(jnp.concatenate([w_end[i][None], c + w_edge[:, i]]))
		return c.at[i].set(ci), None

	c, _ = lax.scan(step, jnp.full(N, _NEG), jnp.arange(N)[::-1])
	return c


# ---------------------------------------------------------------------------
# Exact marginal likelihood of one gamete-chromosome record
# ---------------------------------------------------------------------------
def record_loglik(x, n, L, nu, p, K=None, exact=False, log1m_G=None):
	"""log P(crossovers at x[:n] on [0, L] | nu, p), pathway labels summed out
	(forward pass only).  K, exact and log1m_G as in path_log_weights."""
	w_start, w_edge, w_end, w_none = path_log_weights(x, n, L, nu, p, K, exact, log1m_G)
	b = _forward(w_start, w_edge)
	return -p * L + logsumexp(jnp.concatenate([w_none[None], b + w_end]))


batch_loglik = jax.vmap(record_loglik, in_axes=(0, 0, 0, None, None))
batch_loglik_per = jax.vmap(record_loglik, in_axes=(0, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Likelihood grouped by chromosome (G* once per chromosome)
# ---------------------------------------------------------------------------
# 1 - G*(L) depends only on (L, nu, p), so every record on the same chromosome
# that shares (nu, p) uses the same value.  Data are arranged as
#   Xg [C, R, N], Ng [C, R], Mg [C, R]  (R = records per chromosome, padded;
#   Mg = 1 for real records, 0 for padding),  Lc [C] chromosome lengths,
# built with group_by_chromosome().  With a parent axis in front
# ([P, C, R, N]) and per-parent (nu, p), G* is computed once per
# (parent, chromosome).
def chromosome_loglik(X, NXO, M, L, nu, p, K=None, exact=False):
	"""Summed log-likelihood of the records X [R, N] on one chromosome of
	length L that share (nu, p).  G* is evaluated once."""
	log1m_G = (log1m_G_star_exact if exact else log1m_G_star)(L, nu, p, K)
	ll = jax.vmap(lambda x, n: record_loglik(x, n, L, nu, p, K, exact, log1m_G))(X, NXO)
	return jnp.sum(jnp.where(M > 0, ll, 0.0))


def grouped_loglik(Xg, Ng, Mg, Lc, nu, p, K=None, exact=False):
	"""Total log-likelihood of chromosome-grouped data (Xg [C, R, N]) under a
	single (nu, p): per-chromosome sums, aggregated across chromosomes."""
	per_chrom = jax.vmap(lambda X, n, m, L: chromosome_loglik(X, n, m, L, nu, p, K, exact))(
		Xg, Ng, Mg, Lc)
	return jnp.sum(per_chrom)


# One (nu, p) per parent: Xg [P, C, R, N], Ng/Mg [P, C, R], Lc [C], nu/p [P] -> [P]
parent_grouped_loglik = jax.vmap(grouped_loglik, in_axes=(0, 0, 0, None, 0, 0))


# ---------------------------------------------------------------------------
# Chromosome-specific padding width
# ---------------------------------------------------------------------------
# The cost of a record grows with the padded width N (1 - F* quadrature at 2N
# points, f* on N x N), so padding every chromosome to the dataset-wide maximum
# wastes most of the work on short chromosomes.  group_by_chromosome_width()
# pads each chromosome only to the largest crossover count observed on it, and
# collects chromosomes of equal width into one "bucket" (Xg, Ng, Mg, Lc) with
# the same layout as group_by_chromosome().  The likelihood is a Python sum over
# buckets, so under jit each distinct width is compiled once.
def bucketed_loglik(buckets, nu, p, K=None, exact=False):
	"""Total log-likelihood under a single (nu, p); buckets from
	group_by_chromosome_width() without parent_idx."""
	return sum(grouped_loglik(X, n, m, L, nu, p, K, exact) for X, n, m, L in buckets)


def parent_bucketed_loglik(buckets, nu, p):
	"""Per-parent log-likelihoods [P]; buckets from group_by_chromosome_width()
	with parent_idx, nu and p of shape [P]."""
	return sum(parent_grouped_loglik(X, n, m, L, nu, p) for X, n, m, L in buckets)


# ---------------------------------------------------------------------------
# Records grouped by crossover count (no padding)
# ---------------------------------------------------------------------------
# Records with n crossovers are stored in an array exactly n wide, so no work is
# spent on padded crossover slots.  Records with no crossovers need no forward
# pass: their likelihood is exp(-pL) (1 - G*(L)), so only their number per
# (parent, chromosome) is kept.  1 - G*(L) is computed once per (parent,
# chromosome) as a table and looked up by each record.
#
# Data (a pytree, built by group_by_count()):
#   {"count0": [P, C] number of crossover-free records per (parent, chromosome),
#    "groups": tuple over n = 1, 2, ... of (X [R_n, n], parent [R_n] int,
#              chrom [R_n] int, mask [R_n])}     mask = 0 marks dummy records
# with chromosome lengths Lc [C] passed separately.
def log1m_G_table(Lc, nu, p, K=None, exact=False):
	"""log(1 - G*(L)) for every parent and chromosome -> [P, C].
	Lc: [C] (one length per chromosome) or [P, C] (per-parent lengths, e.g. by
	sex).  nu, p: [P] (one value per parent) or [P, C] (per parent and chromosome)."""
	f = log1m_G_star_exact if exact else log1m_G_star
	per_parent_len = jnp.ndim(Lc) == 2
	if jnp.ndim(nu) == 1:
		if per_parent_len:
			return jax.vmap(lambda Lrow, a, b: f(Lrow, a, b, K))(Lc, nu, p)
		return jax.vmap(lambda a, b: f(Lc, a, b, K))(nu, p)
	cell = jax.vmap(lambda L, a, b: f(L, a, b, K), in_axes=(0, 0, 0))
	return jax.vmap(cell, in_axes=(0 if per_parent_len else None, 0, 0))(Lc, nu, p)


def count_loglik_per_parent(data, Lc, nu, p, K=None, exact=False):
	"""Per-parent log-likelihoods [P] for count-grouped data.

	nu, p: [P] for one value per parent, or [P, C] for a separate value for
	each parent and chromosome (e.g. parent plus chromosome random effects).
	Chromosome lengths: data["L"] [P, C] if present (per-parent lengths, e.g.
	looked up by sex with parent_chrom_lengths), otherwise Lc [C], the same
	for every parent.  G* is computed once per (parent, chromosome) either way."""
	count0 = data["count0"]
	P = count0.shape[0]
	per_chrom = jnp.ndim(nu) == 2
	L = data["L"] if "L" in data else Lc                       # [P, C] or [C]
	L_pc = L if jnp.ndim(L) == 2 else L[None, :]
	log1mG = log1m_G_table(L, nu, p, K, exact)
	p_pc = p if per_chrom else p[:, None]
	ll = jnp.sum(count0 * (log1mG - p_pc * L_pc), axis=1)
	per_parent_len = jnp.ndim(L) == 2
	for X, par, ch, m in data["groups"]:
		n = X.shape[1]

		def f(x, i, c):
			Lic = L[i, c] if per_parent_len else L[c]
			nu_ic = nu[i, c] if per_chrom else nu[i]
			p_ic = p[i, c] if per_chrom else p[i]
			return record_loglik(x, n, Lic, nu_ic, p_ic, K, exact, log1mG[i, c])

		lln = jax.vmap(f)(X, par, ch)
		ll = ll + jax.ops.segment_sum(jnp.where(m > 0, lln, 0.0), par, num_segments=P)
	return ll


def count_loglik(data, Lc, nu, p, K=None, exact=False):
	"""Total log-likelihood under a single (nu, p).  Build the data without
	parent_idx so that G* is computed once per chromosome."""
	P = data["count0"].shape[0]
	return jnp.sum(count_loglik_per_parent(data, Lc, jnp.full(P, nu), jnp.full(P, p), K, exact))


# ---------------------------------------------------------------------------
# Forward-backward: per-crossover pathway posteriors
# ---------------------------------------------------------------------------
def forward_backward(x, n, L, nu, p):
	"""Returns (log_b [N], log_c [N], loglik).  Padded positions are <= _NEG."""
	w_start, w_edge, w_end, w_none = path_log_weights(x, n, L, nu, p)
	b = _forward(w_start, w_edge)
	c = _backward(w_edge, w_end)
	loglik = -p * L + logsumexp(jnp.concatenate([w_none[None], b + w_end]))
	return b, c, loglik


def record_escape_posterior(x, n, L, nu, p):
	"""P(crossover i is non-interfering (escape) | this record's crossovers, nu, p)
	for i < n; NaN at padded positions.

	A labelling has crossover i interfering iff its path passes through node i,
	so  P(i interfering) = b_i c_i / Z  with  Z = exp(pL) * likelihood."""
	b, c, loglik = forward_backward(x, n, L, nu, p)
	log_post_int = b + c - (loglik + p * L)
	p_escape = -jnp.expm1(log_post_int)        # 1 - P(interfering), accurate near 0
	return jnp.where(jnp.arange(x.shape[0]) < n, p_escape, jnp.nan)


batch_escape_posterior = jax.vmap(record_escape_posterior, in_axes=(0, 0, 0, None, None))
batch_escape_posterior_per = jax.vmap(record_escape_posterior, in_axes=(0, 0, 0, 0, 0))
_batch_escape_posterior_per_jit = jax.jit(batch_escape_posterior_per)


def escape_posterior(X, NXO, LEN, nu, p, chunk=2000):
	"""P(escape) for every crossover in a padded dataset, processed in chunks.

	X [R, N], NXO [R], LEN [R] as from hsmodel.data.pad_records.  nu and p are scalars or
	per-record arrays of length R.  Returns a NumPy array [R, N], NaN at padding.
	Peak memory grows like chunk * N * N_QUAD * K_TERMS doubles (about 160 MB
	per 1,000 records at N = 8), so keep chunk at a few thousand."""
	import numpy as np
	R = X.shape[0]
	nu = np.broadcast_to(np.asarray(nu, dtype=float), (R,))
	p = np.broadcast_to(np.asarray(p, dtype=float), (R,))
	out = np.empty(X.shape)
	for s in range(0, R, chunk):
		e = min(s + chunk, R)
		pad = chunk - (e - s) if R > chunk else 0          # fixed shape -> one compile
		sl = lambda a, fill: np.concatenate([a[s:e], np.full((pad,) + a.shape[1:], fill, a.dtype)])
		res = _batch_escape_posterior_per_jit(
			jnp.asarray(sl(X, 0.0)), jnp.asarray(sl(NXO, 0)), jnp.asarray(sl(LEN, 1.0)),
			jnp.asarray(sl(nu, 2.0)), jnp.asarray(sl(p, 0.1)))
		out[s:e] = np.asarray(res)[: e - s]
	return out


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def group_by_parent(X, NXO, LEN, parent_idx, n_parents):
	"""Pad records into [parent, record, xo] blocks so SVI can subsample parents."""
	import numpy as np
	counts = np.bincount(parent_idx, minlength=n_parents)
	R = counts.max()
	Xp = np.zeros((n_parents, R, X.shape[1]))
	Np = np.zeros((n_parents, R), dtype=np.int32)
	Lp = np.ones((n_parents, R))
	Mp = np.zeros((n_parents, R))
	slot = np.zeros(n_parents, dtype=int)
	for r, i in enumerate(parent_idx):
		s = slot[i]; slot[i] += 1
		Xp[i, s], Np[i, s], Lp[i, s], Mp[i, s] = X[r], NXO[r], LEN[r], 1.0
	return Xp, Np, Lp, Mp


def group_by_chromosome(X, NXO, chrom_idx, chrom_len, parent_idx=None, n_parents=None):
	"""Arrange padded records (from hsmodel.data.pad_records) by chromosome.

	chrom_idx [R] gives each record's chromosome (0..C-1); chrom_len [C] the
	chromosome lengths in Morgans.  Returns (Xg, Ng, Mg, Lc):
	  without parent_idx:  Xg [C, Rmax, N], Ng/Mg [C, Rmax]
	  with parent_idx:     Xg [P, C, Rmax, N], Ng/Mg [P, C, Rmax]
	Rmax is the largest number of records in any group; Mg marks real records."""
	import numpy as np
	chrom_idx = np.asarray(chrom_idx)
	C = len(chrom_len)
	if parent_idx is None:
		parent_idx = np.zeros(len(chrom_idx), dtype=int)
		n_parents, squeeze = 1, True
	else:
		parent_idx = np.asarray(parent_idx)
		n_parents = n_parents or int(parent_idx.max()) + 1
		squeeze = False
	counts = np.zeros((n_parents, C), dtype=int)
	np.add.at(counts, (parent_idx, chrom_idx), 1)
	Rmax = max(1, counts.max())
	Xg = np.zeros((n_parents, C, Rmax, X.shape[1]))
	Ng = np.zeros((n_parents, C, Rmax), dtype=np.int32)
	Mg = np.zeros((n_parents, C, Rmax))
	slot = np.zeros((n_parents, C), dtype=int)
	for r, (i, c) in enumerate(zip(parent_idx, chrom_idx)):
		s = slot[i, c]; slot[i, c] += 1
		Xg[i, c, s], Ng[i, c, s], Mg[i, c, s] = X[r], NXO[r], 1.0
	Lc = np.asarray(chrom_len, dtype=float)
	if squeeze:
		return Xg[0], Ng[0], Mg[0], Lc
	return Xg, Ng, Mg, Lc


def group_by_chromosome_width(X, NXO, chrom_idx, chrom_len, parent_idx=None, n_parents=None):
	"""Like group_by_chromosome(), but each chromosome is padded only to the
	largest number of crossovers observed on it (minimum 1).

	Returns (buckets, chroms): buckets is a list of (Xg, Ng, Mg, Lc) tuples, one
	per distinct width, in the group_by_chromosome() layout; chroms[b] holds the
	original chromosome indices in bucket b, in the order of its Lc."""
	import numpy as np
	X, NXO = np.asarray(X), np.asarray(NXO)
	chrom_idx = np.asarray(chrom_idx)
	chrom_len = np.asarray(chrom_len, dtype=float)
	C = len(chrom_len)
	if parent_idx is not None:
		parent_idx = np.asarray(parent_idx)
		n_parents = n_parents or int(parent_idx.max()) + 1
	width = np.ones(C, dtype=int)
	np.maximum.at(width, chrom_idx, NXO)
	buckets, chroms = [], []
	for w in np.unique(width):
		cs = np.flatnonzero(width == w)
		remap = np.full(C, -1)
		remap[cs] = np.arange(len(cs))
		sel = remap[chrom_idx] >= 0
		g = group_by_chromosome(X[sel, :w], NXO[sel], remap[chrom_idx[sel]], chrom_len[cs],
		                        None if parent_idx is None else parent_idx[sel], n_parents)
		buckets.append(g)
		chroms.append(cs)
	return buckets, chroms


def parent_chrom_lengths(chrom_len_by_sex, sex, fill=1.0):
	"""Per-parent chromosome lengths [P, C] from sex-specific maps.

	chrom_len_by_sex: [2, C] (row 0 male, row 1 female, as written by
	hsmodel preprocess); sex: [P] codes 1 (male) or 2 (female).  Entries with no
	map (NaN) are set to `fill`: such (parent, chromosome) cells must have no
	records, and a finite placeholder keeps the likelihood and its gradient
	finite (their count of crossover-free records is 0)."""
	import numpy as np
	L = np.asarray(chrom_len_by_sex, dtype=float)[np.asarray(sex) - 1]
	return np.where(np.isfinite(L), L, fill)


def group_by_count(X, NXO, chrom_idx, n_chrom, parent_idx=None, n_parents=None, parent_lengths=None):
	"""Group padded records (from hsmodel.data.pad_records) by crossover count; see the
	layout above.  Without parent_idx all records belong to one parent.
	parent_lengths: optional [P, C] per-parent chromosome lengths (e.g. from
	parent_chrom_lengths); stored as data["L"] and used by the likelihood
	instead of one length per chromosome."""
	import numpy as np
	X, NXO, chrom_idx = np.asarray(X), np.asarray(NXO), np.asarray(chrom_idx)
	if parent_idx is None:
		par, P = np.zeros(len(NXO), dtype=int), 1
	else:
		par = np.asarray(parent_idx)
		P = n_parents or int(par.max()) + 1
	count0 = np.zeros((P, n_chrom))
	z = NXO == 0
	np.add.at(count0, (par[z], chrom_idx[z]), 1.0)
	groups = []
	for n in range(1, int(NXO.max()) + 1 if len(NXO) else 1):
		sel = NXO == n
		if sel.any():
			groups.append((X[sel, :n].copy(), par[sel].astype(np.int32),
			               chrom_idx[sel].astype(np.int32), np.ones(int(sel.sum()))))
	out = {"count0": count0, "groups": tuple(groups)}
	if parent_lengths is not None:
		L = np.asarray(parent_lengths, dtype=float)
		if L.shape != (P, n_chrom):
			raise ValueError(f"parent_lengths has shape {L.shape}, expected {(P, n_chrom)}")
		if not np.all(np.isfinite(L)):
			raise ValueError("parent_lengths must be finite (see parent_chrom_lengths)")
		out["L"] = L
	return out
