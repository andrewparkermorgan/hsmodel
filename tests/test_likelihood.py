"""The record likelihood: agreement with simulation, quadrature vs. closed
forms, and the forward-backward escape posteriors (brute force over all 2^n
labellings, forward/backward totals, calibration against true labels)."""
import itertools

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from hsmodel.data import pad_records
from hsmodel.likelihood import (record_loglik, record_escape_posterior, escape_posterior, forward_backward,
                                path_log_weights, log_g_star, log_f_star, log1m_F_star, log1m_F_star_exact,
                                log1m_G_star, log1m_G_star_exact)
from hsmodel.simulate import simulate_product

# compiled once; n, L, nu, p are traced so every record reuses the same kernel
record_loglik_jit = jax.jit(record_loglik)
record_escape_posterior_jit = jax.jit(record_escape_posterior)
forward_backward_jit = jax.jit(forward_backward)
path_log_weights_jit = jax.jit(path_log_weights)


# ---------------------------------------------------------------------------
# Likelihood vs. simulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu, p, L", [(10.0, 0.1, 1.5), (4.0, 0.3, 0.8), (15.0, 0.05, 2.5)])
def test_count_probabilities_match_simulation(nu, p, L):
	"""P(0, 1, 2 crossovers), integrating the likelihood over positions, equals
	the simulated frequencies of 40,000 products."""
	rng = np.random.default_rng(0)
	n_sim = 40000
	counts = np.bincount([len(simulate_product(nu, p, L, rng)) for _ in range(n_sim)], minlength=5) / n_sim
	f = jax.jit(lambda x, n: record_loglik(x, n, jnp.array(L), nu, p))
	P0 = float(jnp.exp(f(jnp.zeros(4), 0)))
	g = np.linspace(0, L, 801)[1:-1]
	P1 = float(np.sum(jax.vmap(lambda a: jnp.exp(f(jnp.array([a, 0.0, 0.0, 0.0]), 1)))(g)) * (g[1] - g[0]))
	G = 200
	gg = (np.arange(G) + 0.5) * L / G
	A, B = np.meshgrid(gg, gg, indexing="ij")
	m = A < B
	pts = np.stack([A[m], B[m], 0 * A[m], 0 * A[m]], 1)
	P2 = float(np.sum(jax.vmap(lambda v: jnp.exp(f(v, 2)))(pts)) * (L / G) ** 2)
	for k, lik in enumerate([P0, P1, P2]):
		se = np.sqrt(lik * (1 - lik) / n_sim)
		assert abs(counts[k] - lik) < 4 * se + 2e-3, (k, counts[k], lik)


def test_gradient_finite():
	gr = jax.grad(lambda t: record_loglik(jnp.array([0.2, 0.21, 0.9, 0.0]), 3, jnp.array(1.2),
	                                      1 + jnp.exp(t[0]), jax.nn.sigmoid(t[1])))(jnp.array([2.0, -2.0]))
	assert np.all(np.isfinite(np.asarray(gr)))


@pytest.mark.parametrize("nu, p", [(2.0, 0.3), (11.0, 0.06), (30.0, 0.02)])
def test_quadrature_matches_closed_form(nu, p):
	"""Quadrature 1 - F*, 1 - G* vs. the incomplete-gamma closed forms.  The
	closed forms get K = 60 terms: truncating their sum at K drops ~2^-K of
	survival mass, whereas the quadrature (1 - int f*) keeps it."""
	d = jnp.array([0.05, 0.3, 0.6, 1.2, 2.8])
	Ls = jnp.array([0.6, 1.5, 2.8])
	eF = float(jnp.max(jnp.abs(jnp.exp(log1m_F_star(d, nu, p)) - jnp.exp(log1m_F_star_exact(d, nu, p, K=60)))))
	eG = float(jnp.max(jnp.abs(jnp.exp(log1m_G_star(Ls, nu, p)) - jnp.exp(log1m_G_star_exact(Ls, nu, p, K=60)))))
	assert eF < 1e-12 and eG < 1e-12


# ---------------------------------------------------------------------------
# Forward-backward escape posteriors
# ---------------------------------------------------------------------------
def brute_force(x, L, nu, p):
	"""Enumerate every labelling directly from the Housworth-Stahl formula.
	Returns (loglik, P(escape) per crossover)."""
	n = len(x)
	xs = jnp.asarray(x)
	lg = np.asarray(log_g_star(xs, nu, p))
	lf = np.asarray(log_f_star(xs[:, None] - xs[None, :], nu, p))
	l1F = np.asarray(log1m_F_star(L - xs, nu, p))
	l1G = float(log1m_G_star(jnp.asarray(L), nu, p))
	logw, labels = [], []
	for S in itertools.product([0, 1], repeat=n):          # 1 = interfering
		s = [i for i in range(n) if S[i]]
		w = (n - len(s)) * np.log(p) - p * L                # escape (Poisson) part
		if s:
			w += lg[s[0]] + sum(lf[s[j], s[j - 1]] for j in range(1, len(s))) + l1F[s[-1]]
		else:
			w += l1G
		logw.append(w); labels.append(S)
	logw, labels = np.array(logw), np.array(labels)
	logZ = np.logaddexp.reduce(logw)
	post = np.exp(logw - logZ)
	return logZ, 1.0 - (post[:, None] * labels).sum(0)


def test_forward_backward_matches_brute_force():
	rng = np.random.default_rng(42)
	worst_ll, worst_pe = 0.0, 0.0
	for trial in range(60):
		nu, p = rng.uniform(1.5, 25), rng.uniform(0.01, 0.4)
		L = rng.uniform(0.5, 3.0)
		n = int(rng.integers(1, 8))
		x = np.sort(rng.uniform(0, L, n))
		if trial % 4 == 0 and n >= 2:                      # include near-coincident pairs
			x[1] = x[0] + 1e-3
			x = np.sort(x)
		ll_bf, pe_bf = brute_force(x, L, nu, p)
		xp = jnp.asarray(np.r_[x, np.zeros(10 - n)])       # padded, as in real batches
		ll = float(record_loglik_jit(xp, n, jnp.asarray(L), nu, p))
		pe = np.asarray(record_escape_posterior_jit(xp, n, jnp.asarray(L), nu, p))
		assert np.all(np.isnan(pe[n:])), "padding should be NaN"
		worst_ll = max(worst_ll, abs(ll - ll_bf))
		worst_pe = max(worst_pe, np.max(np.abs(pe[:n] - pe_bf)))
	assert worst_ll < 1e-9 and worst_pe < 1e-9


def test_forward_and_backward_totals_agree():
	rng = np.random.default_rng(7)
	worst = 0.0
	for _ in range(30):
		nu, p, L = rng.uniform(1.5, 25), rng.uniform(0.01, 0.4), rng.uniform(0.5, 3.0)
		x, _ = simulate_product(nu, p, L, rng, return_labels=True)
		n = len(x)
		xp = jnp.asarray(np.r_[x, np.zeros(12 - n)])
		w_start, w_edge, w_end, w_none = path_log_weights_jit(xp, n, jnp.asarray(L), nu, p)
		b, c, ll = forward_backward_jit(xp, n, jnp.asarray(L), nu, p)
		ll_back = -p * L + jax.scipy.special.logsumexp(jnp.concatenate([w_none[None], w_start + c]))
		worst = max(worst, abs(float(ll) - float(ll_back)))
	assert worst < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("nu, p", [(11.0, 0.06), (11.0, 0.20), (4.0, 0.20)])
def test_escape_posterior_calibration(nu, p):
	"""Posterior P(escape) is calibrated against the true labels of 40,000
	simulated products: overall, and within bins of the posterior."""
	rng = np.random.default_rng(42)
	lens = rng.uniform(0.6, 2.8, 40000)
	sims = [simulate_product(nu, p, L, rng, return_labels=True) for L in lens]
	X, NXO, LEN = pad_records([s[0] for s in sims], lens)
	lab = np.full(X.shape, np.nan)
	for r, (_, l) in enumerate(sims):
		lab[r, :len(l)] = l
	pe = escape_posterior(X, NXO, LEN, nu, p)
	m = ~np.isnan(lab)
	pe, lab = pe[m], lab[m]
	assert abs(pe.mean() - lab.mean()) < 0.01
	edges = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
	for lo, hi in zip(edges[:-1], edges[1:]):
		sel = (pe >= lo) & (pe < hi)
		if sel.sum() >= 500:
			q = pe[sel].mean()
			se = np.sqrt(q * (1 - q) / sel.sum())
			assert abs(lab[sel].mean() - q) < 4 * se + 0.005, (lo, hi, q, lab[sel].mean())
