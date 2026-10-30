"""
Correlation between the posterior means of nu_ic and p_ic across all
(parent, chromosome) cells, from an SVI fit of the default simulated
dataset (examples/svi_demo.py saves its guide parameters to svi_params.npz).

  nu_ic = 1 + exp(a_nu + b_nu z_i + s_nu e_nu_i + c_nu_c)
  p_ic  = sigmoid(a_p + b_p z_i + s_p e_p_i + c_p_c)

The guide is mean-field Gaussian, so each posterior mean E[nu_ic] and E[p_ic]
is estimated by Monte Carlo from the guide (n_draws draws, processed in chunks
of parents).  The correlation is reported on the model's scale,
log(E[nu_ic] - 1) vs logit(E[p_ic]); the true values' correlation is shown for
comparison.

Usage: python examples/plot_nu_p_correlation.py [svi_params.npz] [OUT.png]
"""
import sys
import numpy as np

from hsmodel.model import as_covariate_matrix
from hsmodel.plots import STYLE, _setup_matplotlib
from hsmodel.simulate import simulate_dataset


def posterior_means(params, z, n_draws=1000, chunk=100, seed=0):
	"""Monte Carlo E[p_ic] and E[nu_ic] under the guide: arrays [P, C].
	z: [P, K] covariates (1-D: one covariate)."""
	rng = np.random.default_rng(seed)
	z = as_covariate_matrix(z)
	P, C = len(z), len(params["c_p_loc"])

	def g(name, shape=()):
		return params[f"{name}_loc"] + params[f"{name}_scale"] * rng.standard_normal((n_draws,) + shape)

	K = z.shape[1]
	a_p, s_p = g("a_p"), np.exp(g("log_s_p"))
	a_nu, s_nu = g("a_nu"), np.exp(g("log_s_nu"))
	b_p = (np.atleast_1d(params["b_p_loc"]) + np.atleast_1d(params["b_p_scale"])
	       * rng.standard_normal((n_draws, K)))                                    # [draws, K]
	b_nu = (np.atleast_1d(params["b_nu_loc"]) + np.atleast_1d(params["b_nu_scale"])
	        * rng.standard_normal((n_draws, K)))
	c_p, c_nu = g("c_p", (C,)), g("c_nu", (C,))
	Ep, Enu = np.empty((P, C)), np.empty((P, C))
	for start in range(0, P, chunk):
		sl = slice(start, min(start + chunk, P))
		n = sl.stop - sl.start
		e_p = params["e_p_loc"][sl] + params["e_p_scale"][sl] * rng.standard_normal((n_draws, n))
		e_nu = params["e_nu_loc"][sl] + params["e_nu_scale"][sl] * rng.standard_normal((n_draws, n))
		lp = (a_p[:, None] + b_p @ z[sl].T + s_p[:, None] * e_p)[:, :, None] + c_p[:, None, :]
		ln = (a_nu[:, None] + b_nu @ z[sl].T + s_nu[:, None] * e_nu)[:, :, None] + c_nu[:, None, :]
		Ep[sl] = (1 / (1 + np.exp(-lp))).mean(0)
		Enu[sl] = (1 + np.exp(ln)).mean(0)
	return Ep, Enu


def main(params_path="svi_params.npz", out="svi_nu_p_correlation.png"):
	params = dict(np.load(params_path))
	d = simulate_dataset()
	P = len(params["e_p_loc"])
	Ep, Enu = posterior_means(params, d["z"][:P])
	C = Ep.shape[1]
	r = np.corrcoef(np.log(Enu.ravel() - 1), np.log(Ep.ravel() / (1 - Ep.ravel())))[0, 1]
	pt, nt = d["p_ic"][:P], d["nu_ic"][:P]
	r_true = np.corrcoef(np.log(nt.ravel() - 1), np.log(pt.ravel() / (1 - pt.ravel())))[0, 1]

	plt = _setup_matplotlib()
	S = STYLE
	fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=S["surface"])
	ax.set_facecolor(S["surface"])
	ax.plot(Enu.ravel(), 100 * Ep.ravel(), "o", color=S["post"], markersize=2.5, alpha=0.15, markeredgewidth=0)
	ax.grid(color=S["grid"], linewidth=0.8)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.set_xlabel("posterior mean of nu_ic (interference)")
	ax.set_ylabel("posterior mean of p_ic, escape fraction (%)")
	ax.text(0.98, 0.97, f"r = {r:.2f}\n(true values: r = {r_true:.2f})", transform=ax.transAxes,
	        ha="right", va="top", fontsize=10, color=S["ink"])
	ax.set_title(f"Posterior means of nu_ic and p_ic: {P:,} parents x {C} chromosomes = {P * C:,} cells (SVI)",
	             loc="left", fontsize=11, color=S["ink"], pad=24)
	ax.text(0, 1.02, "Correlation on the model's scale: log(E[nu_ic] - 1) vs logit(E[p_ic]). "
	                 "Means estimated from 1,000 draws of the variational posterior.",
	        transform=ax.transAxes, fontsize=8.5, color=S["ink2"])
	fig.savefig(out, dpi=150, facecolor=S["surface"], bbox_inches="tight")
	plt.close(fig)
	print(f"{P * C:,} cells: correlation of posterior means r = {r:.3f}; true values r = {r_true:.3f}")
	return out


if __name__ == "__main__":
	print(main(*sys.argv[1:3]))
