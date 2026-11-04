"""
Parameter names, priors and labels of the hierarchical Housworth-Stahl model,
and posterior summaries of an SVI guide's parameters.

Plain NumPy (no JAX), so `hsmodel inspect` can read fits quickly.
hsmodel.model and hsmodel.svi re-export everything here.
"""
import numpy as np

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


def default_covariate_names(K):
	return ["z"] if K == 1 else [f"z{k + 1}" for k in range(K)]


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


# ---------------------------------------------------------------------------
# Summaries of SVI guide parameters
# ---------------------------------------------------------------------------
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


def intercept_nu(params):
	"""Posterior of nu at the intercept: every covariate 0 and the chromosome
	and parent effects at their mean, 0, so nu = 1 + exp(a_nu).  Under the
	guide a_nu ~ N(m, s^2), so nu - 1 is lognormal.  Returns
	{"median", "mean", "lower95", "upper95"}."""
	m, s = float(params["a_nu_loc"]), float(params["a_nu_scale"])
	return {"median": 1.0 + np.exp(m), "mean": 1.0 + np.exp(m + 0.5 * s * s),
	        "lower95": 1.0 + np.exp(m - 1.96 * s), "upper95": 1.0 + np.exp(m + 1.96 * s)}


def intercept_p(params, n_nodes=64):
	"""Posterior of the escape fraction p at the intercept: every covariate 0
	and the chromosome and parent effects at their mean, 0, so
	p = 1 / (1 + exp(-a_p)).  Under the guide a_p ~ N(m, s^2): p is monotone in
	a_p, so the median and interval map directly; the mean (no closed form) is
	computed by Gauss-Hermite quadrature.  Returns {"median", "mean",
	"lower95", "upper95"}."""
	m, s = float(params["a_p_loc"]), float(params["a_p_scale"])
	sig = lambda x: 1.0 / (1.0 + np.exp(-x))
	t, w = np.polynomial.hermite_e.hermegauss(n_nodes)           # weight exp(-t^2 / 2)
	mean = float(np.sum(w * sig(m + s * t)) / np.sqrt(2 * np.pi))
	return {"median": sig(m), "mean": mean, "lower95": sig(m - 1.96 * s), "upper95": sig(m + 1.96 * s)}


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


# ---------------------------------------------------------------------------
# Summaries of NUTS draws
# ---------------------------------------------------------------------------
def global_summary(by_chain):
	"""numpyro summary (mean, sd, quantiles, n_eff, r_hat) for the globals present.
	by_chain: {name: [chains, draws, ...]}, e.g. mcmc.get_samples(group_by_chain=True).
	Imports numpyro (and with it JAX) when called."""
	from numpyro.diagnostics import summary as mcmc_summary
	return mcmc_summary({k: np.asarray(by_chain[k]) for k in GLOBALS if k in by_chain}, prob=0.95)


def chrom_draw_summary(by_chain):
	"""{c_p / c_nu: (mean [C], 2.5% [C], 97.5% [C])} from the NUTS draws."""
	out = {}
	for n in CHROM_SITES:
		if n in by_chain:
			d = np.asarray(by_chain[n]).reshape(-1, np.asarray(by_chain[n]).shape[-1])
			out[n] = (d.mean(0), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0))
	return out
