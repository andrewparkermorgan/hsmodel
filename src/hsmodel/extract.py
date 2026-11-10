"""
Implied nu and p for every individual, from an SVI fit (`hsmodel extract`).

For individual i the linear predictors are
    logit p_i      = a_p  + z_i . b_p  + s_p  e_p_i
    log(nu_i - 1)  = a_nu + z_i . b_nu + s_nu e_nu_i
and, with --per-chromosome, for individual i and chromosome c
    logit p_ic     = logit p_i     + c_p_c
    log(nu_ic - 1) = log(nu_i - 1) + c_nu_c.
Without --per-chromosome the chromosome effects are set to their prior mean,
0 (a "typical" chromosome).  Terms the fit does not have (individual or
chromosome effects) are left out.

Draws of every parameter are taken jointly from the variational posterior
(for the low-rank guide, with the posterior correlations among the globals
and the chromosome effects, and the coupling of each individual's effects to
the globals), the linear predictors are computed draw by draw, and p and nu
are obtained on the natural scale.  The posterior mean, median and 95%
interval (2.5% and 97.5% quantiles) of each are written to stdout as a
tab-separated table with a header line:
    iid [chromosome] nu_mean nu_median nu_lower95 nu_upper95
                     p_mean  p_median  p_lower95  p_upper95

Covariates: fits written by this version of `hsmodel infer` store the
covariate values they were fitted with, and nothing else is needed.  For older
fits give the dataset the model was fitted to with --data (and, unless the
data were simulated, the covariate file with --covariates); the covariates
the fit used are selected by name and, if the fit standardized them,
standardized with the means and SDs it stored.

Examples:
    hsmodel extract svi_fit_params.npz > individual_nu_p.tsv
    hsmodel extract svi_fit_params.npz --per-chromosome > individual_chrom_nu_p.tsv
    hsmodel extract old_fit_params.npz --data crossovers.npz --covariates covariates.txt
"""
import argparse
import sys

import numpy as np

from .errors import CommandError, InputError
from .summaries import COEFS, globals_used

CHUNK = 64          # individuals processed at once (memory: draws x CHUNK x chromosomes)


def add_arguments(ap):
	ap.add_argument("fit", metavar="FIT_params.npz", help="an SVI fit written by `hsmodel infer`")
	ap.add_argument("--per-chromosome", action="store_true",
	                help="one row per individual and chromosome, including the chromosome effects "
	                     "(default: one row per individual, chromosome effects at 0)")
	ap.add_argument("--draws", type=int, default=4000,
	                help="draws from the variational posterior (default 4000)")
	ap.add_argument("--seed", type=int, default=0, help="random seed for the draws (default 0)")
	ap.add_argument("--data", default=None,
	                help="the dataset the model was fitted to; needed only for fits that do not store "
	                     "their covariate values (written by older versions of `hsmodel infer`)")
	ap.add_argument("--covariates", default=None,
	                help="the covariate file the model was fitted with (with --data, for empirical data)")


def run(a):
	if a.draws < 2:
		raise CommandError("--draws must be at least 2")
	if a.covariates is not None and a.data is None:
		raise CommandError("--covariates needs --data")
	try:
		from .inspect_command import open_npz
		kind, f = open_npz(a.fit)
		if kind != "svi":
			raise InputError(f"{a.fit}: a {'NUTS fit' if kind == 'nuts' else kind}, not an SVI fit from "
			                 f"`hsmodel infer`")
		iids, z = fit_covariates(f, a)
		if a.per_chromosome and "c_p_loc" not in f:
			print("warning: the fit has no chromosome effects; --per-chromosome gives the same values "
			      "for every chromosome", file=sys.stderr)
		chrom_names = chromosome_names(f) if a.per_chromosome else None
		rows = individual_table(f, z, a.draws, np.random.default_rng(a.seed), chrom_names)
	except (InputError, OSError, KeyError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1
	head = ["iid"] + (["chromosome"] if a.per_chromosome else []) + \
		[f"{q}_{s}" for q in ("nu", "p") for s in ("mean", "median", "lower95", "upper95")]
	out = sys.stdout
	out.write("\t".join(head) + "\n")
	for i, c, vals in rows:
		cells = [iids[i]] + ([chrom_names[c]] if a.per_chromosome else []) + [f"{v:.6g}" for v in vals]
		out.write("\t".join(cells) + "\n")
	effects = [e for e, on in (("individual", "e_p_loc" in f or "local_loc" in f),
	                           ("chromosome", a.per_chromosome and "c_p_loc" in f)) if on]
	print(f"{len(iids):,} individuals; {a.draws:,} draws from the variational posterior "
	      f"({'low-rank' if 'glob_loc' in f else 'mean-field'} guide); covariates"
	      + (f", {' and '.join(effects)} effects" if effects else "") + " included", file=sys.stderr)
	return 0


# ---------------------------------------------------------------------------
# Individuals and covariates
# ---------------------------------------------------------------------------
def fit_covariates(f, a):
	"""(iids [P] str, z [P, K]): the individuals of the fit, in its order, and
	the covariate values it was fitted with."""
	if "individual_ids" not in f:
		raise InputError(f"{a.fit}: the fit does not record its individuals (written by `hsmodel nuts "
		                 f"--svi-init`?); use the OUT_params.npz file from `hsmodel infer`")
	iids = [str(v) for v in np.atleast_1d(f["individual_ids"])]
	if "covariate_values" in f:
		if a.data is not None:
			print(f"warning: {a.fit} stores the covariate values it was fitted with, which are used; "
			      f"--data{' and --covariates are' if a.covariates else ' is'} ignored", file=sys.stderr)
		z = np.asarray(f["covariate_values"], dtype=float)
		return iids, (z[:, None] if z.ndim == 1 else z)
	if a.data is None:
		raise InputError(f"{a.fit} does not store its covariate values (it was written by an older "
		                 f"`hsmodel infer`): give the dataset it was fitted to with --data, and for "
		                 f"empirical data the covariate file with --covariates")
	from .inputs import load_fit_inputs
	names = [str(v) for v in np.atleast_1d(f["covariate_names"])]
	args = argparse.Namespace(data=a.data, covariates=a.covariates, coefficients=",".join(names),
	                          standardize=False, chrom_effects=True, parent_effects=True)
	fi = load_fit_inputs(args, warn=lambda msg: print(f"warning: {msg}", file=sys.stderr))
	if fi.iids != iids:
		raise InputError(f"the individuals in {a.data} are not those of the fit {a.fit} (or not in the "
		                 f"same order); give the dataset the model was fitted to")
	z = fi.z
	if bool(f.get("standardized", False)):
		z = (z - np.asarray(f["covariate_means"], dtype=float)) / np.asarray(f["covariate_sds"], dtype=float)
	return iids, z


def chromosome_names(f):
	if "chrom_names" in f:
		return [str(v) for v in np.atleast_1d(f["chrom_names"])]
	if "c_p_loc" in f:
		return [str(c + 1) for c in range(len(f["c_p_loc"]))]
	raise InputError("the fit records neither chromosome names nor chromosome effects")


# ---------------------------------------------------------------------------
# Draws from the variational posterior
# ---------------------------------------------------------------------------
class GuideDraws:
	"""Joint draws of the globals and chromosome effects from an SVI guide,
	and draws of the individual effects (conditional on the globals for the
	low-rank guide).  Globals are [S] arrays, or [S, K] for b_p and b_nu;
	c_p, c_nu are [S, C] (absent without chromosome effects)."""

	def __init__(self, f, K, S, rng):
		self.f, self.rng, self.S = f, rng, S
		self.chrom = "c_p_loc" in f
		self.parent = "e_p_loc" in f or "local_loc" in f
		self.lowrank = "glob_loc" in f
		sites = globals_used(self.chrom, self.parent)
		g = {}
		if self.lowrank:
			mu = np.asarray(f["glob_loc"], dtype=float)
			W = np.asarray(f["glob_cov_factor"], dtype=float)
			dg = np.asarray(f["glob_cov_diag"], dtype=float)
			theta = (mu + rng.standard_normal((S, W.shape[1])) @ W.T
			         + np.sqrt(dg) * rng.standard_normal((S, len(mu))))          # [S, D]
			start = 0
			for n in sites:
				size = K if n in COEFS else 1
				g[n] = theta[:, start:start + size] if n in COEFS else theta[:, start]
				start += size
			self.G = start
			if self.chrom:
				C = (len(mu) - start) // 2
				g["c_p"], g["c_nu"] = theta[:, start:start + C], theta[:, start + C:]
			self.dglob = theta[:, :start] - mu[:start]                               # [S, G]
		else:
			for n in sites + (["c_p", "c_nu"] if self.chrom else []):
				loc = np.asarray(f[f"{n}_loc"], dtype=float)
				scale = np.asarray(f[f"{n}_scale"], dtype=float)
				g[n] = loc + scale * rng.standard_normal((S,) + loc.shape)
		self.g = g

	def individual_effects(self, ix):
		"""(e_p, e_nu), each [S, len(ix)], for the individuals ix."""
		S, f, rng = self.S, self.f, self.rng
		if self.lowrank:
			m = np.asarray(f["local_loc"], dtype=float)[ix]                       # [n, 2]
			L = np.asarray(f["local_scale_tril"], dtype=float)[ix]                # [n, 2, 2]
			B = np.asarray(f["local_coupling"], dtype=float)[ix]                  # [n, 2, G]
			eps = rng.standard_normal((S, len(ix), 2))
			e = m + np.einsum("ikg,sg->sik", B, self.dglob) + np.einsum("ikj,sij->sik", L, eps)
			return e[..., 0], e[..., 1]
		out = []
		for n in ("e_p", "e_nu"):
			loc = np.asarray(f[f"{n}_loc"], dtype=float)[ix]
			scale = np.asarray(f[f"{n}_scale"], dtype=float)[ix]
			out.append(loc + scale * rng.standard_normal((S, len(ix))))
		return tuple(out)


def linear_predictors(q, z, ix):
	"""(logit p, log(nu - 1)), each [S, len(ix)], at chromosome effect 0."""
	g = q.g
	lp = g["a_p"][:, None] + g["b_p"].reshape(q.S, -1) @ z[ix].T                 # [S, n]
	ln = g["a_nu"][:, None] + g["b_nu"].reshape(q.S, -1) @ z[ix].T
	if q.parent:
		e_p, e_nu = q.individual_effects(ix)
		lp = lp + np.exp(g["log_s_p"])[:, None] * e_p
		ln = ln + np.exp(g["log_s_nu"])[:, None] * e_nu
	return lp, ln


def natural_summary(x, axis=0):
	"""[mean, median, 2.5%, 97.5%] along `axis` -> array with that axis replaced by 4 (first)."""
	return np.stack([x.mean(axis), *np.percentile(x, [50, 2.5, 97.5], axis=axis)])


def individual_table(f, z, S, rng, chrom_names=None):
	"""Rows (individual index, chromosome index or None, [nu mean, median, lo, hi,
	p mean, median, lo, hi]).  With chrom_names, one row per individual and
	chromosome, including the chromosome effects."""
	P, K = z.shape
	n_fit = len(f["individual_ids"])
	if P != n_fit:
		raise InputError(f"{P} individuals with covariates but {n_fit} in the fit")
	K_fit = int(np.size(f["b_p_loc"]))
	if K != K_fit:
		raise InputError(f"{K} covariate(s) but the fit has {K_fit} coefficient(s) per linear predictor")
	q = GuideDraws(f, K, S, rng)
	rows = []
	sig = lambda x: 1.0 / (1.0 + np.exp(-x))
	for start in range(0, P, CHUNK):
		ix = np.arange(start, min(start + CHUNK, P))
		lp, ln = linear_predictors(q, z, ix)                                     # [S, n]
		if chrom_names is None:
			nu, p = natural_summary(1.0 + np.exp(ln)), natural_summary(sig(lp))       # [4, n]
			rows += [(i, None, np.concatenate([nu[:, j], p[:, j]])) for j, i in enumerate(ix)]
			continue
		C = len(chrom_names)
		if q.chrom:
			lp, ln = lp[:, :, None] + q.g["c_p"][:, None, :], ln[:, :, None] + q.g["c_nu"][:, None, :]
		else:
			lp, ln = np.repeat(lp[:, :, None], C, 2), np.repeat(ln[:, :, None], C, 2)
		nu, p = natural_summary(1.0 + np.exp(ln)), natural_summary(sig(lp))           # [4, n, C]
		rows += [(i, c, np.concatenate([nu[:, j, c], p[:, j, c]])) for j, i in enumerate(ix) for c in range(C)]
	return rows
