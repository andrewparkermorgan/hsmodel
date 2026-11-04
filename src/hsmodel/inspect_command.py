"""
Describe an .npz file written by this package (`hsmodel inspect FILE.npz`).

The kind of file is recognized from its contents:
  crossover dataset   from `hsmodel simulate` or `hsmodel preprocess`: the
                      individuals, chromosomes and crossovers, distributions
                      of crossover counts and inter-crossover distances, a
                      per-chromosome table and, for simulations, the values
                      the data were simulated with;
  SVI fit             OUT_params.npz from `hsmodel infer` (or the
                      OUT_svi_params.npz starting point of `hsmodel nuts
                      --svi-init`): how the model was fitted, the convergence
                      of the optimization, and posterior summaries;
  NUTS fit            OUT_samples.npz from `hsmodel nuts`: sampler
                      diagnostics and posterior summaries with effective
                      sample sizes and R-hat.

Posterior summaries are given on the scale the model is fitted on (logit p,
log(nu - 1), log SDs) and, for the main quantities, on the natural scale
(p, nu, SDs).  Everything is written to stdout.

Example:
    hsmodel inspect simulated_data.npz
    hsmodel inspect svi_fit_params.npz
"""
import os
import sys
import zipfile

import numpy as np

from .data import load_dataset, summarize_dataset
from .errors import InputError
from .summaries import (GLOBALS, default_covariate_names, global_entries, entry_value, summarize, chrom_summary,
                        chrom_draw_summary)

DATASET_KEYS = ("X", "NXO", "LEN", "par", "chrom", "chrom_len", "P")

# Natural-scale versions of the globals: (site, name, description, transform)
_sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
NATURAL = [
	("a_p", "p", "escape fraction p at the intercept", _sig),
	("a_nu", "nu", "interference shape nu at the intercept", lambda x: 1.0 + np.exp(x)),
	("log_s_p", "s_p", "between-individual SD of logit p", np.exp),
	("log_s_nu", "s_nu", "between-individual SD of log(nu - 1)", np.exp),
	("log_t_p", "t_p", "between-chromosome SD of logit p", np.exp),
	("log_t_nu", "t_nu", "between-chromosome SD of log(nu - 1)", np.exp),
]
INTERCEPT_NOTE = ("At the intercept every covariate is 0 (its mean, if standardized) and the chromosome and "
                  "individual effects are 0.")


def add_arguments(ap):
	ap.add_argument("file", metavar="FILE.npz",
	                help="a dataset (from simulate or preprocess) or a fit (from infer or nuts)")


def run(a):
	try:
		kind, f = open_npz(a.file)
		text = {"dataset": describe_dataset, "svi": describe_svi, "nuts": describe_nuts}[kind](a.file, f)
	except (InputError, OSError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1
	print(text)
	return 0


# ---------------------------------------------------------------------------
# Recognizing the file
# ---------------------------------------------------------------------------
def open_npz(path):
	"""(kind, {key: array}) with kind 'dataset', 'svi' or 'nuts'; InputError
	if the file is not an .npz archive or not one written by this package."""
	if not os.path.isfile(path):
		raise InputError(f"{path}: no such file")
	try:
		with np.load(path, allow_pickle=False) as z:
			f = {k: z[k] for k in z.files}
	except (zipfile.BadZipFile, ValueError, EOFError):
		raise InputError(f"{path}: not an .npz archive (expected a dataset or fit written by hsmodel)") from None
	if not isinstance(f, dict):
		raise InputError(f"{path}: a single .npy array, not an .npz archive")
	return classify(f, path), f


def classify(f, path="file"):
	"""'dataset', 'svi' or 'nuts' from the keys of an .npz file."""
	if all(k in f for k in DATASET_KEYS):
		return "dataset"
	if "a_nu_loc" in f and "a_p_loc" in f:
		return "svi"
	if "a_nu" in f and "a_p" in f and np.ndim(f["a_nu"]) == 2:          # [chains, draws]
		return "nuts"
	shown = ", ".join(sorted(f)[:12]) + (", ..." if len(f) > 12 else "")
	raise InputError(f"{path}: not a dataset or a model fit written by hsmodel (it holds: {shown or 'nothing'})")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def align(rows, right=None):
	"""Rows of strings -> aligned columns (2 spaces apart); the first row is
	the header, underlined.  Columns listed in `right` (default: all but the
	first and last) are right-aligned."""
	n = max(len(r) for r in rows)
	rows = [list(r) + [""] * (n - len(r)) for r in rows]
	w = [max(len(r[j]) for r in rows) for j in range(n)]
	right = set(range(1, n - 1)) if right is None else set(right)
	fmt = lambda r: "  ".join(c.rjust(w[j]) if j in right else c.ljust(w[j]) for j, c in enumerate(r)).rstrip()
	return "\n".join(["  " + fmt(rows[0]), "  " + "  ".join("-" * x for x in w)] + ["  " + fmt(r) for r in rows[1:]])


def fields(pairs, pad=40):
	"""'  name:   value' lines."""
	return "\n".join(f"  {name + ':':{pad}}{value}" for name, value in pairs)


def num(v, digits=4):
	v = float(v)
	return "NA" if not np.isfinite(v) else f"{v:#.{digits}g}".rstrip(".")


def text_list(values, n=6):
	values = [str(v) for v in values]
	return ", ".join(values[:n]) + (f", ... ({len(values):,} in all)" if len(values) > n else "")


def scalar(f, key, default=None):
	return f[key].item() if key in f and np.ndim(f[key]) == 0 else default


def strings(f, key):
	return [str(v) for v in np.atleast_1d(f[key])] if key in f else None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def describe_dataset(path, f):
	d = load_dataset(path)
	simulated = "z" in d or "truth" in d
	empirical = "parent_names" in d
	if simulated:
		origin = "simulated with `hsmodel simulate`" + (f" (seed {d['seed']})" if "seed" in d else "")
	elif empirical:
		origin = "empirical crossover data read by `hsmodel preprocess`"
	else:
		origin = "origin not recorded"
	out = [f"{path}: crossover dataset, {origin}", "",
	       summarize_dataset(d, title="Summary")]
	if empirical:
		out += ["", "Individuals:", fields([("identifiers", text_list(d["parent_names"]))])]
	if simulated:
		out += ["", simulation_settings(d)]
	out += ["", chromosome_table(d)]
	return "\n".join(out)


def simulation_settings(d):
	"""The true values used by `hsmodel simulate`, and the covariates."""
	lines = ["Simulation settings (true values):"]
	t = d.get("truth")
	pairs = [("meiotic products per individual", f"{d['meioses']} per chromosome" if "meioses" in d else "NA")]
	if "chrom_effects" in d:
		pairs.append(("chromosome effects simulated", "yes" if d["chrom_effects"] else "no"))
	if "z" in d:
		z = np.asarray(d["z"], dtype=float)
		z = z[:, None] if z.ndim == 1 else z
		names = [f"z{k + 1}" for k in range(z.shape[1])]          # as `hsmodel infer` names them
		pairs.append(("covariates", ", ".join(f"{n} (mean {m:.3f}, SD {s:.3f})"
		                                      for n, m, s in zip(names, z.mean(0), z.std(0)))))
	lines.append(fields(pairs))
	if t:
		rows = [["parameter", "value", "description"]]
		desc = {"a_p": "logit p at the intercept", "a_nu": "log(nu - 1) at the intercept",
		        "s_p": "between-individual SD of logit p", "s_nu": "between-individual SD of log(nu - 1)",
		        "t_p": "between-chromosome SD of logit p", "t_nu": "between-chromosome SD of log(nu - 1)"}
		for k in ("a_p", "b_p", "s_p", "t_p", "a_nu", "b_nu", "s_nu", "t_nu"):
			if k not in t:
				continue
			v = np.atleast_1d(t[k])
			if k in ("b_p", "b_nu"):
				for j, x in enumerate(v):
					on = "logit p" if k == "b_p" else "log(nu - 1)"
					rows.append([f"{k}[z{j + 1}]", num(x), f"effect of covariate z{j + 1} on {on}"])
			else:
				rows.append([k, num(v[0]), desc[k]])
		lines += ["", align(rows)]
		if "a_p" in t and "a_nu" in t:
			lines.append(f"  (at the intercept: p = {num(_sig(t['a_p']))}, nu = {num(1 + np.exp(t['a_nu']))})")
	if "p_i" in d and "nu_i" in d:
		lines += ["", "True individual values (at a chromosome effect of 0):",
		          fields([("p_i", spread(d["p_i"])), ("nu_i", spread(d["nu_i"]))])]
	return "\n".join(lines)


def spread(x):
	x = np.asarray(x, dtype=float)
	q = np.percentile(x, [0, 2.5, 50, 97.5, 100])
	return (f"median {num(q[2])}, middle 95% {num(q[1])} to {num(q[3])}, range {num(q[0])} to {num(q[4])}")


def chromosome_table(d):
	"""Per chromosome: map length(s), meiotic products, crossovers per product."""
	C = len(d["chrom_len"])
	names = [str(v) for v in d["chrom_names"]] if "chrom_names" in d else [str(c + 1) for c in range(C)]
	n = np.bincount(d["chrom"], minlength=C)
	co = np.bincount(d["chrom"], weights=d["NXO"], minlength=C)
	by_sex = "chrom_len_by_sex" in d and "sex" in d
	sexes = [(k, s) for k, s in ((0, "male"), (1, "female")) if by_sex and np.any(d["sex"] == k + 1)]
	head = ["chromosome"] + ([f"{s} length (cM)" for _, s in sexes] if sexes else ["length (cM)"]) + \
		["meiotic products", "crossovers", "COs per product", "expected"]
	rows = [head]
	for c in range(C):
		lens = [d["chrom_len_by_sex"][k][c] for k, _ in sexes] if sexes else [d["chrom_len"][c]]
		rows.append([names[c]] + [f"{100 * v:.1f}" for v in lens] + [f"{n[c]:,}", f"{int(co[c]):,}",
		            f"{co[c] / n[c]:.3f}" if n[c] else "NA", f"{np.mean(d['LEN'][d['chrom'] == c]):.3f}" if n[c] else "NA"])
	note = ("  expected = mean map length of the products in Morgans (the expected number of crossovers per "
	        "product)")
	return "Chromosomes:\n" + align(rows, right=range(1, len(head))) + "\n" + note


# ---------------------------------------------------------------------------
# Fits: shared pieces
# ---------------------------------------------------------------------------
def fit_settings(f, sites, P, C, extra=()):
	"""Lines describing the data and model of a fit."""
	K = int(np.size(f["b_p_loc"] if "b_p_loc" in f else f["b_p"][0, 0]))
	names = strings(f, "covariate_names") or default_covariate_names(K)
	cov = ", ".join(names)
	if "standardized" in f:
		if bool(f["standardized"]):
			cov += "; standardized before fitting: " + "; ".join(
				f"{n} had mean {m:.4g}, SD {s:.4g}" for n, m, s in zip(names, f["covariate_means"], f["covariate_sds"]))
		else:
			cov += "; used as given"
	chrom = "log_t_p" in sites
	parent = "log_s_p" in sites
	ids = strings(f, "individual_ids")
	shown = (", ".join(ids[:4]) + (", ..." if len(ids) > 4 else "")) if ids else ""
	pairs = [("individuals", f"{P:,}" + (f"  ({shown})" if shown else "") if P else "not recorded"),
	         ("covariates", f"{K}  ({cov})"),
	         ("chromosome effects", f"included ({C} chromosomes)" if chrom else "not included"),
	         ("individual effects", "included" if parent else "not included")]
	return names, fields(pairs + list(extra))


def global_rows(summ, names, K, extra_cols=()):
	"""Aligned table of the globals; summ: {key: (mean, sd, lo, hi, *extra)}."""
	rows = [["parameter", "mean", "sd", "lower95", "upper95", *extra_cols, "description"]]
	present = [n for n in GLOBALS if any(key == n or key.startswith(n + "[") for key in summ)]
	for key, site, k, label in global_entries(present, K, names):
		if key in summ:
			v = summ[key]
			rows.append([key] + [f"{x:.4f}" for x in v[:4]] + [str(x) for x in v[4:]] + [label])
	return "Global parameters (posterior, on the fitted scale):\n" + align(rows)


def natural_rows(draws_or_normal):
	"""Aligned table of natural-scale quantities: {site: (median, mean, lo, hi)}."""
	rows = [["quantity", "median", "mean", "lower95", "upper95", "description"]]
	for site, name, desc, _ in NATURAL:
		if site in draws_or_normal:
			rows.append([name] + [num(x) for x in draws_or_normal[site]] + [desc])
	return "Natural scale:\n" + align(rows) + "\n  " + INTERCEPT_NOTE


def chrom_rows(cs, chrom_names):
	"""Aligned table of the chromosome effects; cs: {c_p / c_nu: (mean, lo, hi)}."""
	C = len(cs["c_p"][0])
	chrom_names = chrom_names or [str(c + 1) for c in range(C)]
	rows = [["chromosome", "c_p", "lower95", "upper95", "c_nu", "lower95", "upper95"]]
	for c in range(C):
		rows.append([str(chrom_names[c])] + [f"{v[j][c]:+.3f}" for v in (cs["c_p"], cs["c_nu"]) for j in range(3)])
	return ("Chromosome effects (posterior mean and 95% interval; c_p on logit p, c_nu on log(nu - 1)):\n"
	        + align(rows, right=range(1, 7)))


def individual_lines(means, sds, ids, site_scale):
	"""Spread of the individual effects' posterior means, and the most extreme individuals."""
	lines = []
	for name, m, s, scale in zip(("e_p", "e_nu"), means, sds, site_scale):
		m, s = np.asarray(m, dtype=float), np.asarray(s, dtype=float)
		lo, hi = np.argsort(m)[:3], np.argsort(m)[::-1][:3]
		lab = lambda i: (ids[i] if ids else str(i))
		lines.append(f"  {name} ({scale}): posterior means have SD {m.std():.3f} across individuals "
		             f"(range {m.min():+.2f} to {m.max():+.2f}); typical posterior SD {np.median(s):.3f}")
		lines.append(f"      lowest: {', '.join(f'{lab(i)} ({m[i]:+.2f})' for i in lo)};  "
		             f"highest: {', '.join(f'{lab(i)} ({m[i]:+.2f})' for i in hi)}")
	return ("Individual effects (standard normal; multiplied by s_p or s_nu in the model):\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# SVI fits
# ---------------------------------------------------------------------------
def describe_svi(path, f):
	sites = [n for n in GLOBALS if f"{n}_loc" in f]
	K = int(np.size(f["b_p_loc"])) if "b_p_loc" in f else 1
	lowrank = "glob_loc" in f
	from_nuts = "losses" not in f and "covariate_names" not in f
	P = len(f["individual_ids"]) if "individual_ids" in f else (len(f["e_p_loc"]) if "e_p_loc" in f else 0)
	C = len(f["c_p_loc"]) if "c_p_loc" in f else 0
	guide = "low-rank" if lowrank else "mean-field"
	if lowrank:
		guide += f" (rank {np.shape(f['glob_cov_factor'])[1]})"
	extra = [("variational guide", guide)]
	if "likelihood_exact" in f:
		extra.append(("likelihood", ("F* and G* in closed form (incomplete gamma)" if bool(f["likelihood_exact"])
		                             else "F* and G* by quadrature") + f"; sum in f* truncated at K = "
		                                                              f"{scalar(f, 'likelihood_K', 'NA')}"))
	if "losses" in f:
		extra += convergence(f["losses"])
	names, settings = fit_settings(f, sites, P, C, extra)
	what = ("SVI fit from `hsmodel nuts --svi-init` (the starting point of the NUTS chains)" if from_nuts
	        else "SVI fit from `hsmodel infer`")
	out = [f"{path}: {what}", "", "Settings:", settings, "",
	       global_rows(summarize(f, names), names, K), "",
	       natural_rows({s: normal_natural(f, s, tr) for s, _, _, tr in NATURAL if f"{s}_loc" in f})]
	if lowrank and "glob_corr" in f:
		out += ["", posterior_correlations(f["glob_corr"], [key for key, *_ in global_entries(sites, K, names)])]
	cs = chrom_summary(f)
	if cs:
		out += ["", chrom_rows(cs, strings(f, "chrom_names"))]
	if "e_p_loc" in f:
		out += ["", individual_lines((f["e_p_loc"], f["e_nu_loc"]), (f["e_p_scale"], f["e_nu_scale"]),
		                             strings(f, "individual_ids"), ("logit p", "log(nu - 1)"))]
	out += ["", "Intervals are mean +/- 1.96 SD of the variational posterior (approximate)."]
	return "\n".join(out)


def normal_natural(f, site, transform, n_nodes=64):
	"""(median, mean, lo, hi) of transform(x) for x ~ N(loc, scale^2) (the guide's
	marginal); the transforms are monotone, so quantiles map directly, and the
	mean is computed by Gauss-Hermite quadrature."""
	m, s = float(f[f"{site}_loc"]), float(f[f"{site}_scale"])
	t, w = np.polynomial.hermite_e.hermegauss(n_nodes)
	mean = float(np.sum(w * transform(m + s * t)) / np.sqrt(2 * np.pi))
	return (float(transform(m)), mean, float(transform(m - 1.96 * s)), float(transform(m + 1.96 * s)))


def convergence(losses):
	"""[(name, value)]: steps, and the -ELBO over the last tenth of the run
	compared with the tenth before (a crude check of convergence)."""
	losses = np.asarray(losses, dtype=float)
	n = len(losses)
	if n < 10:
		return [("optimization", f"{n:,} steps" + (f"; final -ELBO {losses[-1]:,.1f}" if n else ""))]
	w = n // 10
	last, prev = losses[-w:].mean(), losses[-2 * w:-w].mean()
	return [("optimization", f"{n:,} steps"),
	        ("-ELBO, last 10% of steps", f"mean {last:,.1f} (minibatch SD {losses[-w:].std():,.1f}); "
	                                     f"{'down' if prev > last else 'up'} {abs(prev - last):,.1f} from the "
	                                     f"10% before")]


def posterior_correlations(corr, keys, threshold=0.3):
	"""Pairs of globals that the low-rank guide finds correlated."""
	G = len(keys)
	corr = np.asarray(corr)[:G, :G]
	pairs = sorted(((abs(corr[i, j]), i, j) for i in range(G) for j in range(i + 1, G)
	                if abs(corr[i, j]) >= threshold), reverse=True)
	head = f"Posterior correlations between global parameters (low-rank guide; |r| >= {threshold}):"
	if not pairs:
		return head + "\n  none"
	return head + "\n" + "\n".join(f"  {keys[i]:>14} ~ {keys[j]:<14} r = {corr[i, j]:+.2f}" for _, i, j in pairs[:12])


# ---------------------------------------------------------------------------
# NUTS fits
# ---------------------------------------------------------------------------
def describe_nuts(path, f):
	from .summaries import global_summary              # imports numpyro (and JAX)
	sites = [n for n in GLOBALS if n in f]
	chains, draws = np.shape(f["a_nu"])
	K = int(np.shape(f["b_p"])[-1]) if "b_p" in f and np.ndim(f["b_p"]) == 3 else 1
	P = len(f["individual_ids"]) if "individual_ids" in f else (np.shape(f["e_p"])[-1] if "e_p" in f else 0)
	C = np.shape(f["c_p"])[-1] if "c_p" in f else 0
	extra = [("sampler", f"NUTS, {chains} chain(s) x {draws:,} draws after warmup")]
	n_div = scalar(f, "n_divergent")
	if n_div is not None:
		extra.append(("divergent transitions", f"{n_div}"))
	if "accept_prob" in f:
		extra.append(("mean acceptance probability", f"{np.mean(f['accept_prob']):.3f}"))
	if "num_steps" in f:
		st = np.asarray(f["num_steps"])
		extra.append(("leapfrog steps per draw", f"median {np.median(st):.0f}, max {st.max()}"))
	names, settings = fit_settings(f, sites, P, C, extra)

	summ = global_summary({k: f[k] for k in sites})
	rows, worst = {}, []
	for key, site, k, _ in global_entries(sites, K, names):
		s = {q: entry_value(summ[site][q], k) for q in ("mean", "std", "2.5%", "97.5%", "n_eff", "r_hat")}
		rows[key] = (s["mean"], s["std"], s["2.5%"], s["97.5%"], f"{s['n_eff']:.0f}", f"{s['r_hat']:.3f}")
		worst.append((s["r_hat"], s["n_eff"], key))
	flat = lambda x: np.asarray(x, dtype=float).reshape(-1)
	nat = {}
	for site, _, _, tr in NATURAL:
		if site in f:
			v = tr(flat(f[site]))
			nat[site] = (np.median(v), v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5))
	out = [f"{path}: NUTS fit from `hsmodel nuts`", "", "Settings:", settings, "",
	       global_rows(rows, names, K, extra_cols=("n_eff", "r_hat")), "", natural_rows(nat)]
	cs = chrom_draw_summary(f)
	if cs:
		out += ["", chrom_rows(cs, strings(f, "chrom_names"))]
	if "e_p" in f and "e_nu" in f:
		e = {n: np.asarray(f[n]).reshape(-1, np.shape(f[n])[-1]) for n in ("e_p", "e_nu")}
		out += ["", individual_lines((e["e_p"].mean(0), e["e_nu"].mean(0)), (e["e_p"].std(0), e["e_nu"].std(0)),
		                             strings(f, "individual_ids"), ("logit p", "log(nu - 1)"))]
	# diagnostics
	problems = []
	if n_div:
		problems.append(f"{n_div} divergent transition(s): the posterior may be explored incompletely")
	r_max, _, key = max(worst)
	if r_max > 1.01:
		problems.append(f"largest R-hat among the globals is {r_max:.3f} ({key}); want <= 1.01")
	ess_min, key = min((w[1], w[2]) for w in worst)
	if ess_min < 100 * chains:
		problems.append(f"smallest effective sample size among the globals is {ess_min:.0f} ({key}); "
		                f"want at least ~{100 * chains}")
	out += ["", "Diagnostics:"] + ([f"  warning: {p}" for p in problems] if problems else
	                                ["  no divergences; R-hat <= 1.01 and effective sample sizes adequate "
	                                 "for every global parameter"])
	return "\n".join(out)
