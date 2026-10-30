"""
Inputs shared by `hsmodel infer` and `hsmodel nuts`: the dataset, the
parent-level covariates (from the dataset itself for simulated data, otherwise
from a covariate file), the data summary written to stderr, and the true
parameter values when the data were simulated.
"""
import argparse
import sys

import numpy as np

from .data import load_dataset
from .errors import InputError

COVARIATE_HELP = """\
Covariates:
  Simulated data (from `hsmodel simulate`) carry their covariate values z, and
  these are used automatically; --covariates is then not needed (and is
  ignored, with a warning, if given).  Simulated covariates are named z1, z2,
  ..., and simulated individuals 0, 1, ..., P-1.  For other data:
  --covariates FILE     parent-level covariates, one individual per line,
                        whitespace- or tab-separated.  Optional header line
                        starting with '#' giving the column names; the first
                        column is always the individual identifier and must be
                        named 'iid'.  Without a header the covariates are named
                        z1, z2, ...  The individuals in the data and in the
                        covariate file must match one to one.
  --coefficients LIST   covariates to include, comma-separated, each given by
                        name or by index (1 = the first covariate column, i.e.
                        the column after iid).  Default: all covariates.

The model assumes standardized covariates (mean 0, SD 1); --standardize
standardizes the selected columns first (coefficients are then per SD), and
otherwise the fit warns about columns that do not look standardized.  If the
data carry sex-specific maps, each individual's chromosome lengths are taken
from the map for its sex.  For simulated data, the true values are shown
alongside the estimates (unless --standardize changed the covariates' scale).
"""


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------
def read_covariates(path):
	"""-> (iids [N] str, names [K] str, values [N, K] float).  Missing values
	(empty, NA, NaN) are read as NaN and checked later, for the individuals used."""
	names, rows, iids = None, [], []
	with open(path) as fh:
		for ln, line in enumerate(fh, 1):
			s = line.strip()
			if not s:
				continue
			if s.startswith("#"):
				if names is not None or rows:
					continue                                     # later comment lines are ignored
				head = s[1:].split()
				if not head or head[0] != "iid":
					raise InputError(f"{path}, line {ln}: the header's first column must be named 'iid', "
					                 f"got {head[0] if head else '(nothing)'!r}")
				if len(set(head)) != len(head):
					raise InputError(f"{path}, line {ln}: duplicate column names in the header")
				names = head[1:]
				continue
			f = s.split("\t") if "\t" in s else s.split()
			f = [x.strip() for x in f]
			if names is not None and len(f) != len(names) + 1:
				raise InputError(f"{path}, line {ln}: expected {len(names) + 1} columns (iid and "
				                 f"{len(names)} covariates), got {len(f)}")
			if rows and len(f) != len(rows[0]) + 1:
				raise InputError(f"{path}, line {ln}: expected {len(rows[0]) + 1} columns, got {len(f)}")
			if len(f) < 2:
				raise InputError(f"{path}, line {ln}: need an iid and at least one covariate")
			vals = []
			for j, x in enumerate(f[1:], 2):
				if x in ("", "NA", "na", "NaN", "nan", "."):
					vals.append(np.nan)
					continue
				try:
					vals.append(float(x))
				except ValueError:
					raise InputError(f"{path}, line {ln}, column {j}: {x!r} is not a number") from None
			iids.append(f[0]); rows.append(vals)
	if not rows:
		raise InputError(f"{path}: no covariate rows found")
	K = len(rows[0])
	if names is None:
		names = [f"z{k + 1}" for k in range(K)]
	dup = sorted({i for i in iids if iids.count(i) > 1})
	if dup:
		raise InputError(f"{path}: individual(s) listed more than once: {', '.join(dup[:10])}"
		                 + (" ..." if len(dup) > 10 else ""))
	return np.array(iids), names, np.array(rows, dtype=float)


def select_coefficients(spec, names):
	"""Column indices (0-based into the covariates) for --coefficients."""
	if spec is None or not spec.strip():
		return list(range(len(names)))
	out = []
	for tok in [t.strip() for t in spec.split(",") if t.strip()]:
		if tok in names:
			j = names.index(tok)
		elif tok.isdigit() and 1 <= int(tok) <= len(names):
			j = int(tok) - 1
		else:
			raise InputError(f"--coefficients: {tok!r} is neither a covariate name ({', '.join(names)}) "
			                 f"nor an index from 1 to {len(names)}")
		if j in out:
			raise InputError(f"--coefficients: covariate {names[j]!r} selected twice")
		out.append(j)
	return out


def match_individuals(data_iids, cov_iids):
	"""Row order of the covariate table matching the data's individuals; raises
	unless the two sets of identifiers correspond one to one."""
	data_set, cov_set = set(data_iids), set(cov_iids)
	only_data = [i for i in data_iids if i not in cov_set]
	only_cov = [i for i in cov_iids if i not in data_set]
	if only_data or only_cov:
		msg = ["individuals in the crossover data and the covariate file do not match one to one:"]
		if only_data:
			msg.append(f"  {len(only_data)} in the data but not the covariates: {', '.join(only_data[:10])}"
			           + (" ..." if len(only_data) > 10 else ""))
		if only_cov:
			msg.append(f"  {len(only_cov)} in the covariates but not the data: {', '.join(only_cov[:10])}"
			           + (" ..." if len(only_cov) > 10 else ""))
		raise InputError("\n".join(msg))
	pos = {iid: r for r, iid in enumerate(cov_iids)}
	return np.array([pos[i] for i in data_iids])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def add_input_arguments(ap):
	ap.add_argument("--data", required=True, help=".npz file from hsmodel preprocess or hsmodel simulate")
	ap.add_argument("--covariates", default=None,
	                help="parent-level covariate file (see above); not needed for simulated data")
	ap.add_argument("--coefficients", default=None,
	                help="covariates to include: comma-separated names or 1-based indices (default: all)")
	ap.add_argument("--standardize", action="store_true",
	                help="standardize the selected covariates to mean 0, SD 1 before fitting")
	ap.add_argument("--chrom-effects", action=argparse.BooleanOptionalAction, default=True,
	                help="include chromosome-specific effects (default: yes; --no-chrom-effects to drop them)")


class FitInputs:
	"""The dataset and covariates for a fit.

	d: dataset dict; P; iids [P]; z [P, K] (standardized if requested);
	names [K]; means, sds [K] (before standardization); source ("simulated" or
	the covariate file); cols: selected covariate columns; truth, truth_chrom:
	true values on the model's sampled scale (simulated data only, else None)."""


def load_fit_inputs(a, warn=None):
	"""Load the dataset and covariates named by the parsed arguments `a`.
	Raises InputError for problems with the input files."""
	warn = warn or (lambda msg: print(f"warning: {msg}", file=sys.stderr))
	f = FitInputs()
	d = f.d = load_dataset(a.data)
	P = f.P = int(d["P"])
	f.iids = ([str(x) for x in d["parent_names"]] if "parent_names" in d else [str(i) for i in range(P)])
	if "z" in d:
		z_all = np.asarray(d["z"], dtype=float)
		z_all = z_all[:, None] if z_all.ndim == 1 else z_all
		all_names = [f"z{k + 1}" for k in range(z_all.shape[1])]
		f.source = "simulated"
		if a.covariates is not None:
			warn(f"{a.data} holds simulated covariate values, which are used; --covariates "
			     f"{a.covariates} is ignored")
	else:
		if a.covariates is None:
			raise InputError(f"{a.data} holds no covariate values (it is not simulated data): "
			                 f"give a covariate file with --covariates")
		cov_iids, all_names, cov_vals = read_covariates(a.covariates)
		z_all = cov_vals[match_individuals(f.iids, list(cov_iids))]
		f.source = a.covariates
	f.cols = select_coefficients(a.coefficients, all_names)
	z = z_all[:, f.cols]
	f.names = [all_names[j] for j in f.cols]
	bad = np.argwhere(~np.isfinite(z))
	if bad.size:
		ex = ", ".join(f"{f.iids[i]}:{f.names[j]}" for i, j in bad[:10])
		raise InputError(f"{len(bad)} missing covariate value(s) among the selected covariates "
		                 f"(individual:covariate): {ex}" + (" ..." if len(bad) > 10 else ""))
	f.means, f.sds = z.mean(0), z.std(0)
	if a.standardize:
		if np.any(f.sds == 0):
			raise InputError("cannot standardize constant covariate(s): "
			                 + ", ".join(n for n, s in zip(f.names, f.sds) if s == 0))
		z = (z - f.means) / f.sds
	f.z = z
	f.truth, f.truth_chrom = simulated_truth(d, f.cols, a.chrom_effects, a.standardize)
	return f


def simulated_truth(d, cols, chrom_effects, standardized):
	"""(truth, truth_chrom) for simulated data, else (None, None).  truth is on
	the sampled scale of the globals (log SDs) with b_p, b_nu restricted to the
	selected covariates; omitted when the covariates were re-standardized
	(the coefficients are then on a different scale)."""
	if "truth" not in d or standardized:
		return None, None
	from .simulate import truth_sampled
	t = truth_sampled(d["truth"])
	t["b_p"], t["b_nu"] = t["b_p"][cols], t["b_nu"][cols]
	simulated_chrom = bool(d.get("chrom_effects", True))
	if not simulated_chrom:                        # no chromosome effects were simulated
		t.pop("log_t_p"); t.pop("log_t_nu")
	truth_chrom = None
	if chrom_effects and "c_p" in d:
		truth_chrom = {"c_p": np.asarray(d["c_p"]), "c_nu": np.asarray(d["c_nu"])}
	return t, truth_chrom


def subset_parents(f, k):
	"""Restrict the inputs to the first k individuals (in data order)."""
	if k is None or k >= f.P:
		return f
	d = dict(f.d)
	sel = d["par"] < k
	for key in ("X", "NXO", "LEN", "par", "replicate", "chrom"):
		d[key] = d[key][sel]
	for key in ("parent_names", "sex", "z", "p_i", "nu_i", "p_ic", "nu_ic"):
		if key in d:
			d[key] = d[key][:k]
	d["P"] = k
	f.d, f.P, f.iids, f.z = d, k, f.iids[:k], f.z[:k]
	return f


def parent_lengths_of(d):
	"""Per-individual chromosome lengths [P, C] from sex-specific maps, or None."""
	if "chrom_len_by_sex" in d and "sex" in d:
		from .likelihood import parent_chrom_lengths
		return parent_chrom_lengths(d["chrom_len_by_sex"], d["sex"])
	return None


def data_summary(f, chrom_effects, standardize):
	"""The lines written to stderr before a fit."""
	d, K = f.d, len(f.names)
	return (f"individuals with data:            {f.P:,}\n"
	        f"meiotic products:                 {len(d['NXO']):,}\n"
	        f"crossovers:                       {int(d['NXO'].sum()):,}\n"
	        f"model coefficients to estimate:   {2 * K}  ({K} covariate(s): {', '.join(f.names)}; "
	        f"one coefficient each for logit p and log(nu - 1))\n"
	        f"chromosome-specific effects:      {'included' if chrom_effects else 'not included'}"
	        f" ({len(d['chrom_len'])} chromosomes)\n"
	        f"covariates:                       "
	        f"{'simulated values from the data file' if f.source == 'simulated' else 'from ' + f.source}; "
	        f"{'standardized here' if standardize else 'used as given'}")


def global_table(rows, truth_x, extra_cols=()):
	"""Tab-separated table of global parameters.  rows: {key: (mean, sd, lo,
	hi, *extra)}; truth_x: {key: true value} or None (adds a truth column)."""
	head = ["parameter", "mean", "sd", "lower95", "upper95", *extra_cols] + (["truth"] if truth_x else [])
	out = ["\t".join(head)]
	for key, vals in rows.items():
		cells = [key] + [f"{v:.5f}" if isinstance(v, float) else str(v) for v in vals]
		if truth_x:
			cells.append(f"{truth_x[key]:.5f}" if key in truth_x else "NA")
		out.append("\t".join(cells))
	return "\n".join(out)


def chrom_table(cs, chrom_names, truth_chrom=None):
	"""Tab-separated table of chromosome effects; cs: {site: (mean, lo, hi)}."""
	head = "chromosome\teffect_on_logit_p\tlower95\tupper95\teffect_on_log_nu_minus_1\tlower95\tupper95"
	if truth_chrom:
		head += "\ttrue_effect_on_logit_p\ttrue_effect_on_log_nu_minus_1"
	out = [head]
	for c, name in enumerate(chrom_names):
		(mp, lop, hip), (mn, lon, hin) = [(v[0][c], v[1][c], v[2][c]) for v in (cs["c_p"], cs["c_nu"])]
		line = f"{name}\t{mp:.5f}\t{lop:.5f}\t{hip:.5f}\t{mn:.5f}\t{lon:.5f}\t{hin:.5f}"
		if truth_chrom:
			line += f"\t{truth_chrom['c_p'][c]:.5f}\t{truth_chrom['c_nu'][c]:.5f}"
		out.append(line)
	return "\n".join(out)
