"""
Simulate crossover data under the hierarchical Housworth-Stahl model with
parent and chromosome random effects (`hsmodel simulate`).

Terminology: a meiotic product is one simulated chromosome in one individual
(one chromosome of one gamete), i.e. one record of the output.

For parent i with K standardized covariates z_i ~ N(0, I_K) and chromosome c:
    logit p_ic      = a_p  + z_i . b_p  + s_p  * e_p_i  + c_p_c,    e_p_i, e_nu_i ~ N(0, 1)
    log(nu_ic - 1)  = a_nu + z_i . b_nu + s_nu * e_nu_i + c_nu_c,   c_p_c ~ N(0, t_p^2), c_nu_c ~ N(0, t_nu^2)
K is the length of b_p (a scalar b_p means K = 1; a scalar b_nu is repeated K
times).  Each parent contributes `meioses` meiotic products for every
chromosome, so the dataset has parents x meioses x chromosomes meiotic
products.  Crossovers on a meiotic product follow simulate_product (gamma
renewal chiasmata plus a Poisson escape pathway, each chiasma seen in the
product with probability 1/2).

Command line:
    hsmodel simulate [--parents 1500] [--meioses 3]
                     [--chrom-lengths 0.6,0.7,... | FILE] [--a-p ...] ... [--t-nu ...]
                     [--no-chrom-effects] [--seed 1] [--out simulated_data.npz]
  --chrom-lengths FILE: one chromosome per line as a (name, length in Morgans)
  tuple, e.g. "(chr1, 2.70)", "chr1, 2.70", "chr1 2.70" or tab-separated;
  blank lines and lines starting with # are ignored.

Prints a summary to stdout (parents, meiotic products, crossovers, mean
crossovers per meiotic product, and stem-and-leaf plots of crossovers per
meiotic product and of the distance between consecutive crossovers on the same
meiotic product), and writes the data to --out; hsmodel.data.load_dataset(path)
reads it back as the same dict simulate_dataset returns.
"""
import os

import numpy as np

from .data import pad_records, save_dataset, summarize_dataset
from .errors import CommandError

# Default truth (natural scale for the SDs) and chromosome lengths.
DEFAULT_TRUTH = dict(a_p=float(np.log(0.06 / 0.94)), b_p=0.4, s_p=0.3, t_p=0.3,
                     a_nu=float(np.log(10.0)), b_nu=0.0, s_nu=0.15, t_nu=0.2)
DEFAULT_CHROM_LEN = np.linspace(0.6, 2.8, 22)          # Morgans, roughly female-like


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_product(nu, p, L, rng, burn=10.0, return_labels=False):
	"""Crossover positions on one meiotic product (one chromosome of length L
	Morgans in one gamete).  Interfering chiasmata: stationary gamma renewal
	process (shape nu, rate 2 nu (1 - p)), made stationary by a burn-in;
	escape chiasmata: Poisson with intensity 2p; each chiasma is kept with
	probability 1/2.  Independent of the likelihood code, for validation.
	With return_labels=True also returns a boolean array, True = escape crossover."""
	rate = 2.0 * nu * (1.0 - p)
	pts, pos = [], -burn
	while True:
		pos += rng.gamma(nu, 1.0 / rate)
		if pos > L:
			break
		if pos >= 0:
			pts.append(pos)
	esc = rng.uniform(0, L, rng.poisson(2 * p * L))
	allp = np.concatenate([np.asarray(pts, dtype=float), esc])
	lab = np.concatenate([np.zeros(len(pts), bool), np.ones(len(esc), bool)])
	order = np.argsort(allp)
	allp, lab = allp[order], lab[order]
	keep = rng.random(len(allp)) < 0.5
	return (allp[keep], lab[keep]) if return_labels else allp[keep]


def simulate_dataset(P=1500, meioses=3, chrom_len=DEFAULT_CHROM_LEN, truth=DEFAULT_TRUTH, seed=1,
                     chrom_effects=True, chrom_names=None):
	"""Simulate the dataset.  Returns a dict with the padded records (X, NXO,
	LEN; one record = one meiotic product, i.e. one chromosome in one
	individual; parents x meioses x chromosomes records), each record's parent,
	replicate (which of the parent's `meioses` products on that chromosome,
	0..meioses-1) and chromosome, z [P, K], the true
	parent-level p_i and nu_i (at a chromosome effect of 0), the true
	chromosome effects c_p and c_nu, the per-(parent, chromosome) p_ic and
	nu_ic, chrom_len, chrom_names, P and meioses.

	Random streams: parents' covariates and effects, then all crossovers, come
	from default_rng(seed); chromosome effects from default_rng(seed + 1000),
	so chrom_effects=False leaves the crossovers identical to a run without
	chromosome effects."""
	chrom_len = np.asarray(chrom_len, dtype=float)
	C = len(chrom_len)
	if chrom_names is None:
		chrom_names = [str(c + 1) for c in range(C)]
	rng = np.random.default_rng(seed)
	rng_c = np.random.default_rng(seed + 1000)
	b_p, b_nu = coefficient_vectors(truth)
	K = b_p.size
	z = rng.standard_normal((P, K))            # K = 1: the same draws as standard_normal(P)
	lp = truth["a_p"] + z @ b_p + truth["s_p"] * rng.standard_normal(P)
	ln = truth["a_nu"] + z @ b_nu + truth["s_nu"] * rng.standard_normal(P)
	if chrom_effects:
		c_p = truth["t_p"] * rng_c.standard_normal(C)
		c_nu = truth["t_nu"] * rng_c.standard_normal(C)
	else:
		c_p, c_nu = np.zeros(C), np.zeros(C)
	p_ic = 1 / (1 + np.exp(-(lp[:, None] + c_p[None, :])))
	nu_ic = 1 + np.exp(ln[:, None] + c_nu[None, :])
	xo, lens, par, rep, chrom = [], [], [], [], []
	for i in range(P):
		for m in range(meioses):
			for c, L in enumerate(chrom_len):
				xo.append(simulate_product(nu_ic[i, c], p_ic[i, c], L, rng))
				lens.append(L); par.append(i); rep.append(m); chrom.append(c)
	X, NXO, LEN = pad_records(xo, lens)
	return dict(X=X, NXO=NXO, LEN=LEN, par=np.array(par), replicate=np.array(rep), chrom=np.array(chrom), z=z,
	            p_i=1 / (1 + np.exp(-lp)), nu_i=1 + np.exp(ln), c_p=c_p, c_nu=c_nu, p_ic=p_ic, nu_ic=nu_ic,
	            chrom_len=chrom_len, chrom_names=np.array(chrom_names), P=P, meioses=meioses)


def coefficient_vectors(truth):
	"""(b_p, b_nu) as K-vectors; a scalar b_nu is repeated to match b_p."""
	b_p = np.atleast_1d(np.asarray(truth["b_p"], dtype=float))
	b_nu = np.atleast_1d(np.asarray(truth["b_nu"], dtype=float))
	if b_nu.size == 1 and b_p.size > 1:
		b_nu = np.full(b_p.size, b_nu[0])
	if b_nu.size != b_p.size:
		raise ValueError(f"b_p has {b_p.size} coefficients but b_nu has {b_nu.size}")
	return b_p, b_nu


def truth_sampled(truth):
	"""A truth dict (natural-scale SDs) on the scale the model samples its
	globals (log SDs; see hsmodel.model.GLOBALS); b_p and b_nu as K-vectors."""
	b_p, b_nu = coefficient_vectors(truth)
	return {"a_p": float(truth["a_p"]), "b_p": b_p, "log_s_p": float(np.log(truth["s_p"])),
	        "log_t_p": float(np.log(truth["t_p"])), "a_nu": float(truth["a_nu"]), "b_nu": b_nu,
	        "log_s_nu": float(np.log(truth["s_nu"])), "log_t_nu": float(np.log(truth["t_nu"]))}


TRUTH_SAMPLED = truth_sampled(DEFAULT_TRUTH)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_chrom_lengths(arg):
	"""'0.6,0.7,...' or a file of (name, length in Morgans) tuples -> (names, lengths)."""
	if os.path.isfile(arg):
		names, lens = [], []
		with open(arg) as fh:
			for ln_no, line in enumerate(fh, 1):
				s = line.strip()
				if not s or s.startswith("#"):
					continue
				s = s.strip("()[] ")
				parts = [p.strip().strip("'\"") for p in s.replace("\t", ",").replace(",", " ").split()]
				if len(parts) != 2:
					raise ValueError(f"{arg}, line {ln_no}: expected (name, length), got {line.strip()!r}")
				try:
					length = float(parts[1])
				except ValueError:
					raise ValueError(f"{arg}, line {ln_no}: length {parts[1]!r} is not a number") from None
				if length <= 0:
					raise ValueError(f"{arg}, line {ln_no}: length must be positive")
				names.append(parts[0]); lens.append(length)
		if not lens:
			raise ValueError(f"{arg}: no chromosomes found")
		return names, np.array(lens)
	lens = np.array([float(v) for v in arg.split(",") if v.strip()])
	if lens.size == 0 or np.any(lens <= 0):
		raise ValueError("--chrom-lengths: give positive lengths in Morgans, comma-separated, or a file")
	return [str(c + 1) for c in range(len(lens))], lens


def add_arguments(ap):
	ap.add_argument("--parents", type=int, default=1500)
	ap.add_argument("--meioses", type=int, default=3,
	                help="meiotic products per parent per chromosome (a meiotic product is one "
	                     "chromosome in one individual)")
	ap.add_argument("--chrom-lengths", default=None,
	                help="comma-separated lengths in Morgans, or a file of (name, length) lines "
	                     "(default: 22 chromosomes evenly spaced from 0.6 to 2.8 M)")
	for k, v in DEFAULT_TRUTH.items():
		if k in ("b_p", "b_nu"):
			ap.add_argument(f"--{k.replace('_', '-')}", default=str(v),
			                help=f"comma-separated coefficients, one per covariate (default {v:.4g}); "
			                     f"the number given for --b-p sets the number of covariates K"
			                     + ("; a single --b-nu value is repeated K times" if k == "b_nu" else ""))
		else:
			ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v, help=f"default {v:.4g}")
	ap.add_argument("--no-chrom-effects", action="store_true", help="set all chromosome effects to 0")
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--out", default="simulated_data.npz")


def run(a):
	if a.parents < 1 or a.meioses < 1:
		raise CommandError("--parents and --meioses must be at least 1")
	for k in ("s_p", "t_p", "s_nu", "t_nu"):
		if getattr(a, k) < 0:
			raise CommandError(f"--{k.replace('_', '-')} must be non-negative")
	if a.chrom_lengths is None:
		names, lens = [str(c + 1) for c in range(len(DEFAULT_CHROM_LEN))], DEFAULT_CHROM_LEN
	else:
		try:
			names, lens = parse_chrom_lengths(a.chrom_lengths)
		except ValueError as e:
			raise CommandError(str(e)) from None
	truth = {k: getattr(a, k) for k in DEFAULT_TRUTH}
	try:
		for k in ("b_p", "b_nu"):
			truth[k] = np.array([float(v) for v in str(truth[k]).split(",") if v.strip()])
		truth["b_p"], truth["b_nu"] = coefficient_vectors(truth)
	except ValueError as e:
		raise CommandError(f"--b-p / --b-nu: {e}") from None
	d = simulate_dataset(a.parents, a.meioses, lens, truth, a.seed, not a.no_chrom_effects, names)
	print(summarize_dataset(d))
	save_dataset(a.out, d, truth, a.seed, not a.no_chrom_effects)
	print(f"\nwritten to {a.out}")
	return 0
