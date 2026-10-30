"""
Convert empirical crossover data into the dataset format written by
`hsmodel simulate` (`hsmodel preprocess`).

A meiotic product is one chromosome in one individual: one line of the
crossover file, and one record of the output.

Inputs (tab-separated text):
  crossover file (-i), one meiotic product per line:
      individual <TAB> sex <TAB> chromosome <TAB> [pos1,pos2,...]
    individual and chromosome identifiers are text; sex is 1/m/M (male) or
    2/f/F (female); positions are a bracketed, comma-separated list ('[]' for
    no crossovers), in Morgans (or cM with --cM).
  map file (-m), one line per sex and chromosome:
      sex <TAB> chromosome <TAB> genetic length
    in Morgans (or cM with --cM).
  Blank lines and lines starting with # are ignored in both files.

Each meiotic product gets the genetic length of its chromosome in the map for
its individual's sex.  Output (.npz, readable with hsmodel.data.load_dataset)
has the simulator's fields - X, NXO, LEN, par, replicate, chrom, chrom_len,
chrom_names, P - plus parent_names, sex (per parent: 1 male, 2 female) and
chrom_len_by_sex [2, C] (row 0 male, row 1 female; NaN where a sex has no map
for that chromosome).  Fields that only exist for simulated data (z, true
effects, truth, seed) are absent.

chrom_len (one length per chromosome) is that sex's map when the data contain
one sex, and NaN when they contain both.  To fit both sexes together, give the
likelihood per-parent lengths looked up by sex:
    L = hsmodel.likelihood.parent_chrom_lengths(d["chrom_len_by_sex"], d["sex"])   # [P, C]
    group_by_count(..., parent_lengths=L)       # NUTS / full-data likelihood
    ParentBatcher(..., parent_lengths=L)        # SVI
(`hsmodel infer` does this automatically.)  To fit one sex on its own,
select_sex(d, "m" or "f") returns its records with chrom_len set to that sex's map.

Usage:
    hsmodel preprocess -m MAP [-i CROSSOVERS | -] -o OUT.npz [--cM]
"""
import sys

import numpy as np

from .data import pad_records, save_dataset, summarize_dataset
from .errors import InputError

MALE, FEMALE = 1, 2
_SEX_CODES = {"1": MALE, "m": MALE, "M": MALE, "2": FEMALE, "f": FEMALE, "F": FEMALE}


def parse_sex(text, where):
	code = _SEX_CODES.get(text.strip())
	if code is None:
		raise InputError(f"{where}: sex must be 1/m/M (male) or 2/f/F (female), got {text.strip()!r}")
	return code


def _data_lines(fh, name):
	"""(line number, fields) for every non-blank, non-comment line."""
	for ln, line in enumerate(fh, 1):
		s = line.rstrip("\r\n")
		if not s.strip() or s.lstrip().startswith("#"):
			continue
		yield ln, s.split("\t")


def read_map(fh, name, scale=1.0):
	"""{(sex, chromosome): length in Morgans}, and chromosome names in file order."""
	lengths, order = {}, []
	for ln, f in _data_lines(fh, name):
		where = f"{name}, line {ln}"
		if len(f) != 3:
			raise InputError(f"{where}: expected 3 tab-separated fields (sex, chromosome, length), got {len(f)}")
		sex = parse_sex(f[0], where)
		chrom = f[1].strip()
		if not chrom:
			raise InputError(f"{where}: empty chromosome identifier")
		try:
			length = float(f[2]) * scale
		except ValueError:
			raise InputError(f"{where}: length {f[2].strip()!r} is not a number") from None
		if not np.isfinite(length) or length <= 0:
			raise InputError(f"{where}: length must be positive, got {f[2].strip()!r}")
		if (sex, chrom) in lengths:
			raise InputError(f"{where}: second length for sex {sex}, chromosome {chrom!r}")
		lengths[(sex, chrom)] = length
		if chrom not in order:
			order.append(chrom)
	if not lengths:
		raise InputError(f"{name}: no map entries found")
	return lengths, order


def parse_positions(text, where, scale=1.0):
	s = text.strip()
	if not (s.startswith("[") and s.endswith("]")):
		raise InputError(f"{where}: crossover positions must be a bracketed list like [1.0,5.0] or [], "
		                 f"got {s!r}")
	inner = s[1:-1].strip()
	if not inner:
		return np.array([])
	try:
		pos = np.array([float(v) for v in inner.split(",")]) * scale
	except ValueError:
		raise InputError(f"{where}: could not read the crossover positions {s!r}") from None
	if not np.all(np.isfinite(pos)):
		raise InputError(f"{where}: crossover positions must be finite numbers, got {s!r}")
	return pos


def read_crossovers(fh, name, maps, map_order, scale=1.0):
	"""Parse the crossover file and build the dataset dict (see hsmodel.data)."""
	parent_idx, parent_names, parent_sex = {}, [], []
	records, lens, par, chrom_ids, sexes = [], [], [], [], []
	n_unsorted, n_coincident = 0, 0
	for ln, f in _data_lines(fh, name):
		where = f"{name}, line {ln}"
		if len(f) != 4:
			raise InputError(f"{where}: expected 4 tab-separated fields (individual, sex, chromosome, "
			                 f"positions), got {len(f)}")
		ind, chrom = f[0].strip(), f[2].strip()
		if not ind or not chrom:
			raise InputError(f"{where}: empty individual or chromosome identifier")
		sex = parse_sex(f[1], where)
		if ind in parent_idx:
			i = parent_idx[ind]
			if parent_sex[i] != sex:
				sexname = {MALE: "male", FEMALE: "female"}
				raise InputError(f"{where}: individual {ind!r} is {sexname[sex]} here but "
				                 f"{sexname[parent_sex[i]]} earlier in the file")
		else:
			i = parent_idx[ind] = len(parent_names)
			parent_names.append(ind); parent_sex.append(sex)
		L = maps.get((sex, chrom))
		if L is None:
			raise InputError(f"{where}: no {'male' if sex == MALE else 'female'} map length for "
			                 f"chromosome {chrom!r}")
		pos = parse_positions(f[3], where, scale)
		if np.any(np.diff(pos) < 0):
			n_unsorted += 1
			pos = np.sort(pos)
		if np.any(pos < 0) or np.any(pos > L):
			raise InputError(f"{where}: crossover position(s) {pos[(pos < 0) | (pos > L)].tolist()} M lie "
			                 f"outside the chromosome (0 to {L:g} M in the {'male' if sex == MALE else 'female'} "
			                 f"map)")
		n_coincident += int(np.sum(np.diff(pos) == 0))
		records.append(pos); lens.append(L); par.append(i); chrom_ids.append(chrom); sexes.append(sex)
	if not records:
		raise InputError(f"{name}: no meiotic products found")

	# chromosomes present in the data, in map-file order
	present = set(chrom_ids)
	chrom_names = [c for c in map_order if c in present]
	cidx = {c: k for k, c in enumerate(chrom_names)}
	chrom = np.array([cidx[c] for c in chrom_ids])
	par = np.array(par)
	# replicate: which of the parent's products on that chromosome (order of appearance)
	replicate = np.zeros(len(par), dtype=int)
	seen = {}
	for r, key in enumerate(zip(par, chrom)):
		replicate[r] = seen.get(key, 0)
		seen[key] = replicate[r] + 1
	X, NXO, LEN = pad_records(records, lens)
	C = len(chrom_names)
	by_sex = np.full((2, C), np.nan)
	for (sx, c), L in maps.items():
		if c in cidx:
			by_sex[sx - 1, cidx[c]] = L
	parent_sex = np.array(parent_sex)
	sexes_present = sorted(set(parent_sex.tolist()))
	if len(sexes_present) == 1:
		chrom_len = by_sex[sexes_present[0] - 1].copy()
	else:
		chrom_len = np.full(C, np.nan)
	d = dict(X=X, NXO=NXO, LEN=LEN, par=par, replicate=replicate, chrom=chrom, chrom_len=chrom_len,
	         chrom_names=np.array(chrom_names), P=len(parent_names), parent_names=np.array(parent_names),
	         sex=parent_sex, chrom_len_by_sex=by_sex,
	         meioses=int(np.bincount(par * C + chrom).max()))
	notes = []
	if n_unsorted:
		notes.append(f"{n_unsorted:,} meiotic product(s) listed crossovers out of order; they were sorted")
	if n_coincident:
		notes.append(f"{n_coincident:,} pair(s) of crossovers share an identical position")
	return d, notes


def select_sex(d, sex):
	"""The records of one sex ("m"/"f"/1/2), with chrom_len set to that sex's map,
	ready for the fitting code.  Parents are renumbered 0..P'-1."""
	code = parse_sex(str(sex), "select_sex")
	keep_par = np.flatnonzero(d["sex"] == code)
	if keep_par.size == 0:
		raise ValueError(f"no {'male' if code == MALE else 'female'} individuals in the data")
	new_idx = np.full(int(d["P"]), -1)
	new_idx[keep_par] = np.arange(keep_par.size)
	rec = new_idx[d["par"]] >= 0
	out = dict(d)
	for k in ("X", "NXO", "LEN", "replicate", "chrom"):
		out[k] = d[k][rec]
	out["par"] = new_idx[d["par"][rec]]
	out["P"] = int(keep_par.size)
	out["parent_names"] = d["parent_names"][keep_par]
	out["sex"] = d["sex"][keep_par]
	out["chrom_len"] = d["chrom_len_by_sex"][code - 1].copy()
	return out


def add_arguments(ap):
	ap.add_argument("-i", "--input", default="-",
	                help="crossover file (individual, sex, chromosome, [positions]); '-' or omitted: stdin")
	ap.add_argument("-m", "--map", required=True,
	                help="sex-specific chromosome lengths (sex, chromosome, length)")
	ap.add_argument("-o", "--output", required=True, help="output .npz file")
	ap.add_argument("--cM", action="store_true",
	                help="crossover positions and map lengths are in centimorgans (default: Morgans)")


def run(a):
	scale = 0.01 if a.cM else 1.0
	try:
		with open(a.map) as fh:
			maps, order = read_map(fh, a.map, scale)
		if a.input in (None, "-"):
			d, notes = read_crossovers(sys.stdin, "<stdin>", maps, order, scale)
		else:
			with open(a.input) as fh:
				d, notes = read_crossovers(fh, a.input, maps, order, scale)
	except (InputError, OSError) as e:
		print(f"error: {e}", file=sys.stderr)
		return 1
	print(summarize_dataset(d, title="Empirical crossover data"))
	for n in notes:
		print(f"\nnote: {n}")
	if np.any(np.isnan(d["chrom_len"])):
		print("\nnote: the data contain both sexes, so chrom_len is NaN; hsmodel infer looks up each "
		      "individual's chromosome lengths by sex (hsmodel.likelihood.parent_chrom_lengths)")
	save_dataset(a.output, d)
	print(f"\nwritten to {a.output}")
	return 0
