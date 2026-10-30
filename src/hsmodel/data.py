"""
Crossover datasets: padded record arrays, the .npz file format shared by
`hsmodel simulate` and `hsmodel preprocess`, and the text summary both print.

A meiotic product is one chromosome in one individual: one record.

Dataset dict (one row per record unless stated):
  X [R, N]      sorted crossover positions in Morgans, padded with 0
  NXO [R]       number of crossovers
  LEN [R]       genetic length of the record's chromosome (Morgans)
  par [R]       parent (individual) index 0..P-1
  replicate [R] which of the parent's products on that chromosome
  chrom [R]     chromosome index 0..C-1
  chrom_len [C], chrom_names [C], P, meioses
Simulated data add z [P, K] and the true effects; empirical data add
parent_names [P], sex [P] (1 male, 2 female) and chrom_len_by_sex [2, C].

This module does not import JAX, so the simulate and preprocess commands
start quickly.
"""
import math

import numpy as np


def pad_records(xo_lists, lengths, max_n=None):
	"""List of per-record crossover position lists (Morgans) -> padded arrays
	(X [R, N], NXO [R], LEN [R])."""
	max_n = max_n or max(1, max(len(v) for v in xo_lists))
	X = np.zeros((len(xo_lists), max_n))
	NXO = np.zeros(len(xo_lists), dtype=np.int32)
	for r, v in enumerate(xo_lists):
		v = np.sort(np.asarray(v, dtype=float))[:max_n]
		X[r, : len(v)] = v
		NXO[r] = len(v)
	return X, NXO, np.asarray(lengths, dtype=float)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def save_dataset(path, d, truth=None, seed=None, chrom_effects=None):
	"""Write a dataset dict (plus, for simulations, the truth, seed and whether
	chromosome effects were simulated) to a compressed .npz file."""
	extra = {}
	if truth is not None:
		extra.update({f"truth_{k}": v for k, v in truth.items()})
	if seed is not None:
		extra["seed"] = seed
	if chrom_effects is not None:
		extra["chrom_effects"] = chrom_effects
	np.savez_compressed(path, **d, **extra)


def load_dataset(path):
	"""Read a file written by save_dataset back into the dataset dict (plus
	'truth', 'seed' and 'chrom_effects' if they were saved)."""
	f = np.load(path)
	d = {k: f[k] for k in f.files if not k.startswith("truth_")}
	for k in ("P", "meioses", "seed"):
		if k in d:
			d[k] = int(d[k])
	if "chrom_effects" in d:
		d["chrom_effects"] = bool(d["chrom_effects"])
	truth = {k[len("truth_"):]: (float(f[k]) if f[k].ndim == 0 else f[k])
	         for k in f.files if k.startswith("truth_")}
	if truth:
		d["truth"] = truth
	return d


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def stem_and_leaf(values, unit_label="", width=60, target_lines=(8, 30)):
	"""Text stem-and-leaf plot of non-negative values.

	The leaf unit is a power of ten chosen so the plot has a readable number of
	lines; stems are split into 2 or 5 lines each when there would otherwise be
	too few.  Leaves are truncated (not rounded), as is conventional.  When a
	line would hold more than `width` leaves, each printed leaf stands for
	about k observations (stated in the header): every line gets
	round(count / k) leaves (at least one if the line is not empty), shared
	among its digits in proportion to their counts, so line length stays
	proportional to frequency.  The exact count ends every line."""
	x = np.asarray(values, dtype=float)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return "  (no values)"
	lo, hi = float(x.min()), float(x.max())
	span = max(hi - lo, 1e-12)
	# leaf unit u = 10^k: one stem per 10u; aim for <= target_lines[1] stems
	k = math.floor(math.log10(span / target_lines[1])) if span > 0 else 0
	u = 10.0 ** k
	while (math.floor(hi / (10 * u)) - math.floor(lo / (10 * u)) + 1) > target_lines[1]:
		u *= 10
	if np.allclose(x, np.round(x)) and u < 1:     # integer data: leaf unit at least 1
		u = 1.0
	n_stems = math.floor(hi / (10 * u)) - math.floor(lo / (10 * u)) + 1
	split = 1
	for s in (2, 5):
		if n_stems * split < target_lines[0]:
			split = s
	scaled = np.floor(x / u + 1e-9).astype(np.int64)  # in leaf units, truncated
	stems, leaves = scaled // 10, scaled % 10
	per_line = 10 // split
	lines = []
	for st in range(int(stems.min()), int(stems.max()) + 1):
		for part in range(split):
			lo_leaf, hi_leaf = part * per_line, (part + 1) * per_line
			sel = (stems == st) & (leaves >= lo_leaf) & (leaves < hi_leaf)
			counts = np.bincount(leaves[sel], minlength=10)
			lines.append((st, part, counts))
	# trim empty lines at both ends
	while lines and lines[0][2].sum() == 0:
		lines.pop(0)
	while lines and lines[-1][2].sum() == 0:
		lines.pop()
	max_n = max(int(c.sum()) for _, _, c in lines)
	per = max(1, math.ceil(max_n / width))
	marks = {1: [""], 2: ["*", "."], 5: ["*", "t", "f", "s", "."]}[split]
	stem_w = max(len(str(st)) for st, _, _ in lines)
	head = [f"  leaf unit = {u:g}{(' ' + unit_label) if unit_label else ''}"
	        f"  (stem | leaves: value = (10 x stem + leaf) x {u:g})"]
	if per > 1:
		head.append(f"  each leaf represents about {per} observations; exact counts in brackets")

	def leaf_string(counts):
		total = int(counts.sum())
		if per == 1:
			return "".join(str(dg) * int(n) for dg, n in enumerate(counts))
		n_leaves = max(1 if total else 0, int(round(total / per)))
		share = counts * n_leaves / max(total, 1)          # largest-remainder allocation
		alloc = np.floor(share).astype(int)
		for dg in np.argsort(-(share - alloc), kind="stable")[: n_leaves - alloc.sum()]:
			alloc[dg] += 1
		return "".join(str(dg) * int(n) for dg, n in enumerate(alloc))

	out = []
	for st, part, counts in lines:
		out.append(f"  {st:>{stem_w}}{marks[part]:1s} | {leaf_string(counts)}  [{int(counts.sum())}]")
	return "\n".join(head + out)


def summarize_dataset(d, title="Simulated Housworth-Stahl data"):
	"""Text summary of a dataset, simulated or empirical (which adds per-parent
	sex and sex-specific maps).  One record = one meiotic product."""
	NXO, X, LEN = d["NXO"], d["X"], d["LEN"]          # one row per meiotic product
	gaps = [np.diff(X[r, :n]) for r, n in enumerate(NXO) if n >= 2]
	gaps_cm = 100 * np.concatenate(gaps) if gaps else np.array([])
	C = len(d["chrom_len"])
	P = int(d["P"])
	pad = 40
	lines = [title, f"  {'parents:':{pad}}{P:,}"]
	if "z" in d:
		K = 1 if np.ndim(d["z"]) == 1 else np.shape(d["z"])[1]
		lines.append(f"  {'parent covariates:':{pad}}{K}")
	if "sex" in d:
		n_m, n_f = int(np.sum(d["sex"] == 1)), int(np.sum(d["sex"] == 2))
		lines[-1] += f"  ({n_m:,} male, {n_f:,} female)"
	# map lengths: one map, or one per sex present
	if "chrom_len_by_sex" in d and "sex" in d:
		maps = [(name, d["chrom_len_by_sex"][k]) for k, name in ((0, "male"), (1, "female"))
		        if np.any(d["sex"] == k + 1)]
	else:
		maps = [("", d["chrom_len"])]
	desc = "; ".join(f"{(name + ' ') if name else ''}map {np.nansum(m):.2f} M, mean {np.nanmean(m):.3f} M"
	                 for name, m in maps)
	lines.append(f"  {'chromosomes:':{pad}}{C}  ({desc})")
	# meiotic products per parent and chromosome
	per_pc = np.bincount(d["par"] * C + d["chrom"], minlength=P * C)
	per_pc = per_pc[per_pc > 0]
	if per_pc.size and per_pc.min() == per_pc.max():
		n_pc = f"{per_pc.min()} per parent per chromosome"
	else:
		n_pc = f"{per_pc.min()}-{per_pc.max()} per parent per chromosome typed"
	lines += [
		f"  {'meioses (meiotic products):':{pad}}{len(NXO):,}  ({n_pc})",
		f"  {'crossovers (COs):':{pad}}{int(NXO.sum()):,}",
		f"  {'mean COs per meiotic product:':{pad}}{NXO.mean():.3f}"
		f"  (expected from the products' map lengths: {LEN.mean():.3f})",
		"",
		f"COs per meiotic product (n = {len(NXO):,}):",
		stem_and_leaf(NXO, "CO"),
		"",
		f"Distance between consecutive COs on the same meiotic product (n = {len(gaps_cm):,} pairs):",
		stem_and_leaf(gaps_cm, "cM"),
	]
	return "\n".join(lines)
