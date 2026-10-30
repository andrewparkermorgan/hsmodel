"""
Host-side minibatches of parents with fixed array shapes, for SVI.
"""
import numpy as np
import jax
import jax.numpy as jnp


class ParentBatcher:
	"""Draws minibatches of `batch` parents and packs their records in the
	hsmodel.likelihood.group_by_count layout, with parent indices local to the batch.

	For every crossover count n the batch holds a fixed number of records
	(capacity), padded with dummy records (mask 0), so array shapes never
	change.  Capacities are the largest count seen over `n_probe` random
	partitions, plus `slack`; an epoch whose partition would exceed them is
	simply redrawn.  Parents left over when P is not a multiple of `batch` sit
	out that epoch (a different random set each time).

	parent_lengths: optional [P, C] per-parent chromosome lengths (e.g. by sex,
	from hsmodel.likelihood.parent_chrom_lengths); each batch then carries its parents' rows
	as data["L"], which the likelihood uses instead of chrom_len."""

	def __init__(self, X, NXO, chrom_idx, parent_idx, n_parents, chrom_len, batch,
	             seed=0, n_probe=50, slack=2, parent_lengths=None):
		X, NXO = np.asarray(X), np.asarray(NXO)
		chrom_idx, parent_idx = np.asarray(chrom_idx), np.asarray(parent_idx)
		self.P, self.B, self.C = n_parents, batch, len(chrom_len)
		if batch > n_parents:
			raise ValueError("batch is larger than the number of parents")
		chrom_len = np.asarray(chrom_len, dtype=float)
		self.parent_lengths = None
		if parent_lengths is not None:
			pl = np.asarray(parent_lengths, dtype=float)
			if pl.shape != (n_parents, self.C) or not np.all(np.isfinite(pl)):
				raise ValueError(f"parent_lengths must be finite with shape {(n_parents, self.C)}")
			self.parent_lengths = pl
			if not np.all(np.isfinite(chrom_len)):         # e.g. NaN for mixed-sex data
				chrom_len = pl.mean(axis=0)                  # placeholder; not used by the likelihood
		elif not np.all(np.isfinite(chrom_len)):
			raise ValueError("chrom_len has missing values: pass parent_lengths "
			                 "(hsmodel.likelihood.parent_chrom_lengths) for sex-specific maps")
		self.Lc = jnp.asarray(chrom_len)
		self.rng = np.random.default_rng(seed)
		self.ns = [int(n) for n in np.unique(NXO) if n > 0]
		# crossover-free records: counts per (parent, chromosome)
		self.count0 = np.zeros((self.P, self.C))
		z = NXO == 0
		np.add.at(self.count0, (parent_idx[z], chrom_idx[z]), 1.0)
		# records with n crossovers, sorted by parent; starts[i]:starts[i+1] are parent i's
		self.recs = {}
		self.n_per_parent = np.zeros((self.P, len(self.ns)), dtype=int)
		for k, n in enumerate(self.ns):
			sel = np.flatnonzero(NXO == n)
			order = sel[np.argsort(parent_idx[sel], kind="stable")]
			starts = np.searchsorted(parent_idx[order], np.arange(self.P + 1))
			self.recs[n] = (X[order, :n], chrom_idx[order], starts)
			self.n_per_parent[:, k] = np.diff(starts)
		# capacities from random partitions
		peak = np.zeros(len(self.ns), dtype=int)
		for _ in range(n_probe):
			peak = np.maximum(peak, self._batch_counts(self._partition()).max(axis=0))
		self.cap = peak + slack
		# dummy record: crossovers evenly spaced on the shortest chromosome (shortest
		# for any parent, so the positions are valid whichever parent it is assigned to)
		if self.parent_lengths is None:
			self._dummy_c = int(np.argmin(chrom_len))
			self._dummy_L = float(np.min(chrom_len))
		else:
			self._dummy_c = int(np.argmin(self.parent_lengths.min(axis=0)))
			self._dummy_L = float(self.parent_lengths.min())

	def _partition(self):
		nb = self.P // self.B
		return self.rng.permutation(self.P)[: nb * self.B].reshape(nb, self.B)

	def _batch_counts(self, part):
		return self.n_per_parent[part].sum(axis=1)                # [n_batches, len(ns)]

	def _build(self, ix):
		groups = []
		for k, n in enumerate(self.ns):
			Xn, cn, starts = self.recs[n]
			idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in ix])
			local = np.repeat(np.arange(self.B), self.n_per_parent[ix, k])
			pad = self.cap[k] - len(idx)
			dummy_x = (np.arange(1, n + 1) / (n + 1)) * self._dummy_L
			groups.append((
				np.concatenate([Xn[idx], np.tile(dummy_x, (pad, 1))]),
				np.concatenate([local, np.zeros(pad, dtype=int)]).astype(np.int32),
				np.concatenate([cn[idx], np.full(pad, self._dummy_c)]).astype(np.int32),
				np.concatenate([np.ones(len(idx)), np.zeros(pad)])))
		out = {"count0": self.count0[ix], "groups": tuple(groups)}
		if self.parent_lengths is not None:
			out["L"] = self.parent_lengths[ix]
		return out

	def epoch(self, max_tries=1000):
		"""Yields (ix [B], data) for every batch of one random partition."""
		for _ in range(max_tries):
			part = self._partition()
			if np.all(self._batch_counts(part) <= self.cap):
				break
		else:
			raise RuntimeError("could not draw a partition within capacity; increase slack")
		for ix in part:
			yield jnp.asarray(ix), jax.tree_util.tree_map(jnp.asarray, self._build(ix))

	def stream(self):
		while True:
			yield from self.epoch()

	def padding_summary(self):
		"""{n: (mean real records per batch, capacity)}"""
		real = self.n_per_parent.sum(axis=0) / (self.P / self.B)
		return {n: (float(r), int(c)) for n, r, c in zip(self.ns, real, self.cap)}
