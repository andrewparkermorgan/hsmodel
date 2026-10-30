"""
SVI fit of the hierarchical Housworth-Stahl model to the default simulated
dataset (1500 parents, 3 meioses, 22 chromosomes), compared with the truth.

Usage: python examples/svi_demo.py [meanfield|lowrank] [rank] [steps]

Writes svi_params.npz (guide parameters; examples/plot_nu_p_correlation.py
reads it), elbo_trace.npy and svi_summary.png (suffix _lowrank for the
low-rank guide).  For data on disk, use `hsmodel infer` instead.
"""
import sys

import numpy as np

from hsmodel.batching import ParentBatcher
from hsmodel.model import as_covariate_matrix, expand_truth
from hsmodel.plots import plot_summary
from hsmodel.simulate import simulate_dataset, TRUTH_SAMPLED
from hsmodel.svi import fit, summarize, chrom_summary, chrom_recovery


def main(argv):
	guide_type = argv[0] if len(argv) > 0 else "meanfield"
	rank = int(argv[1]) if len(argv) > 1 else 4
	steps = int(argv[2]) if len(argv) > 2 else 2500
	suffix = "" if guide_type == "meanfield" else f"_{guide_type}"
	print(f"guide: {guide_type}" + (f" (rank {rank})" if guide_type == "lowrank" else ""))
	print("simulating ...", flush=True)
	d = simulate_dataset()
	P = d["P"]
	print(f"  {len(d['NXO']):,} meiotic products (chromosome records), {d['NXO'].sum():,} crossovers, "
	      f"max/record {d['NXO'].max()}")
	batcher = ParentBatcher(d["X"], d["NXO"], d["chrom"], d["par"], P, d["chrom_len"], batch=32)

	# ~400 updates per parent (steps ~ 400 P / B = 18,750) are needed for the
	# between-parent SDs; the default 2,500 steps is a quick look.
	print(f"fitting ({steps:,} steps) ...", flush=True)
	params, losses = fit(batcher, d["z"], steps=steps, lr=0.02, log_every=max(1, steps // 10),
	                     guide_type=guide_type, rank=rank)

	print("\nposterior (mean, sd, 95% interval)  vs truth")
	trx = expand_truth(TRUTH_SAMPLED, as_covariate_matrix(d["z"]).shape[1])
	for n, (m, sd, lo, hi) in summarize(params).items():
		print(f"  {n:9s} {m:7.3f} ({sd:.3f})  [{lo:7.3f}, {hi:7.3f}]   truth {trx[n]:7.3f}")
	truth_chrom = {"c_p": d["c_p"], "c_nu": d["c_nu"]}
	for site, est in chrom_summary(params).items():
		r, rmse, cover = chrom_recovery(est, truth_chrom[site])
		print(f"  {site:5s} vs true chromosome effects: correlation {r:.2f}, RMSE {rmse:.3f}, "
		      f"95% intervals covering the truth {cover:.0%}")
	np.save(f"elbo_trace{suffix}.npy", losses)
	np.savez(f"svi_params{suffix}.npz", **params)
	plot = plot_summary(params, losses, f"svi_summary{suffix}.png", truth=TRUTH_SAMPLED, truth_chrom=truth_chrom)
	print(f"\nSVI parameters: svi_params{suffix}.npz; summary plot: {plot}")


if __name__ == "__main__":
	main(sys.argv[1:])
