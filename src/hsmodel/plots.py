"""
Summary plots: prior vs. posterior for each global parameter, chromosome
effects, and the SVI loss trace (plot_summary) or NUTS traces
(plot_nuts_summary).  Matplotlib is imported only when a plot is drawn.
"""
import numpy as np

from .model import GLOBALS, CHROM_SITES, PRIORS, LABELS, global_entries, entry_value, expand_truth
from .svi import chrom_summary
from .nuts import global_summary, chrom_draw_summary


STYLE = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
             axis="#c3c2b7", post="#2a78d6", svi="#eb6834")
CHROM_LABELS = {"c_p": "chromosome effects on logit p", "c_nu": "chromosome effects on log(nu - 1)"}


def _setup_matplotlib():
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	plt.rcParams.update({"font.size": 9, "axes.edgecolor": STYLE["axis"], "axes.labelcolor": STYLE["ink2"],
	                     "xtick.color": STYLE["muted"], "ytick.color": STYLE["muted"], "text.color": STYLE["ink"]})
	return plt


def param_layout(entries):
	"""Grid positions (row, col) for the global panels, keyed by display key:
	escape-fraction parameters on row 0, interference parameters on row 1.
	entries: from global_entries."""
	rows = [[key for key, site, _, _ in entries if site.endswith("_p")],
	        [key for key, site, _, _ in entries if site.endswith("_nu")]]
	return {key: (r, c) for r, row in enumerate(rows) for c, key in enumerate(row)}


def density_panel(ax, site, x_range_from, curves, truth=None, text="", show_ylabel=False, key=None, label=None):
	"""Prior (grey, filled) and posterior curves for one global, each scaled to
	a peak of 1.  curves: list of (x -> density, colour, linestyle, fill).
	site: the model site (for its prior); key, label: panel title lines."""
	S = STYLE
	pm, ps = PRIORS[site]
	lo = min(pm - 3.5 * ps, x_range_from[0])
	hi = max(pm + 3.5 * ps, x_range_from[1])
	x = np.sort(np.concatenate([np.linspace(lo, hi, 1500),
	                            np.linspace(x_range_from[2] - 6 * x_range_from[3],
	                                        x_range_from[2] + 6 * x_range_from[3], 600)]))
	prior = np.exp(-0.5 * ((x - pm) / ps) ** 2)
	ax.fill_between(x, prior, color=S["muted"], alpha=0.18, linewidth=0)
	ax.plot(x, prior, color=S["muted"], linewidth=1.5)
	for f, colour, ls, fill in curves:
		y = f(x); y = y / y.max()
		if fill:
			ax.fill_between(x, y, color=colour, alpha=0.22, linewidth=0)
		ax.plot(x, y, color=colour, linewidth=2 if fill else 1.6, linestyle=ls)
	top = 1.6
	if truth is not None:
		ax.axvline(truth, ymax=1.08 / top, color=S["ink"], linewidth=1.2, linestyle=(0, (1.5, 2)))
	ax.set_xlim(lo, hi); ax.set_ylim(0, top); ax.set_yticks([])
	for side in ("top", "right", "left"):
		ax.spines[side].set_visible(False)
	ax.set_title(f"{key or site}\n{label or LABELS[site]}", loc="left", fontsize=9, color=S["ink"], pad=6)
	left = (x_range_from[2] - lo) / (hi - lo) > 0.5          # annotate on the side away from the posterior
	ax.text(0.02 if left else 0.98, 0.98, text, transform=ax.transAxes,
	        ha="left" if left else "right", va="top", fontsize=7.5, color=S["ink2"])
	if show_ylabel:
		ax.set_ylabel("density (each scaled to peak 1)", fontsize=8, color=S["muted"])


def chrom_effect_panel(ax, site, est, truth=None, est2=None, labels=("posterior", "SVI"), chrom_names=None):
	"""Posterior mean and 95% interval of each chromosome's effect, in
	chromosome order, with the true values if known.  est, est2: (mean, lo, hi)."""
	S = STYLE
	m, lo, hi = est
	C = len(m)
	x = np.arange(1, C + 1)
	off = 0.18 if est2 is not None else 0.0
	ax.axhline(0, color=S["axis"], linewidth=1)
	ax.vlines(x - off, lo, hi, color=S["post"], linewidth=2)
	ax.plot(x - off, m, "o", color=S["post"], markersize=5, markeredgecolor=S["surface"], markeredgewidth=1,
	        label=f"{labels[0]} mean, 95% interval")
	if est2 is not None:
		m2, lo2, hi2 = est2
		ax.vlines(x + off, lo2, hi2, color=S["svi"], linewidth=2)
		ax.plot(x + off, m2, "s", color=S["svi"], markersize=4.5, markeredgecolor=S["surface"],
		        markeredgewidth=1, label=f"{labels[1]} mean, 95% interval")
	if truth is not None:
		ax.plot(x, truth, "D", markerfacecolor="none", markeredgecolor=S["ink"], markersize=5.5,
		        markeredgewidth=1.2, label="true value")
	ax.set_xlim(0.3, C + 0.7)
	if chrom_names is not None:
		ax.set_xticks(x)
		ax.set_xticklabels([str(n) for n in chrom_names], rotation=90 if C > 12 else 0, fontsize=7.5)
		ax.set_xlabel("chromosome")
	else:
		ax.set_xticks(x[::3] if C > 12 else x)
		ax.set_xlabel("chromosome (in the order given; demo: shortest to longest)")
	ax.grid(axis="y", color=S["grid"], linewidth=0.8); ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	# title on its own line above the legend row, so long legends cannot overprint it
	ax.set_title(CHROM_LABELS[site], loc="left", fontsize=9.5, color=S["ink"], pad=22)
	ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8,
	          borderaxespad=0.2, handletextpad=0.3, columnspacing=1.0)


def plot_summary(params, losses, path="svi_summary.png", truth=None, truth_chrom=None, title="Housworth-Stahl SVI fit",
                 covariate_names=None, chrom_names=None):
	"""Prior vs. posterior for each global parameter, the chromosome effects
	(if fitted), and the loss trace.

	Densities are Normal on the sampled (unconstrained) scale: priors from
	PRIORS, posteriors from the guide's loc/scale.  Each curve is scaled to a
	peak of 1, because posteriors are often ~100x narrower than priors and
	would otherwise flatten them; widths and locations along x are exact.
	truth: optional {site: value} on the same scale (dotted line; K-vectors for
	b_p, b_nu); truth_chrom: optional {"c_p": [C], "c_nu": [C]}."""
	plt = _setup_matplotlib()
	from matplotlib.lines import Line2D
	from matplotlib.patches import Patch
	S = STYLE
	names = [n for n in GLOBALS if f"{n}_loc" in params]
	K = np.atleast_1d(params["b_p_loc"]).size if "b_p_loc" in params else 1
	entries = global_entries(names, K, covariate_names)
	truth_x = expand_truth(truth, K, covariate_names)
	chrom = chrom_summary(params)
	pos = param_layout(entries)
	ncol = max(4, max(c for _, c in pos.values()) + 1)
	n_rows = 2 + (1 if chrom else 0) + 1
	fig = plt.figure(figsize=(3.75 * ncol, 4.0 * n_rows), facecolor=S["surface"])
	gs = fig.add_gridspec(n_rows, ncol, hspace=0.75, wspace=0.16)
	for key, site, k, label in entries:
		r, c = pos[key]
		ax = fig.add_subplot(gs[r, c], facecolor=S["surface"])
		qm, qs = entry_value(params[f"{site}_loc"], k), entry_value(params[f"{site}_scale"], k)
		txt = f"posterior {qm:.3f} \u00b1 {qs:.3f}"
		t = None if truth_x is None else truth_x.get(key)
		if t is not None:
			txt += f"\ntruth {t:.3f}"
		density_panel(ax, site, (qm - 4 * qs, qm + 4 * qs, qm, qs),
		              [(lambda x, m=qm, s=qs: np.exp(-0.5 * ((x - m) / s) ** 2), S["post"], "-", True)],
		              truth=t, text=txt, show_ylabel=(c == 0), key=key, label=label)
	row = 2
	if chrom:
		half = ncol // 2
		for k, site in enumerate(CHROM_SITES):
			ax = fig.add_subplot(gs[row, (0, half)[k]:(half, ncol)[k]], facecolor=S["surface"])
			chrom_effect_panel(ax, site, chrom[site], None if truth_chrom is None else truth_chrom.get(site),
			                   chrom_names=chrom_names)
		row += 1

	ax = fig.add_subplot(gs[row, :], facecolor=S["surface"])
	losses = np.asarray(losses)
	it = np.arange(1, len(losses) + 1)
	w = max(1, len(losses) // 50)
	running = np.convolve(losses, np.ones(w) / w, mode="valid")
	ax.plot(it, losses, color=S["muted"], linewidth=0.6, alpha=0.6)
	ax.plot(it[w - 1:], running, color=S["post"], linewidth=2)
	settled = losses[len(losses) // 10:] if len(losses) >= 20 else losses
	pad = 0.15 * (np.percentile(settled, 99.5) - np.percentile(settled, 0.5) + 1e-9)
	ax.set_ylim(np.percentile(settled, 0.5) - pad, np.percentile(settled, 99.5) + pad)
	ax.set_xlim(1, len(losses))
	ax.grid(axis="y", color=S["grid"], linewidth=0.8); ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.set_xlabel("SVI step"); ax.set_ylabel("loss (negative ELBO)")
	ax.set_title("Loss trace", loc="left", fontsize=9.5, color=S["ink"], pad=6)
	ax.legend([Line2D([], [], color=S["muted"], linewidth=1), Line2D([], [], color=S["post"], linewidth=2)],
	          ["per step (minibatch)", f"running mean ({w} steps)"],
	          loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
	clipped = int(np.sum(losses > ax.get_ylim()[1]))
	if clipped:
		ax.text(0.01, 0.97, f"{clipped} early steps above the plotted range", transform=ax.transAxes,
		        ha="left", va="top", fontsize=8, color=S["muted"])

	handles = [Patch(facecolor=S["muted"], alpha=0.4, edgecolor=S["muted"], label="prior"),
	           Patch(facecolor=S["post"], alpha=0.4, edgecolor=S["post"], label="posterior (variational)")]
	if truth is not None:
		handles.append(Line2D([], [], color=S["ink"], linewidth=1.2, linestyle=(0, (1.5, 2)), label="true value"))
	fig.suptitle(title, x=0.012, y=0.995, ha="left", fontsize=12, color=S["ink"])
	fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
	           bbox_to_anchor=(0.006, 0.98), fontsize=9)
	fig.savefig(path, dpi=150, facecolor=S["surface"], bbox_inches="tight")
	plt.close(fig)
	return path


def plot_nuts_summary(by_chain, n_divergent, path="nuts_summary.png", truth=None, svi_params=None,
                      truth_chrom=None, covariate_names=None, chrom_names=None, title="Housworth-Stahl NUTS fit"):
	"""Prior vs. NUTS posterior for each global (optionally the SVI posterior
	too), the chromosome effects (NUTS and optionally SVI, with the truth if
	known), and the trace of each global, one line per chain.
	by_chain: {name: [chains, draws, ...]} (mcmc.get_samples(group_by_chain=True),
	or the saved *_samples.npz); n_divergent: number of divergent transitions.
	Densities are on the sampled scale and each is scaled to a peak of 1 (see
	plot_summary); the NUTS posterior is a kernel density estimate."""
	plt = _setup_matplotlib()
	from matplotlib.lines import Line2D
	from matplotlib.patches import Patch
	from scipy.stats import gaussian_kde
	S = STYLE
	# chain colours skip orange, which marks the SVI posterior; order validated for CVD separation
	chain_c = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948"]

	names = [n for n in GLOBALS if n in by_chain]
	glob = {k: np.asarray(by_chain[k]) for k in names}
	n_chains, n_draws = glob["a_p"].shape[:2]
	K = glob["b_p"].shape[2] if glob["b_p"].ndim == 3 else 1
	entries = global_entries(names, K, covariate_names)
	truth_x = expand_truth(truth, K, covariate_names)

	def draws_of(site, k):                                       # [chains, draws]
		a = glob[site]
		return a if k is None or a.ndim == 2 else a[..., k]

	summ = global_summary(glob)
	chrom = chrom_draw_summary(by_chain)
	chrom_svi = chrom_summary(svi_params) if svi_params is not None else {}
	pos = param_layout(entries)
	ncol = max(4, max(c for _, c in pos.values()) + 1)
	n_rows = 2 + (1 if chrom else 0) + 2
	fig = plt.figure(figsize=(3.75 * ncol, 3.9 * n_rows), facecolor=S["surface"])
	gs = fig.add_gridspec(n_rows, ncol, hspace=0.8, wspace=0.16)

	for key, site, k, label in entries:
		r, c = pos[key]
		ax = fig.add_subplot(gs[r, c], facecolor=S["surface"])
		draws = draws_of(site, k).ravel()
		qm, qs = float(draws.mean()), float(draws.std())
		kde = gaussian_kde(draws)
		curves = [(kde, S["post"], "-", True)]
		r_hat, n_eff = entry_value(summ[site]["r_hat"], k), entry_value(summ[site]["n_eff"], k)
		txt = f"NUTS {qm:.3f} \u00b1 {qs:.3f}\nR-hat {r_hat:.3f}, ESS {n_eff:.0f}"
		lo_x, hi_x = qm - 5 * qs, qm + 5 * qs
		if svi_params is not None and f"{site}_loc" in svi_params:
			sm, ss = entry_value(svi_params[f"{site}_loc"], k), entry_value(svi_params[f"{site}_scale"], k)
			curves.append((lambda x, m=sm, s=ss: np.exp(-0.5 * ((x - m) / s) ** 2), S["svi"], (0, (4, 2)), False))
			txt += f"\nSVI {sm:.3f} \u00b1 {ss:.3f}"
			lo_x, hi_x = min(lo_x, sm - 4 * ss), max(hi_x, sm + 4 * ss)
		t = None if truth_x is None else truth_x.get(key)
		if t is not None:
			txt += f"\ntruth {t:.3f}"
		density_panel(ax, site, (lo_x, hi_x, qm, qs), curves, truth=t, text=txt, show_ylabel=(c == 0),
		              key=key, label=label)

	row = 2
	if chrom:
		half = ncol // 2
		for k, site in enumerate(CHROM_SITES):
			if site not in chrom:
				continue
			ax = fig.add_subplot(gs[row, (0, half)[k]:(half, ncol)[k]], facecolor=S["surface"])
			chrom_effect_panel(ax, site, chrom[site], None if truth_chrom is None else truth_chrom.get(site),
			                   est2=chrom_svi.get(site), labels=("NUTS", "SVI"), chrom_names=chrom_names)
		row += 1

	it = np.arange(1, n_draws + 1)
	for key, site, k, label in entries:
		r, c = pos[key]
		ax = fig.add_subplot(gs[row + r, c], facecolor=S["surface"])
		tr_draws = draws_of(site, k)
		for ch in range(n_chains):
			ax.plot(it, tr_draws[ch], color=chain_c[ch % len(chain_c)], linewidth=0.7, alpha=0.85)
		if truth_x is not None and key in truth_x:
			ax.axhline(truth_x[key], color=S["ink"], linewidth=1.0, linestyle=(0, (1.5, 2)))
		ax.set_xlim(1, n_draws)
		ax.grid(axis="y", color=S["grid"], linewidth=0.8); ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		ax.set_title(f"trace: {key}", loc="left", fontsize=9.5, color=S["ink"], pad=4)
		if r == 1:
			ax.set_xlabel("post-warmup draw")
	trace_top = 1 - row / n_rows
	fig.legend([Line2D([], [], color=chain_c[ch % len(chain_c)], linewidth=1.5) for ch in range(n_chains)],
	           [f"chain {ch + 1}" for ch in range(n_chains)], loc="lower right", ncol=n_chains,
	           frameon=False, fontsize=8.5, bbox_to_anchor=(0.985, trace_top - 0.012))

	handles = [Patch(facecolor=S["muted"], alpha=0.4, edgecolor=S["muted"], label="prior"),
	           Patch(facecolor=S["post"], alpha=0.4, edgecolor=S["post"], label="posterior (NUTS)")]
	if svi_params is not None:
		handles.append(Line2D([], [], color=S["svi"], linewidth=1.6, linestyle=(0, (4, 2)), label="posterior (SVI)"))
	if truth is not None:
		handles.append(Line2D([], [], color=S["ink"], linewidth=1.2, linestyle=(0, (1.5, 2)), label="true value"))
	fig.suptitle(f"{title}  ({n_chains} chains x {n_draws} draws, "
	             f"{n_divergent} divergent transitions)", x=0.012, y=0.995, ha="left", fontsize=12, color=S["ink"])
	fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
	           bbox_to_anchor=(0.006, 0.985), fontsize=9)
	fig.savefig(path, dpi=150, facecolor=S["surface"], bbox_inches="tight")
	plt.close(fig)
	return path
