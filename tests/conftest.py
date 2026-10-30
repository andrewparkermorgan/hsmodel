"""Shared fixtures and helpers for the hsmodel tests."""
import subprocess
import sys

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)


def run_cli(args, stdin=None, cwd=None):
	"""Run `python -m hsmodel ARGS` and return the CompletedProcess."""
	return subprocess.run([sys.executable, "-m", "hsmodel"] + [str(a) for a in args], input=stdin,
	                      capture_output=True, text=True, cwd=cwd)


def draw_sites(model, data, *args, seed=0):
	"""Values for every latent site of a numpyro model, drawn from the prior.
	The model is traced with an empty dataset of the same shape, because the
	draws do not depend on the data and an eager pass over real data is slow."""
	from numpyro import handlers
	empty = {"count0": np.zeros(np.shape(data["count0"])), "groups": ()}
	if "L" in data:
		empty["L"] = data["L"]
	tr = handlers.trace(handlers.seed(model, seed)).get_trace(empty, *args)
	return {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed", False)}


@pytest.fixture(scope="session")
def ragged_data():
	"""Records from 30 parents with their own (nu, p): 1-4 meioses per parent and
	~15% of chromosomes missing, so groups are ragged."""
	from hsmodel.data import pad_records
	from hsmodel.simulate import simulate_product
	rng = np.random.default_rng(11)
	P = 30
	chrom_len = np.linspace(0.6, 2.8, 22)
	nu_i = rng.uniform(2.0, 30.0, P)
	p_i = rng.uniform(0.005, 0.3, P)
	xo, lens, par, chrom = [], [], [], []
	for i in range(P):
		for _ in range(rng.integers(1, 5)):
			for c, L in enumerate(chrom_len):
				if rng.random() < 0.15:
					continue
				xo.append(simulate_product(nu_i[i], p_i[i], L, rng))
				lens.append(L); par.append(i); chrom.append(c)
	X, NXO, LEN = pad_records(xo, lens)
	return dict(P=P, chrom_len=chrom_len, nu_i=nu_i, p_i=p_i, X=X, NXO=NXO, LEN=LEN,
	            par=np.array(par), chrom=np.array(chrom), rng=rng)
