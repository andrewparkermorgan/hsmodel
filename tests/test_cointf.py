"""Agreement with the independent implementation in the `cointf` package
(https://github.com/aabiddanda/cointf); skipped when it is not installed.
examples/compare_cointf.py prints the full comparison."""
from functools import partial

import numpy as np
import pytest
import jax
import jax.numpy as jnp

cointf = pytest.importorskip("cointf")

from hsmodel.data import pad_records
from hsmodel.likelihood import record_loglik
from hsmodel.simulate import simulate_product


@pytest.mark.parametrize("nu, p, L", [(10.0, 0.05, 1.0), (4.0, 0.2, 2.5), (30.0, 0.01, 1.8)])
def test_matches_cointf(nu, p, L):
	rng = np.random.default_rng(1)
	products = [simulate_product(nu, p, L, rng) for _ in range(100)]
	X, NXO, LEN = pad_records(products, [L] * 100)
	args = (jnp.asarray(X), jnp.asarray(NXO), jnp.asarray(LEN), nu, p)
	quad = np.asarray(jax.jit(jax.vmap(record_loglik, in_axes=(0, 0, 0, None, None)))(*args))
	exact = np.asarray(jax.jit(jax.vmap(partial(record_loglik, exact=True, K=200),
	                                    in_axes=(0, 0, 0, None, None)))(*args))
	fast = np.asarray(cointf.loglik_fast([list(x) for x in products], L, nu=nu, p=p, per_meiosis=True))
	naive = np.array([cointf.loglik_naive([list(x)], L, nu=nu, p=p) for x in products])
	assert np.max(np.abs(fast - naive)) < 1e-11
	for ours in (quad, exact):
		assert np.max(np.abs(ours - naive)) < 1e-11
