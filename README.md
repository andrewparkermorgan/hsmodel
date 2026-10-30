# hsmodel

The Housworth-Stahl (2003) two-pathway model of crossover interference,
extended with parent and chromosome random effects and parent-level
covariates:

    logit p_ic      = a_p  + z_i . b_p  + s_p  e_p_i  + c_p_c
    log(nu_ic - 1)  = a_nu + z_i . b_nu + s_nu e_nu_i + c_nu_c

Each meiotic product (one chromosome in one individual) gets its exact
likelihood, with the interfering/non-interfering label of every crossover
summed out by an O(n^2) forward recursion. The model is fitted with
stochastic variational inference (NumPyro), using minibatches of parents, or
with NUTS for full Bayesian inference on smaller datasets.

## Install

    pip install -e .            # from this directory; add [test] for pytest

This installs the `hsmodel` command, which can also be run as `python -m hsmodel`.

## Commands

    hsmodel simulate    [--parents 1500] [--meioses 3] [--chrom-lengths LIST|FILE]
                        [--a-p ...] [--b-p 0.4,-0.3] ... [--no-chrom-effects] [--seed 1] [--out sim.npz]
    hsmodel preprocess  -m MAP [-i CROSSOVERS | -] -o data.npz [--cM]
    hsmodel infer       --data data.npz [--covariates covariates.txt] [--coefficients age,2]
                        [--steps N] [--guide-type meanfield|lowrank] [--lr 0.02] [--rank 4]
                        [--seed 0] [--chrom-effects | --no-chrom-effects] [--batch 32]
                        [--standardize] [--out PREFIX]
    hsmodel nuts        --data data.npz [--covariates covariates.txt] [--coefficients ...]
                        [--parents K] [--chains 4] [--warmup 1000] [--samples 1000]
                        [--target-accept 0.9] [--max-tree-depth 8] [--svi-init [--svi-steps N]]
                        [--chrom-param noncentered|centered] [--k-terms 12] [--quad-nodes 64]
                        [--chain-method auto|sequential|vectorized|parallel] [--seed 0]
                        [--chrom-effects | --no-chrom-effects] [--standardize] [--out PREFIX]

* **simulate** draws a dataset from the model, prints a summary (with
  stem-and-leaf plots of crossovers per meiotic product and of inter-crossover
  distances) and writes it as `.npz`.
* **preprocess** reads tab-separated crossover data (`individual, sex,
  chromosome, [positions]`) and a sex-specific map (`sex, chromosome,
  length`), and writes the same `.npz` format with the same summary.
* **infer** fits the model to a `.npz` dataset plus a covariate file (first
  column `iid`, optional `#` header line). It writes the data summary and
  progress to stderr, a table of posterior summaries to stdout, and
  `PREFIX_params.npz` and `PREFIX_summary.png`.
* **nuts** fits the same model to the same inputs with NUTS, using the whole
  dataset at every step. It is much slower than SVI (`--parents K` fits only
  the first K individuals). It writes the posterior table (with effective
  sample sizes and R-hat) to stdout, sampler diagnostics to stderr, and
  `PREFIX_samples.npz` and `PREFIX_summary.png` (including traces).

For simulated data (`hsmodel simulate`), `infer` and `nuts` take the
covariate values from the data file, so no covariate file is needed. The
simulated covariates are named `z1, z2, ...`. The true values are added to
the tables and plots, except after `--standardize`, which changes the
coefficients' scale.

`hsmodel COMMAND --help` lists every option.

Example:

    hsmodel preprocess -i crossovers.tsv -m map.tsv -o data.npz --cM
    hsmodel infer --data data.npz --covariates covariates.txt --standardize --out fit1

    hsmodel simulate --parents 200 --b-p 0.4,-0.3 --out sim.npz
    hsmodel infer --data sim.npz --out sim_svi
    hsmodel nuts --data sim.npz --parents 50 --chains 2 --warmup 400 --samples 400 --svi-init --out sim_nuts

## Package layout

| module | contents |
|---|---|
| `hsmodel.likelihood` | f*, F*, g*, G*; record likelihood; forward-backward escape posteriors; grouped data layouts |
| `hsmodel.data` | padded records, the `.npz` dataset format, text summaries |
| `hsmodel.simulate` | simulation (`hsmodel simulate`) |
| `hsmodel.preprocess` | empirical data and maps (`hsmodel preprocess`) |
| `hsmodel.model` | priors, the NumPyro model, mean-field and low-rank guides |
| `hsmodel.batching` | `ParentBatcher`: fixed-shape minibatches of parents |
| `hsmodel.svi` | `fit` and posterior summaries |
| `hsmodel.inputs` | datasets and covariates for fitting (shared by `infer` and `nuts`) |
| `hsmodel.infer` | the `hsmodel infer` command |
| `hsmodel.nuts` | full-data NUTS model and sampler |
| `hsmodel.nuts_command` | the `hsmodel nuts` command |
| `hsmodel.plots` | SVI and NUTS summary plots |

`simulate` and `preprocess` do not import JAX, and `nuts` imports it only after
setting the device count that `--chain-method parallel` needs.

`examples/` holds scripts that are not part of the command-line interface:
* `svi_demo.py` and `nuts_demo.py`: SVI and NUTS fits to the default simulated dataset
* `approx_accuracy.py` and `quadrature_accuracy.py`: studies of approximation error
* `plot_nu_p_correlation.py`
* `kinship_model_sketch.py`: a sketch of a kinship-structured model
* `example_chromosomes.txt`: an example `--chrom-lengths` file

## Tests

    pytest                     # everything (~22 min on a 2-core CPU)
    pytest -m "not slow"       # ~14 min; skips the full SVI fit, the Monte Carlo calibration and the longer NUTS run

The tests cover the likelihood (against simulation, closed forms and
brute-force enumeration of labellings), the grouped likelihoods and their
gradients, minibatching, sex-specific map lengths, the model and guides, and
the four commands end to end.
