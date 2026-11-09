# hsmodel

The Housworth-Stahl (2003) two-pathway model of crossover interference,
extended with parent- and chromosome-level random effects and parent-level
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
                        [--a-p ...] [--b-p 0.4,-0.3] ... [--binary [--proportion 0.5]]
                        [--no-chrom-effects] [--seed 1] [--out sim.npz]
    hsmodel preprocess  -m MAP [-i CROSSOVERS | -] -o data.npz [--cM]
    hsmodel infer       --data data.npz [--covariates covariates.txt] [--coefficients age,2]
                        [--steps N] [--guide-type meanfield|lowrank] [--lr 0.02] [--rank 4]
                        [--seed 0] [--chrom-effects | --no-chrom-effects]
                        [--parent-effects | --no-parent-effects] [--batch 32]
                        [--exact] [--truncate-at K] [--standardize] [--out PREFIX]
    hsmodel nuts        --data data.npz [--covariates covariates.txt] [--coefficients ...]
                        [--parents K] [--chains 4] [--warmup 1000] [--samples 1000]
                        [--target-accept 0.9] [--max-tree-depth 8] [--svi-init [--svi-steps N]]
                        [--chrom-param noncentered|centered] [--k-terms 12] [--quad-nodes 64]
                        [--chain-method auto|sequential|vectorized|parallel] [--seed 0]
                        [--chrom-effects | --no-chrom-effects] [--parent-effects | --no-parent-effects]
                        [--standardize] [--out PREFIX]
    hsmodel inspect     FILE.npz [--out FILE]

* **simulate** draws a dataset from the model, prints a summary (with
  stem-and-leaf plots of crossovers per meiotic product and of inter-crossover
  distances) and writes it as `.npz`. Covariates are standard normal, or
  with `--binary` Bernoulli coded 0/1, with P(z = 1) given by `--proportion`
  (one value for all covariates, or one per covariate; default 0.5). Binary
  covariates are not standardized, so `infer` and `nuts` warn about them.
* **preprocess** reads tab-separated crossover data (`individual, sex,
  chromosome, [positions]`) and a sex-specific map (`sex, chromosome,
  length`), and writes the same `.npz` format with the same summary.
* **infer** fits the model to a `.npz` dataset plus a covariate file (first
  column `iid`, optional `#` header line). It writes the data summary and
  progress to stderr, a table of posterior summaries to stdout, and
  `PREFIX_params.npz` and `PREFIX_summary.png`. It ends by printing to stderr
  the implied nu and p at the intercept, 1 + exp(a_nu) and
  1 / (1 + exp(-a_p)): every covariate 0 and the random effects at their mean.
* **nuts** fits the same model to the same inputs with NUTS, using the whole
  dataset at every step. It is much slower than SVI (`--parents K` fits only
  the first K individuals). It writes the posterior table (with effective
  sample sizes and R-hat) to stdout, sampler diagnostics to stderr, and
  `PREFIX_samples.npz` and `PREFIX_summary.png` (including traces).
* **inspect** describes any `.npz` file written by the other commands,
  recognizing from its contents whether it holds a dataset (`simulate`,
  `preprocess`), an SVI fit (`infer`'s `PREFIX_params.npz`, or the
  `PREFIX_svi_params.npz` starting point of `nuts --svi-init`) or a NUTS fit
  (`PREFIX_samples.npz`). For a dataset it prints the data summary, a
  per-chromosome table and, for simulations, the true parameter values. For a
  fit it prints how the model was fitted (covariates and their
  standardization, which random effects, guide or sampler settings, the
  likelihood approximation), convergence (the -ELBO over the last steps, or
  divergences, R-hat and effective sample sizes), the posterior of the global
  parameters on the fitted and the natural scale, the chromosome effects, and
  the spread of the individual effects. For a fit, `--out FILE` also writes
  the table of global parameters on the fitted scale (posterior mean, SD and
  95% interval, plus n_eff and R-hat for NUTS) to FILE, tab-separated with a
  header line. For an SVI fit of simulated data, `infer` stores the true
  values in `PREFIX_params.npz`, and `inspect` shows them in a truth column
  (on screen and in the `--out` file) and beside the chromosome effects.
  Datasets and SVI fits are read without importing JAX.

For simulated data (`hsmodel simulate`), `infer` and `nuts` take the
covariate values from the data file, so no covariate file is needed. The
simulated covariates are named `z1, z2, ...`. The true values are added to
the tables and plots, except after `--standardize`, which changes the
coefficients' scale.

`--no-chrom-effects` drops the chromosome effects `c_p_c, c_nu_c` (and their
SDs `t_p, t_nu`). `--no-parent-effects` drops the individual-specific effects
`s_p e_p_i, s_nu e_nu_i`, so individuals then differ only through their
covariates.

By default the likelihood computes F* and G* by 64-node Gauss-Legendre
quadrature and keeps K = 20 terms of the sum in f*. `infer --exact` uses the
closed forms (incomplete gamma function) instead: about 6x slower, but correct
for very strong interference. Quadrature is accurate to ~1e-10 per meiotic
product up to nu ~ 64 on a 2.8 M chromosome, but its error grows to ~2e-4 at
nu = 150 and ~0.1 at nu = 300. `--truncate-at K` sets the number of terms; the
closed forms drop the omitted ~2^-K tail, so use K of about 60 with `--exact`.
(`nuts` has the corresponding `--k-terms` and `--quad-nodes`.)

`hsmodel COMMAND --help` lists every option.

Example:

    hsmodel preprocess -i crossovers.tsv -m map.tsv -o data.npz --cM
    hsmodel infer --data data.npz --covariates covariates.txt --standardize --out fit1

    hsmodel simulate --parents 200 --b-p 0.4,-0.3 --out sim.npz
    hsmodel infer --data sim.npz --out sim_svi
    hsmodel nuts --data sim.npz --parents 50 --chains 2 --warmup 400 --samples 400 --svi-init --out sim_nuts
    hsmodel inspect sim_svi_params.npz

## Package layout

| module | contents |
|---|---|
| `hsmodel.likelihood` | f*, F*, g*, G*; record likelihood; forward-backward escape posteriors; grouped data layouts |
| `hsmodel.data` | padded records, the `.npz` dataset format, text summaries |
| `hsmodel.simulate` | simulation (`hsmodel simulate`) |
| `hsmodel.preprocess` | empirical data and maps (`hsmodel preprocess`) |
| `hsmodel.summaries` | parameter names, priors and labels; posterior summaries of SVI guides and NUTS draws (no JAX) |
| `hsmodel.model` | the NumPyro model, mean-field and low-rank guides (re-exports the names and priors) |
| `hsmodel.batching` | `ParentBatcher`: fixed-shape minibatches of parents |
| `hsmodel.svi` | `fit` and posterior summaries |
| `hsmodel.inputs` | datasets and covariates for fitting (shared by `infer` and `nuts`) |
| `hsmodel.infer` | the `hsmodel infer` command |
| `hsmodel.nuts` | full-data NUTS model and sampler |
| `hsmodel.nuts_command` | the `hsmodel nuts` command |
| `hsmodel.inspect_command` | the `hsmodel inspect` command |
| `hsmodel.plots` | SVI and NUTS summary plots |

`simulate`, `preprocess` and `inspect` (except for NUTS fits) do not import JAX, and `nuts` imports it only after
setting the device count that `--chain-method parallel` needs.

`examples/` holds scripts that are not part of the command-line interface:
* `svi_demo.py` and `nuts_demo.py`: SVI and NUTS fits to the default simulated dataset
* `approx_accuracy.py` and `quadrature_accuracy.py`: studies of approximation error
* `compare_cointf.py`: checks the likelihood against the independent `cointf` package
  (https://github.com/aabiddanda/cointf); `tests/test_cointf.py` runs when it is installed
* `plot_nu_p_correlation.py`
* `kinship_model_sketch.py`: a sketch of a model with covariance of parent-level effects structured by genetic relatedness (interesting but statistically tricky and computationally demanding)
* `example_chromosomes.txt`: an example `--chrom-lengths` file

## Tests

    pytest                     # everything (~30 min on a 2-core CPU)
    pytest -m "not slow"       # skips the long SVI fits, the Monte Carlo calibration and the longer NUTS run

The tests cover the likelihood (against simulation, closed forms and
brute-force enumeration of labellings), the grouped likelihoods and their
gradients, minibatching, sex-specific map lengths, the model and guides, and
the five commands end to end.

## Credits

Developed by Andrew P Morgan with advice from Arjun Bidanna (http://aabiddanda.github.io/), in collaboration with Claude Code. The statistical model and numerical approximations therein are tested against simulations but your mileage may vary.