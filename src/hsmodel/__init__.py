"""
hsmodel: the Housworth-Stahl two-pathway model of crossover interference,
with parent and chromosome random effects and parent-level covariates.

Modules
  likelihood   exact marginal likelihood (pathway labels summed out), F*, G*,
               forward-backward escape posteriors, grouped data layouts (JAX)
  data         padded records, the .npz dataset format, text summaries
  simulate     simulation under the hierarchical model (`hsmodel simulate`)
  preprocess   reading empirical crossover and map files (`hsmodel preprocess`)
  model        priors, the NumPyro model, mean-field and low-rank guides
  batching     host-side minibatches of parents for SVI
  svi          SVI fitting and posterior summaries
  infer        fitting a dataset plus a covariate file (`hsmodel infer`)
  nuts         full-data NUTS fit of the same model
  nuts_command fitting a dataset with NUTS (`hsmodel nuts`)
  inputs       datasets and covariates for fitting (shared by infer and nuts)
  plots        summary plots for SVI and NUTS fits

Importing hsmodel itself does not import JAX; the likelihood module enables
64-bit floats in JAX when it is imported.
"""
__version__ = "0.1.0"
