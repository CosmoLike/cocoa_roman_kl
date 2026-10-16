# Covariance tests

These tests require the optional covariance build. Follow the
[project build instructions](../../README.md#computing_covariances): unset
`IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE` after activating Cocoa and rebuild.
With the default data-vector-only build this sector reports skips; the
separate `tests/data_vector` suite remains available.

The sector runs the shared
[project forecast check](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#cov_forecast_check)
against this project's galaxy/shear forecast adapter and compiled
covariance bindings, with this project's redshift files and catalog
inputs. The independent component references live in
[LSST Y1's covariance suite](https://github.com/CosmoLike/cocoa_lsst_y1/blob/main/tests/covariance/README.md);
use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. The
[data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately.

We assume Cocoa is installed, users have run `conda activate cocoa`, the
shell is Bash and the current folder is `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile this project's interface with covariance enabled.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE
    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE
    source projects/roman_kl/scripts/compile_roman_kl.sh

**Step :three:**: run the covariance checks.

    python -m pytest projects/roman_kl/tests/covariance
