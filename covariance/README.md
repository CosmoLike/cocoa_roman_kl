# Table of contents

1. [Overview](#overview)
2. [Running from the command line](#command_line)
   - [Running the covariance notebook](#running)
3. [Changing the covariance accuracy](#accuracy)
4. [Reading the figures](#figures)
5. [Running the tests](#tests)
6. [Files](#files)
7. [Appendix](#appendix)
   1. [FAQ: Which survey does the example use?](#survey)
   2. [FAQ: What does the calculation include?](#gaussian)
   3. [FAQ: How can users check convergence?](#convergence)
   4. [FAQ: How can users reuse the calculation?](#reuse)
   5. [FAQ: Which accuracy settings are available?](#accuracy-settings)
   6. [Choosing the Gaussian spectra](#gaussian_spectra)

# Overview <a name="overview"></a>

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.** Both routes
> share the same C kernels and survey inputs. Notebook array conversion and
> rearrangement add overhead; use the CLI for production and HPC runs.

[EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
computes an analogous Fourier-space 3×2pt covariance for
[the supplied dataset](../data/roman_kl_3x2.dataset). It keeps Gaussian (G),
super-sample (SSC), connected non-Gaussian (cNG) and total matrices separately.
The example has 2,200 entries before cuts and 2,200 after the dataset mask.
The shared reader applies that mask to both axes of every component and to
the supplied total; the notebook plots their correlation matrices together.

The primary example is the 20-band Fourier 3×2pt configuration in
`roman_kl_3x2.dataset`, also used by `EXAMPLE_EVALUATE2.yaml`. It has ten lens
and ten source bins and retains only galaxy–shear pairs with source index
greater than lens index. The low shape dispersion is the kinematic-lensing
forecast choice. The cosmic-shear-only `roman_kl_mcmc.dataset` is a different
configuration.

The default computes the native measurement at boost 1. Users can request a
second boost for numerical comparisons and a companion measurement space.
The supplied matrix is read only for comparison; no likelihood files are changed.

> [!NOTE]
> The galaxy/shear Gaussian calculation supports non-Limber clustering
> and galaxy–shear spectra, plus NLA or TATT intrinsic alignment.
> Shear–shear and higher-order TATT spectra remain Limber. SSC/cNG retain
> their zero-IA Limber model. The forecast uses massless neutrinos, linear
> galaxy bias, zero magnification/RSD and a spherical-cap footprint.
> These physical choices differ from the supplied likelihood matrices.
> Matching their measurement layout does not establish physical or numerical
> equivalence.

Matter-halo integrals use **10⁻⁴⁰ to 10¹⁷ solar masses/h**, with the
shared `halo_mass_edges()` panels. Wynn extrapolation estimates the remaining
low-mass tail of I11; a residual completion preserves its unit response at
zero wavenumber. Higher moments use their converged direct integrals.
The fitted halo bias and multiplicity conventions are unchanged. See the
[core halo-moment explanation](https://github.com/CosmoLike/cocoa-cosmolike-core/tree/bugfix/cosmolike/covariances#why-only-i11-uses-wynn-extrapolation).

# Running from the command line <a name="command_line"></a>

The Python runner computes the full galaxy–shear covariance in Fourier space,
using the optimized production interface. It saves G, SSC, cNG and their
sum without plotting or opening a notebook. Numerical kernels and survey
settings are shared with the notebook calculation.

The [production timing table](https://github.com/CosmoLike/cocoa)
reports **35.3 seconds** for this project on an Apple M2 Pro with
eight OpenMP threads (mean of three sequential CLI runs).

This interval includes first-use CosmoLike tables, spectra, halo
calculations, transforms and complete G + SSC + cNG matrix assembly.
Initial survey/CAMB setup, diagnostics and file writing are excluded.

From Bash in `cocoa/Cocoa`, with `conda activate cocoa`:

**Step :one:**: activate Cocoa.

    source start_cocoa.sh

**Step :two:**: enable the installed project.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE

**Step :three:**: enable covariance generation.

    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE

**Step :four:**: compile the project interface.

    source ./projects/roman_kl/scripts/compile_roman_kl.sh

**Step :five:**: select the OpenMP team size.

    export OMP_NUM_THREADS=8

**Step :six:**: run the CLI.

    python ./projects/roman_kl/covariance/compute_covariance.py \
        ./projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.yaml

The `.npz` archive contains the full matrix before likelihood scale cuts,
its components, measurement ordering, resolved settings and stage timings.
Existing output files require `--overwrite`; likelihood inputs are separate.

Set `covariance.space` to `real` or `fourier` to select the measurement.

The [evaluate YAML](../EXAMPLE_EVALUATE_COVARIANCE.yaml) uses Cobaya's YAML reader, with familiar
`theory`, `params`, `sampler: evaluate` and `output` blocks. Fixed parameter
values specify one cosmology; a parameter with a prior must be supplied
explicitly in `sampler.evaluate.override`. No MCMC or random prior draw runs.

Its `covariance` block keeps the project's `default.yaml` baseline,
`accuracy_boost: 1` and `integration_accuracy: 0`. A larger
`accuracy_boost`, such as 2, refines that baseline; `integration_accuracy`
changes the quadrature level independently. Internal accuracy controls can
also be set there.
Use `space` for the measurement space. Set the OpenMP team with
`OMP_NUM_THREADS` in the shell; no thread count belongs in the YAML.

`theory.camb.extra_args` supports `AccuracyBoost`, `kmax`, `k_per_logint`,
`lens_potential_accuracy` and `halofit_version`. CAMB's boost controls
CAMB; the covariance boost controls its own tables and cutoffs.

Paths in the YAML are relative to the working directory, `cocoa/Cocoa`.
`output` names the `.npz` archive; `--output` can override it for an HPC
job. Set `OMP_NUM_THREADS` in that job’s environment. `--help` lists the
command options.

# Running the covariance notebook <a name="running"></a>

The default build omits covariance generation. Unset
`IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE` after activating Cocoa, then recompile
as below. Likelihood evaluation with a supplied covariance remains available
in either build. Restart the Jupyter kernel after a rebuild. To retain this
choice across sessions, comment out the matching export in
[`set_installation_options.sh`](https://github.com/CosmoLike/cocoa/blob/main/Cocoa/set_installation_options.sh).

We assume Cocoa and the Roman kinematic lensing project are installed, users have run
`conda activate cocoa`, the shell is Bash, and the current folder is
`cocoa/Cocoa`. The notebook uses the Python environment activated by Cocoa.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable the installed project.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE

**Step :three:**: enable covariance generation.

    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE

**Step :four:**: compile the interface.

    source ./projects/roman_kl/scripts/compile_roman_kl.sh

**Step :five:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :six:**: open the URL printed by Jupyter and select
`projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :seven:**: select **Kernel → Restart Kernel and Run All Cells**.

The notebook starts with `boosts = [1]` and `spaces = ["fourier"]`.
It computes the native matrix, applies the selected dataset's mask, reports
positivity and plots the computed components and supplied total. Set
`boosts = [1, 2]` to add the numerical-refinement comparison.

The **See the 1h, 2h, 3h and 4h matter trispectra** section can be run
immediately after initialization, before computing the full covariance.
It plots the diagonal matter trispectrum at `halo_redshift = 0.5`, with
the two-halo term combining its 1+3 and 2+2 partitions. Both partitions
and all sampled wavenumber pairs remain available as notebook arrays.
These are matter terms before survey projection; G and SSC are separate.

| Output in `covariance/` | Contents |
| --- | --- |
| `forecast_fourier.npz` | Full computed G, SSC, cNG, total, ordering and settings. |
| `forecast_likelihood_selection.npz` | Cut components, supplied total, original entry indices and probe labels. |
| `forecast_camb.npz` | CAMB tables used by the native calculation. |

Rerunning the final cell replaces these computed output files.

> [!NOTE]
> The notebook assigns eight threads to CosmoLike's OpenMP loops and
> one thread to BLAS. Change `ci.set_omp_threads(n=8)` in the notebook
> if fewer cores are available. Run one calculation at a time when
> measuring execution time.

> [!TIP]
> To inspect the forecast inputs before running CAMB, see
> [which survey the example uses](#survey).

# Changing the covariance accuracy <a name="accuracy"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`. The interface must already be compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: start Jupyter and open
`projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

    jupyter notebook --no-browser --port=8888

**Step :three:**: set the first calculation's accuracy in its configuration cell.

```python
boosts = [1, 2]
settings = survey.configuration(accuracy_boost=boosts[0])
```

The numerical baseline is in [`default.yaml`](default.yaml). The notebook
loads that file through `survey.configuration`. Its internal refinements
multiply the global boost: factors 2 and 3 at boost 1 become 4 and 6 at
boost 2. Resolved settings save both the base controls and effective grids.
`integration_accuracy` is independent of the global boost. Its levels
0/1/2/3/4 select precomputed GSL rules with 96/128/256/512/1024 nodes per
panel for radial, mass and angular integrals. Wide angular bins are split
into panels to resolve high-multipole oscillations. The low-mass Wynn tail
uses 32/64/128/256/512 nodes at the same levels; other rules retain a
64-node minimum. To check quadrature alone, use
`survey.configuration(accuracy_boost=1, integration_accuracy=1)` and compare
with level zero. On an M2 Pro laptop, stop Roman quadrature checks at level 3; reserve
level 4 for a server. Compare the default with the highest completed level
and check its last refinement separately. Keep interpolation settings fixed
in this scan and check them separately. The defaults are being checked against refined full
matrices; a higher level alone is not a convergence certificate.

`power_accuracyboost: 8` prepares all three matter-power tables with
11,993 wavenumber samples from the original 1,500 CAMB samples. Natural
cubic interpolation fills the dense tables once; the C calculations
continue to use linear lookups. This reduces interpolation errors
amplified by cancellation in the four-halo trispectrum. The global boost
also multiplies this refinement: boost 2 gives 23,985 samples, retaining
every boost-1 node. The notebook reinitializes these inputs for each
boost; custom scripts must likewise initialize after changing settings.

`accuracy_boost` controls interpolation and cutoffs. Supported values are
1, 2, 4 and 8. It raises the covariance multipole cutoffs and refines the non-Gaussian,
lensing-window and shared core interpolation tables. It leaves
CAMB and data-vector accuracy settings unchanged.

**Step :four:**: choose which values to compare in the refinement cell.

```python
boosts = [1, 2, 4]
```

For the more expensive comparison, include boost 8:

```python
boosts = [1, 2, 4, 8]
```

**Step :five:**: restart the kernel and run all cells.

The calculation keeps the cosmology, galaxy distributions, noise, angular
bins and Fourier-band endpoints fixed. Only the covariance accuracy changes. The highest
boost in the comparison supplies the reference matrix for the difference
plots and variance-ratio table.

> [!NOTE]
> A larger boost is a numerical resolution, not a guaranteed survey
> accuracy. Boost 1 uses the project baseline. Boost 8 uses more modes and
> table samples and can require substantially more memory and time.
> See [how to check convergence](#convergence).

# Reading the figures <a name="figures"></a>

| Figure | What it teaches |
| --- | --- |
| Split-triangle correlation matrix | Compare the generated native-space covariance, drawn above the diagonal, with the supplied likelihood covariance, drawn below it, after the same cuts. Index 0 sits at the bottom left, so the title's "Lower" and "Upper" name matrix triangles (row > column and row < column), not screen positions. Each uses its own diagonal normalization. |
| G, SSC and cNG maps and histograms | Compare each component after normalization by the total diagonal variances. |
| Halo trispectrum diagonal | See 1h, combined 2h, 3h, 4h and their sum at a chosen redshift, before survey projection. The signed axis retains negative terms. |
| Error changes | With multiple boosts, compare first-source-bin standard deviations with the highest tested boost, in percent, for the native measurement. |
| Generalized-mode report | Bound variance changes over every linear combination of measurements. |

The correlation comparison follows the layout of
[Friedrich et al. (2021), Fig. 6](https://arxiv.org/abs/2012.08568).
The component maps and histogram adapt
[Barreira, Krause & Schmidt (2018), Fig. 1](https://arxiv.org/abs/1807.04266).
These figures display the notebook's calculation, not data from the papers.

Negative correlations remain visible. A grey cell in an element-ratio
map means its denominator is zero or too small for the selected cutoff.
The component titles count undefined plotting ratios, not entries removed
by the likelihood mask. No plot clips eigenvalues or adjusts the covariance.

# Running the tests <a name="tests"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`, with the Roman kinematic lensing interface compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable the installed project.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE

**Step :three:**: enable covariance generation.

    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE

**Step :four:**: compile the interface.

    source ./projects/roman_kl/scripts/compile_roman_kl.sh

**Step :five:**: run the covariance tests.

    python -m pytest projects/roman_kl/tests/covariance

The tests check this project's input layout, real/Fourier assembly,
thread repeatability, subset positivity and saved metadata. The [covariance test guide](../tests/covariance/README.md)
describes each check.

For the ordinary data-vector tests, we again assume the activated Conda
Cocoa environment, Bash, and the current folder `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run only the data-vector tests.

    python -m pytest projects/roman_kl/tests/data_vector

These tests compare likelihood predictions with the stored references.
The [data-vector test guide](../tests/data_vector/README.md) explains them.

# Files <a name="files"></a>

| File or folder | Purpose |
| --- | --- |
| [EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb) | Run, refine and plot real/Fourier G, SSC and cNG matrices. |
| [roman_kl_covariance.py](roman_kl_covariance.py) | Specify survey inputs, initialize this project's interface and call the shared calculation. |
| [Shared covariance package](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/covariance/README.md) | Reuse integration preparation, Gaussian assembly, halo inputs, accuracy settings and diagnostics. |
| [Shared plotting script](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/plot_covariances.py) | Plot covariance arrays from any project. |
| [Covariance C files](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike/covariances/README.md) | Read the physics and each compiled component's role. |

# Appendix <a name="appendix"></a>

## FAQ: Which accuracy settings are available? <a name="accuracy-settings"></a>

Baseline values come from [`default.yaml`](default.yaml). Override them in
the evaluate YAML's `covariance` block or pass them to `survey.configuration`.
Reinitialize notebook inputs after changing the settings.

| Control | Baseline | Purpose |
|---|---:|---|
| `accuracy_boost` | `1` | Overall table and cutoff refinement; 1, 2, 4 or 8. |
| `integration_accuracy` | `0` | Independent quadrature level, 0 through 4. |
| `power_accuracyboost` | `8` | Subdivisions of input log-k intervals; multiplied by the global boost. |
| `ell_max` | `100000` | Real-space transform cutoff; Fourier measurement bands stay fixed. |
| `mask_ell_max` | `32768` | Survey-footprint spectrum cutoff. |
| `ng_ell_intervals` | `127` | Base log-multipole intervals for non-Gaussian interpolation. |
| `non_gaussian_accuracyboost` | `1` | Refinement of the non-Gaussian multipole grid. |
| `window_accuracyboost` | `1` | Lensing-window interpolation refinement. |
| `core_accuracyboost` | `1` | Shared core interpolation refinement for this calculation. |
| `response_step` | `5e-05` | Half-width of the log-k response derivative; divided by the global boost. |
| `nonlimber_lmax` | `4000` | Gaussian gg/gs non-Limber correction cutoff. |
| `nonlimber_accuracyboost` | `2` | Gaussian non-Limber distance-grid refinement. |

| `integration_accuracy` | Ordinary nodes per panel | Wynn-tail nodes per panel |
|---|---:|---:|
| 0 | 96 | 32 |
| 1 | 128 | 64 |
| 2 | 256 | 128 |
| 3 | 512 | 256 |
| 4 | 1024 | 512 |

The default power refinement prepares 11,993 samples from 1,500 inputs for
linear, nonlinear and cold-matter power. It refines interpolation, not the
Boltzmann solution itself, and does not affect ordinary data-vector runs.
Thread counts are environment settings, not accuracy or YAML keys.

Compare G, SSC, cNG and total, including off-diagonal elements, positivity
and generalized variance ratios. Keep the cosmology, catalog and measured
bins fixed while testing one control at a time. Integration, interpolation,
cutoff and parameter-error convergence are separate questions.

On the M2 Pro laptop, stop Roman integration tests at level 3; reserve level
4 for a server. No integration level above 4 is supported.

## FAQ: Which survey does the example use? <a name="survey"></a>

Lenses and sources read the same redshift file, `Roman_tomo_Ntomo10_src.nz` in `data/`.
Each column supplies a bin's radial shape. The catalog densities are
specified separately in [roman_kl_covariance.py](roman_kl_covariance.py).

The area, total source density of 4 galaxies per square arcminute and
quadrature shape dispersion 0.035 follow
[Xu et al.](https://arxiv.org/abs/2201.00739). The code needs dispersion
per component, so the example uses $`0.035/\sqrt{2}`$.
The lens catalog is an illustrative additional choice: total density 4
with the same ten redshift shapes as the sources. Both catalogs are
allocated equally among bins. These assumptions do not reproduce a
calibrated KL clustering sample or the supplied likelihood covariance.

The project uses ten lens and ten source distributions. The Fourier
range and band count follow its dataset; the angular bins illustrate a
companion measurement. Galaxy–shear rows retain only source > lens.

| Survey input | Example choice |
| --- | --- |
| Area | 2000 square degrees |
| Lens bins | 10 |
| Source bins | 10 |
| Angular bins | 20 logarithmic bins, 2.5–250 arcminutes |
| Integer Fourier bands | 20 bands, multipoles 20–4000 |
| Real-space vector | 3,300 entries |
| Fourier vector | 2,200 entries |
| Neutrino mass | Zero |
| Intrinsic alignment | Zero |
| Photo-z shifts | Zero |
| Magnification | Zero |

The adapter lists every density, shape dispersion, galaxy bias and
cosmological input. The notebook prints those settings and archives them
with each matrix. The examples do not replace a supplied likelihood matrix.

## FAQ: What does the calculation include? <a name="gaussian"></a>

For Gaussian fields, a four-point expectation separates into products
of two-point expectations. A covariance between measured spectra AB and
CD therefore needs AC, BD, AD and BC spectra, even if those crossed pairs
are excluded from the measured data vector. The example retains the
complete field matrix before assembling the measured rows.

The Gaussian signal uses nonlinear matter power, adding a non-Limber
linear correction to galaxy clustering and galaxy–shear spectra. Shear–shear
spectra remain Limber. Spherical spin operators average these spectra over
each angular bin. The signal covariance uses the $`f_{\rm sky}`$ approximation.

Pure white noise extends to arbitrarily high multipoles. The calculation
replaces that infinite sum with the analytic pair-noise expression using
a spherical-cap footprint. Signal and mixed signal-noise terms retain
the finite multipole sum. A cap correction to noise pair counts does not
make the signal covariance an exact treatment of an irregular footprint.

Fourier bandpowers include pure noise inside the finite band sum. They
average the core spectra directly. Real-space means retain Cocoa's
extra source-leg factor for matching its angular-transform convention.
Each convention is applied consistently to G, SSC and cNG signal terms;
white noise receives neither source conversion.

SSC describes correlations induced by modes larger than the survey;
cNG describes the connected four-point contribution inside it.
The notebook computes both terms for every measured cross block. SSC
uses common shell responses before their weighted outer products; cNG
uses the 1-halo, 2-halo (1+3), 2-halo (2+2), 3-halo and 4-halo terms.
These are approximations, not a simulation calibration. See
[Krause & Eifler](https://arxiv.org/abs/1601.05779) and
[Takada & Hu](https://arxiv.org/abs/1302.6994) for the physical decomposition.

## FAQ: How can users check convergence? <a name="convergence"></a>

Compare successive boosts with the same physical inputs. The notebook
prints the largest departure of the generalized covariance eigenvalues
from one. They solve

```math
C_b v = \lambda C_{\rm ref} v.
```

The smallest and largest eigenvalues bound the variance ratio for any
linear combination of measurements. This checks coupled directions that
a diagonal comparison alone can miss. Raw largest covariance eigenvalues
do not identify the most informative cosmological modes.

Every total matrix must have nonnegative variance in every direction;
an invertible total must be positive definite. A positive diagonal alone
is insufficient. The notebook reports eigenvalues without repairing them.

The reference boost is only the highest resolution tested. Establish its
stability with further refinement, then assess marginalized parameter
errors and the Fisher Figure of Merit for the intended survey. The
likelihood's data-vector $`\lvert\Delta\chi^2\rvert<0.2`$ rule is not a
covariance convergence requirement.

## FAQ: How can users reuse the calculation? <a name="reuse"></a>

A project supplies its own initialized interface, redshift files, survey
numbers and physical choices. The shared Python package prepares arrays
and asks the covariance C components to perform the integration and
projection. Copy the small survey adapter, not those algorithms.

Other project interfaces must link the same covariance sources before
running these examples. Cluster observables need their own physical
model; changing the number of galaxy bins does not create a cluster
covariance.

Rectangular projection inputs allow Python to request matrix subblocks.
The C routines use OpenMP inside one process and never start MPI work.
A future Python dispatcher can distribute those subblocks while keeping
all cross correlations in the assembled matrix.

## Choosing the Gaussian spectra <a name="gaussian_spectra"></a>

The `gaussian` block selects the physics used in Gaussian covariance.
`nonlimber: true` retains radial mode coupling for every galaxy–galaxy and
galaxy–shear pair, including internal crosses excluded from the data vector.
Shear–shear spectra remain Limber.

For example, to include a constant NLA amplitude of 0.6 in every source bin:

```yaml
covariance:
  space: fourier
  accuracy_boost: 1
  integration_accuracy: 0
  gaussian:
    nonlimber: true
    ia: NLA
    A1: 0.6
```

This amplitude is an illustrative choice. A list supplies one amplitude
per source bin. `ia: none` omits IA; `ia: TATT` also accepts `A2` and
`B_TA`, with the same scalar-or-per-bin convention. The core supplies their
standard growth dependence. These lists are bin amplitudes, not the
amplitude/slope pair used by some likelihood redshift-evolution models.

TATT returns E and B spectra. Real-space shear covariance includes both,
with the appropriate signs for xi+ and xi-. The Fourier example measures
E spectra. Non-Limber corrects the linear-alignment part of galaxy–shear;
higher-order TATT terms remain Limber.

**These options change Gaussian covariance only.** SSC and cNG retain
their existing lensing-only, Limber model, including the original SSC
normalization signal. Their arrays are unchanged when these Gaussian
options change. They do not constitute a complete IA four-point model.

The notebook uses the same choices:

```python
settings = survey.configuration(
    gaussian={"nonlimber": True, "ia": "NLA", "A1": 0.6},
)
```

`nonlimber_lmax` in `default.yaml` controls the correction cutoff.
`nonlimber_accuracyboost` refines its logarithmic distance grid; the global
`accuracy_boost` refines that grid and the cutoff together. Radial nodes
remain nested. `integration_accuracy` independently selects GSL quadrature.
Refine the cutoff and distance sampling when assessing a survey's
covariance accuracy; numerical mode tests do not replace Fisher tests.


For the zero-IA galaxy/shear example, the default cutoff is 4,000
and the distance grid has 8,193 samples. Doubling both changed the
complete Gaussian covariance variance modes by at most
0.0027% across real and Fourier space. Both base and refined matrices
were positive definite with the specified catalog noise. Other numerical
controls were held fixed in this check.
