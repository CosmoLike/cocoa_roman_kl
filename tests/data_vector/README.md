# Unit tests for the likelihoods

The frozen snapshot `frozen/` and its manifest `manifest_sha256.json`
live in the parent `tests/` directory; the paths below are relative to
it.

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads). These
tests also measure the accuracy of the hybrid emulated pipelines and
report whether they are accurate enough for data analysis (advisory:
no pass/fail).

Every model build runs in its own worker subprocess: example1 and
example2 use data sets with different mask and covariance dimensions,
and the cosmolike C layer aborts the whole process when a second
configuration with different dimensions initializes after the first.
The isolation is internal; the commands below stay the same.

The mechanism of every shared check family - the frozen-reference and
race contract, the zero-baseline construction, the sweeps and their
`--high`/`--mask` options, the accuracy settings and their meanings -
is documented once in
[the shared data-vector checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_checks)
of the CosmoLike core. This file records what the Roman
kinematic-lensing project (roman_kl) runs and what it measured.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The CFASTPT vs FASTPT comparison](#cfastpt_fastpt)
    2. [The Halofit vs EE2 checks](#halofit_ee2)
    3. [The EE2 race test](#ee2_race)
    4. [Advisory checks](#advisory_checks)
    5. [Accuracy checks](#accuracy_checks)
    6. [Baryonic feedback checks](#baryon_checks)
    7. [The photo-z convention checks](#photoz_conventions)
    8. [The non-Limber checks](#nonlimber_checks)
    9. [The sector-ladder cache check](#cache_ladder)
3. [Appendix](#appendix)
    1. [FAQ: What does this project's snapshot hold?](#frozen_copy)
    2. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`. Every command
below assumes `start_cocoa.sh` has been sourced once:

    source start_cocoa.sh

Run the full suite with

    python -m pytest ./projects/roman_kl/tests/data_vector

or, without pytest,

    python -m unittest discover -s ./projects/roman_kl/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the computed
$\chi^2$, the stored reference, the difference, and the pass limit.
A full run performs about 173 likelihood evaluations and takes a few
minutes. The test files force `OMP_NUM_THREADS=4` internally.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each - the frozen-reference
drift check and the race check, in NLA and in TATT
([mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen)).
The TATT variants set

    IA_model: 1
    ROMAN_KL_A2_1: 0.05
    ROMAN_KL_BTA_1: 0.05
    ROMAN_KL_A2_2: -1.51541

The suite follows the shared
[test-file naming convention](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_naming): a file name
means the same thing in every project. This project's
configurations and test numbers:

| file | configuration | tests |
|---|---|---|
| `test_example1.py` | cosmic shear | 1-4: drift and race, in NLA then TATT |
| `test_example2.py` | 3x2pt | 5-8: drift and race, in NLA then TATT |
| `test_example2_2x2pt.py` | 2x2pt (`roman_kl.combo_2x2pt`, the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing) | 11-14: drift and race, in NLA then TATT |
| `test_fastpt.py` | the sweep on cosmic shear, on the 3x2pt likelihood (the one-loop bias amplitudes stay fixed at zero) and on the 2x2pt likelihood (`roman_kl.combo_2x2pt`, cosmic shear dropped) | 15, 16, 17 |
| `test_ee2.py` | cosmic shear, NLA with EE2 (`non_linear_emul: 1`) | 18: race, including EE2's own threaded compute |

Tests 9 and 10 are not used: the numbering follows the shared
convention, where they are the frozen FAST-PT reference checks
this project does not ship.

### The CFASTPT vs FASTPT comparison (`test_fastpt.py`, tests 15-17) <a name="cfastpt_fastpt"></a>

The sweep mechanism, the zero-baseline construction and the
`--high`/`--mask` options are in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_fastpt).
Here, test 15 runs the sweep on cosmic shear, test 16 on the 3x2pt
likelihood and test 17 on the 2x2pt likelihood (galaxy clustering
plus galaxy-galaxy lensing, cosmic shear dropped). In test 16 the
frozen configuration fixes the one-loop bias amplitudes
(`ROMAN_KL_B2_*`, `ROMAN_KL_B3NL_*`) at zero, so the sweep scores the
intrinsic-alignment tables alone on the wider data vector; in test 17
clustering carries no intrinsic alignment, so the TATT tables are
scored through galaxy-galaxy lensing alone.

The masks `--mask` accepts here are `frozen` (each example's contract
mask, the default) and `ones` (every data point kept). The 3x2pt
frozen mask is already the all-ones mask (all 2,200 entries 1.0), so
`--mask=ones` reproduces the frozen numbers in tests 16 and 17 by
construction; in test 15 it weights the deviation with the full
3,300-dimension inverse covariance (every row unmasked) instead of
the shear-block masked inverse.

> [!NOTE]
> Before the two-grid upgrade of the fastpt theory block (2026-09)
> there was no upsampling and the difference reached
> $\Delta\chi^2 = 26409$ across the prior.

The point values, the design, and the decision record live with the
lsst_y1 project (its tests/README.md carries the full discussion);
the table below is this project's own measurement:

| output table (points) | internal grid (points) | max $\Delta\chi^2$ | Runtime / first row |
|---|---|---|---|
| 1,100 | 1,100 (shared) | 26409 | 1.00 |
| 1,024,900 (`accuracyboost: 1`, the default) | 1,100 (the default) | 0.1886 | 1.15 |
| 2,048,900 (`accuracyboost: 2`) | 1,300 (`internal_accuracyboost: 2`) | 0.1804 | 1.35 |
| 4,096,900 | 1,100 | 0.177 | 1.70 |
| 8,192,900 | 1,100 | 0.175 | 2.30 |

Runtime ratios use the first row of this table as the baseline, with the
same timed scope and machine. They compare sampling costs within this
project; absolute runtimes depend on the machine.

![The 30 comparison points, colored by the per-point difference](cfastpt_vs_fastpt_points.png)

> [!NOTE]
> The fastpt defaults hold the band on their own. The sweep also
> measured a residual floor near 0.175 that further density does
> not move - the one project where a difference beyond the table
> density is visible, safely inside the band. cfastpt
> (`IA_code: 0`) remains the reference implementation.

Measured on 2026-09-23:

- Test 16 (3x2pt): max $\Delta\chi^2 = 0.174298$ at the default
  camb/cosmolike settings and 0.077547 with `--high=1` (advisory
  doubled-grid columns 0.165475 and 0.051343).
- Test 17 (2x2pt): max $\Delta\chi^2 = 0.000012$ at the default
  settings and 0.000004 with `--high=1`: next to tests 15 and 16,
  essentially all of the implementation difference sits in the
  cosmic-shear block.
- Every `--mask=ones` sweep passed and reproduced its frozen-mask
  numbers at the printed six decimals: test 15 measured max
  $\Delta\chi^2 = 0.188561$ at the default settings (0.076974 with
  `--high=1`).
- At the all-zero IA point the two implementations printed identical
  data vectors on every unmasked row ($\Delta\chi^2 = 0.000000$).

Run the comparison at the default camb/cosmolike settings, at the
pushed settings (the full comparison is both invocations), and with
every data point kept:

    python -m pytest ./projects/roman_kl/tests/data_vector/test_fastpt.py
    python -m pytest ./projects/roman_kl/tests/data_vector/test_fastpt.py --high=1
    python -m pytest ./projects/roman_kl/tests/data_vector/test_fastpt.py --mask=ones

### The Halofit vs EE2 checks (`test_nonlinear.py`, NL1-NL2) <a name="halofit_ee2"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_halofit_ee2):
NL1 evaluates the cosmic-shear data vector and NL2 the 3x2pt one,
each with both nonlinear-power sources at the ten shared cosmologies
(halofit, `non_linear_emul: 2`, is the frozen contracts' setting).
The `--mask` option applies (`frozen` and `ones`). Measured on
2026-09-23 (the figure below, frozen mask):

- NL1 (cosmic shear): per-cosmology $\Delta\chi^2$ between 8.8 and
  491.2 (median 64.4), largest at the high-omegam draws; under
  `--mask=ones` (the full 3,300-dimension inverse covariance, every
  row unmasked) the run reproduced the frozen numbers at the printed
  four decimals: max $\Delta\chi^2 = 491.2244$, median 64.4280.
- NL2 (3x2pt): between 375.0 and 3,092.8 (median 911.8); the frozen
  3x2pt mask is already the all-ones mask, so `--mask=ones`
  reproduces the frozen numbers by construction; the run printed
  identical digits.

![The ten cosmologies, colored by the Halofit-vs-EE2 difference](halofit_vs_ee2_points.png)

### The EE2 race test (`test_ee2.py`, test 18) <a name="ee2_race"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_ee2):
the gate that compiles the pre-modification EuclidEmulator2 build
(commit `ff59f66`) and compares the two builds' data vectors runs as
the lsst_y1 project's test 18 (its `tests/test_ee2.py`) and is not
repeated here. This project's test 18 is the race check with EE2 on
(`non_linear_emul: 1`). Measured on 2026-09-23:

- Test 18: the fresh and 10th-in-a-row fiducial agree to all eight
  printed decimals.
- The pre-modification build, compiled side by side and evaluated at
  the ten cosmologies of the Halofit-vs-EE2 checks on this project's
  cosmic shear under its frozen mask, measured max
  $\Delta\chi^2 = 1.1\times10^{-4}$ against the installed build (max
  fractional data-vector difference $2.7\times10^{-5}$).

### Advisory checks (`test_emul2.py`, E1-E4) <a name="advisory_checks"></a>

These checks test the hybrid emulators: trained machine-learning
networks replace the Boltzmann code. There is no pass/fail; each
check prints four quantities:

| printed quantity | meaning |
|---|---|
| emulator $\chi^2$ | the emulated pipeline evaluated at the fiducial point |
| drift | change against the stored emulator reference; nonzero means the installed emulator no longer reproduces its stored $\chi^2$ |
| $\lvert\chi^2_\text{emulator} - \chi^2_\text{exact}\rvert$ | the emulator error against the exact-physics $\chi^2$ at the same cosmology |
| recommendation | RECOMMENDED for actual data analysis when $\lvert\chi^2_\text{emulator} - \chi^2_\text{exact}\rvert < 0.2$, NOT recommended otherwise |

A race-condition check (OpenMP threading) runs as well, warning
instead of failing.

Each emulated configuration evaluates the same synthetic NLA vector
as its exact counterpart, where the exact reference $\chi^2$ is
0.000000 by construction. The error above therefore compares the two
pipelines on identical data; the shipped EMUL2 modelvectors are not
used.

> [!NOTE]
> The trained-network files are read from
> `external_modules/data/emultrf`, not from the `frozen/` snapshot; the
> network device is pinned to `cpu` so the numbers do not depend on
> GPU availability.

Run them on their own:

    python -m pytest ./projects/roman_kl/tests/data_vector/test_emul2.py

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

[Mechanism, setting meanings and cost order](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_accuracy):
check A0, which runs first, raises one setting at a time on the 3x2pt
NLA configuration (including `accuracyboost: 5` as a stress test);
A1-A6 re-evaluate cosmic shear, 3x2pt and 2x2pt, in NLA and TATT,
with every setting pushed at once. This project's pushed values:

| setting | raised to |
|---------|-----------|
| `accuracyboost` (cosmolike) | 3 |
| `integration_accuracy` (cosmolike) | 10 |
| `internal_accuracyboost` (cosmolike) | 2 |
| `kmax_boltzmann` (cosmolike) | 40 |
| `AccuracyBoost` (CAMB) | 2 |
| `k_per_logint` (CAMB) | 50 |
| `kmax` (CAMB) | 50 |

There is no `lmax` entry here: the ell range lives in the dataset.
The nested-grid construction `accuracyboost` refines is commented in
`likelihood/_cosmolike_prototype_base.py`.

The likelihood yamls set `internal_accuracyboost: 1`, which keeps the
convolution grid equal to the output table: the exact single-grid
path. Coarser grids also hold: the lsst_y1 scan of this setting
measured $`\Delta^T C^{-1} \Delta \le 10^{-9}`$ against the single-grid
path down to `internal_accuracyboost: 0.27`.

#### Why these accuracy defaults? <a name="accuracy_defaults"></a>

The examples default to `k_per_logint: 50`: the old default
undersampled the CAMB transfer functions, and that error masqueraded
as an apparent CAMB `AccuracyBoost` sensitivity until the transfer
sampling was raised.

Run them on their own, or skip them (high-accuracy evaluations take
minutes):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_accuracy.py
    python -m pytest ./projects/roman_kl/tests --ignore ./projects/roman_kl/tests/data_vector/test_accuracy.py

### Baryonic feedback checks (`test_accuracy_baryons.py`, BF1-BF7; `test_baryons.py`, BD1-BD7) <a name="baryon_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_baryons):
the accuracy checks (BF) report the default-versus-pushed
$\Delta\chi^2$ with each feedback method on, one advisory check per
method; check BF0 additionally runs the one-setting-at-a-time scan
with the Akino SP(k) feedback on, so a large delta names the setting
causing it. The drift tests (BD) pin each method's frozen prediction
(`generate_frozen_reference.py --baryons`), one test per method.

Every checked configuration is measurable by construction: the
BACCOemu check evaluates with `omegab: 0.049`, inside that emulator's
baryon-density training box, whose floor sits exactly above the
fiducial `omegab: 0.04`; and the double-power-law point is chosen to
keep the baryon fraction inside SP(k)'s calibrated band over the full
redshift grid.

Run them with

    python -m pytest ./projects/roman_kl/tests/data_vector/test_accuracy_baryons.py
    python -m pytest ./projects/roman_kl/tests/data_vector/test_baryons.py

### The photo-z convention checks (`test_photoz_conventions.py`) <a name="photoz_conventions"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_photoz):
the frozen cosmic-shear fiducial under the default, each alternative
convention, and the default again. Measured on 2026-09-25:

- default: $\chi^2 = 0.000000$ (the frozen reference exactly).
- linear: $\Delta\chi^2 = 2.0\times10^{-5}$; Steffen:
  $\Delta\chi^2 = 1.5\times10^{-6}$. Either interpolant change is far below
  the survey's statistical precision.
- Z_MID: $\Delta\chi^2 = 2.35$ - the half-bin z-column reading
  is the one photo-z convention that matters at this precision.

The figures below are regenerated by
`generate_photoz_convention_figure.py`:

![The Z_LOW vs Z_MID reading of the n(z) z column](photoz_zmid_dcl.png)

![Linear and Steffen n(z) interpolation vs cubic spline](photoz_interp_dcl.png)

### The non-Limber checks (`test_nonlimber_ggl.py`, `test_nonlimber_gg.py`) <a name="nonlimber_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_nonlimber):
`adopt_limber_gs` and `adopt_limber_gg` are both `0` (the exact
projection below $\ell = 150$) in this project, because the measured
differences below are too large to absorb. In this project's
Fourier-space data vector each band center takes its Limber value
plus the non-Limber correction interpolated linearly between the two
integer multipoles around it.

Galaxy-galaxy lensing, measured on 2026-10-01
(`nonlimber_accuracyboost: 4`):

- $\Delta\chi^2 = 0.103$ for the 3x2pt data vector.
- The largest single-pair contributions (each pair's block alone) are
  (5,6), (4,5), (6,7), (3,4) with 0.024, 0.024, 0.022, 0.020: each
  lens bin with the source bin right behind it.
- On 2026-09-28 the default switched from Limber to the exact
  projection: the $\Delta\chi^2 = 1.63$ measured on 2026-09-27 was
  judged too large to absorb. The shipped data vector was regenerated
  with the non-Limber default and the frozen references were refrozen
  from it.
- That 1.63 was mostly the numerical error of the exact projection,
  not the Limber error: the non-Limber chi grid had 512 points
  (`nonlimber_accuracyboost: 1`), too few for the narrow lens bins.
  With today's code the same comparison gives 1.42, 0.100 and 0.103
  at 512, 1024 and 2048 points (measured 2026-10-01).

Galaxy clustering, measured on 2026-10-01
(`nonlimber_accuracyboost: 4`):

- $\Delta\chi^2 = 47.1$ for the 3x2pt data vector, against 0.103 for
  the same comparison in galaxy-galaxy lensing.
- The largest contributions are lens bins 6, 7, 5, 4 with 7.6, 7.5,
  7.1, 6.2 (each bin's block alone).
- 57.4 on 2026-09-28, when the non-Limber chi grid had 512 points
  (`nonlimber_accuracyboost: 1`) and cosmolike's 1% early exit
  truncated the correction: the narrow lens bins need 2048 points (the
  3x2pt $\chi^2$ moves by 8.4, 0.09, $10^{-5}$ from 512 to 1024,
  2048, 4096).
- On 2026-10-01 the clustering default switched from Limber to the
  exact projection: clustering had stayed Limber only because the key
  preserved the old behavior when it was introduced. The shipped data
  vector `roman_kl_3x2.modelvector` was regenerated with the
  non-Limber default and the frozen references were refrozen from it;
  `roman_kl_2.modelvector` (the `roman_kl_mcmc.dataset` vector of the
  cosmic-shear configuration) does not depend on the clustering
  spectra and is unchanged.

### The sector-ladder cache check (`test_cache_consistency.py`) <a name="cache_ladder"></a>

The deterministic mixed-update ladder and its bitwise assertions are
described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_cache_ladder).
This project's ladder walks three cosmology-only steps, then
source-photo-z and shear-calibration steps: the lenses are the source
sample and every IA amplitude is fixed, so the IA-only and
lens-photo-z rungs drop out.

## Appendix <a name="appendix"></a>

## :interrobang: FAQ: What does this project's snapshot hold? <a name="frozen_copy"></a>

The snapshot-protection mechanism is described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen).
This project's `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_*.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks (the EMUL2 trained networks are not copied: they live in `external_modules/data/emultrf` and their drift is part of what the advisory checks measure) |
| `EXAMPLE_*.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

Every variant - NLA and TATT - evaluates against a data vector
generated at the fiducial point when the snapshot was created, one
pair per data set, under `frozen/data/`
([why a generated vector](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen)):

| data set | NLA vector | TATT vector |
|----------|-----------|-------------|
| cosmic shear | `synthetic_roman_kl_shear` | `tatt_roman_kl_shear` |
| 3x2pt | `synthetic_roman_kl_3x2` | `tatt_roman_kl_3x2` |

Each vector is the prediction of its own frozen configuration, so
every exact-physics variant evaluates at its own $\chi^2$ minimum,
where the response to tiny numerical changes is quadratic and the
drift and accuracy numbers stay meaningful.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/roman_kl/tests/generate_frozen_reference.py --overwrite

**Step :three:**: add the feedback vectors of the drift tests

    python ./projects/roman_kl/tests/generate_frozen_reference.py --baryons

The first command deletes and rebuilds `frozen/` from the current
project, prints the eight new reference $\chi^2$ values, and rewrites the
manifest. It does not write the per-method feedback vectors of
`test_baryons.py`; the second command adds them and rewrites the
manifest again. Review the printed $\chi^2$ values against the old
references before committing: they define what every later test run
compares against.
