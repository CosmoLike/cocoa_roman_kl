# Unit tests for the roman_kl likelihoods

These tests catch two kinds of silent breakage: a chi2 that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

Every model build runs in its own worker subprocess: example1 and
example2 use data sets with different mask and covariance dimensions,
and the cosmolike C layer aborts the whole process when a second
configuration with different dimensions initializes after the first.
The isolation is internal; the commands below stay the same.

## Running the tests

From the `Cocoa/` folder, with the cocoa conda environment active and
`start_cocoa.sh` sourced:

    python -m pytest ./projects/roman_kl/tests

Without pytest:

    python -m unittest discover -s ./projects/roman_kl/tests -v

The suite changes no project files. Each test streams a progress line
per model build and per evaluation, then a report block with the
computed chi2, the stored reference, the difference, and the pass
limit. A full run performs about 50 likelihood evaluations and takes a
few minutes. The test modules force `OMP_NUM_THREADS=4` internally.
The suite never waits for a keypress: a space/enter prompt between
tests means the output is being piped through a pager such as `less`,
so run the command with nothing piped after it.

## The tests

1. `test_1`: chi2 of the cosmic-shear likelihood at a fixed reference
   point must stay within 0.2 of the value stored in
   `frozen/reference_chi2.json`.
2. `test_2`: on one model, that point is evaluated fresh and then
   again as the 10th of 10 cosmologies in a row; the two chi2 values
   must agree to 1e-4. Leftover state or an OpenMP race breaks the
   agreement.
3. `test_3`: same as test 1 with the TATT intrinsic-alignment model
   (`IA_model: 1`) and `ROMAN_KL_A2_1=0.05`, `ROMAN_KL_BTA_1=0.05`,
   `ROMAN_KL_A2_2=-1.51541`.
4. `test_4`: same as test 2 with the TATT model.
5. -8. the same four tests for the 3x2pt likelihood.
9. -14. `test_example2_2x2pt.py` (numbered 11-14): the four standard
   tests on `roman_kl.combo_2x2pt` (example2 with the probe selection
   reduced to galaxy clustering plus galaxy-galaxy lensing).

Accuracy checks (`test_accuracy.py`, A1-A6): the three probes with
both IA models re-evaluated with the numerical settings pushed far
beyond the defaults (cosmolike accuracyboost 5, integration_accuracy
10, kmax_boltzmann 40; CAMB AccuracyBoost 2, k_per_logint 50, kmax
50; no lmax here, the ell range lives in the dataset). Each check
reports delta chi2 = chi2(high accuracy) - chi2(default, frozen), no
pass/fail. High-accuracy evaluations take minutes; skip the file with
`--ignore ./projects/roman_kl/tests/test_accuracy.py`.

Every variant evaluates against a data vector generated at the
fiducial point during the freeze (NLA: `synthetic_roman_kl_shear` /
`synthetic_roman_kl_3x2`; TATT: `tatt_roman_kl_shear` /
`tatt_roman_kl_3x2`, one pair per data set, under `frozen/data/`).
The shipped modelvectors sit off the current-code minimum (chi2 10-12
at the fiducial), and away from a minimum the chi2 responds linearly
to tiny numerical changes; at its own minimum the response is
quadratic and the drift and accuracy numbers stay meaningful. The
accuracy file also runs a one-knob-at-a-time scan before the
all-knobs checks, so a large delta can be attributed to the knob
causing it.

## Why the tests keep their own copy of everything

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

- `frozen_config_example{1,2}.py`: the complete cobaya configuration
  as a yaml string plus the exact evaluation point. Every option and
  every parameter is written out, including the ones that normally
  come from `params_source.yaml` and the other default files, so
  editing those files cannot change what the tests evaluate.
- `data/`: the tests' own copy of the data vectors, covariance, n(z),
  and masks.
- `EXAMPLE_EVALUATE{1,2}.yaml`: snapshots kept only so a human can
  diff how the live examples drifted since the freeze.

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every frozen file. Each test
verifies the manifest first and refuses to run when a frozen file was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the frozen state
either.

## Refreshing the frozen state (maintainers only)

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze:

    python ./projects/roman_kl/tests/generate_frozen_reference.py --overwrite

Run it from the `Cocoa/` folder with the environment set up as above.
It rebuilds `frozen/` from the current project, prints the four new
reference chi2 values, and rewrites the manifest. Review the printed
chi2 values against the old references before committing: they define
what every later test run compares against.
