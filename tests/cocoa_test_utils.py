"""Shared harness for the roman_kl unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
worker-subprocess isolation, the race and baryon checks, the
CFASTPT-vs-FASTPT comparison, and the terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file
carries what is roman_kl's alone - the examples table (the exact
configurations plus their EMUL2 emulator twins, each naming its
synthetic NLA dataset), the TATT point, the accuracy knobs, the
CFASTPT-vs-FASTPT comparison contract, and the two EMUL2-only
report printers kept at the end of this file -
and binds it to ONE cocoa_testing.CocoaTestHarness instance whose
methods are re-exported under the historical names, so the test
modules and generate_frozen_reference.py import everything from this
module exactly as before.

The frozen-state doctrine is unchanged: everything a test evaluates
lives under tests/frozen/, pinned byte for byte by
tests/manifest_sha256.json and verified before any model is built;
refreshing the frozen state stays a deliberate maintainer action
(generate_frozen_reference.py --overwrite).
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct

# ---- the project data ------------------------------------------------------

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point A2 and BTA are zero, so
# the nonzero values here make the TATT reference genuinely exercise
# the second-order terms.
TATT_POINT = {
    "ROMAN_KL_A2_1": 0.05,
    "ROMAN_KL_BTA_1": 0.05,
    "ROMAN_KL_A2_2": -1.51541,
}

# The two frozen configurations. "likelihood" is the cobaya component
# name, needed to reach that block inside the loaded info dictionary;
# "provenance" names the human-readable snapshot (never loaded).
# The TATT variants evaluate against data vectors GENERATED WITH TATT
# at the fiducial point. Reason: against an NLA-based vector the TATT
# chi2 sits away from its minimum, where it responds linearly (not
# quadratically) to tiny numerical changes: harmless
# rounding-level shifts would then eat much of the 0.2 chi2 band the
# reference tests allow. The two examples use different data sets, so each gets its
# own generated vector; the 2x2pt configuration shares example2's.
TATT_GENERATORS = {
    "tatt_roman_kl_shear.dataset": "example1",
    "tatt_roman_kl_3x2.dataset": "example2",
}

# The shipped roman_kl modelvectors sit away from the current code's
# minimum (chi2 10-12 at the fiducial), where the chi2 responds
# linearly to tiny numerical changes, and every chi2 comparison
# (reference or accuracy alike) comes out inflated. The NLA variants therefore evaluate against SYNTHETIC
# data vectors generated with the default (NLA) model at the fiducial
# during the freeze, one per data set, exactly like the TATT vectors.
SYNTHETIC_VECTORS = {
    "synthetic_roman_kl_shear.dataset": ("example1", False),
    "synthetic_roman_kl_3x2.dataset": ("example2", False),
    "tatt_roman_kl_shear.dataset": ("example1", True),
    "tatt_roman_kl_3x2.dataset": ("example2", True),
}

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py). This is a Fourier-space project: there is no
# lmax likelihood option (the ell range lives in the dataset), so
# unlike the real-space projects the block carries no lmax key.
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3 is the highest value that stays healthy in every project
    # scanned (desy1xplanck breaks down above it), so the all-knobs
    # check compares the default against 3; the one-at-a-time scan
    # keeps 5 as a deliberate stress knob
    "accuracyboost": 3.0,       # default 1.0
    "integration_accuracy": 10,  # default 0
    "kmax_boltzmann": 40.0,     # default 7.5
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example2 NLA configuration before the all-knobs checks, so a
# large all-knobs delta can be attributed to the knob causing it. The
# accuracyboost=5 entry is a stress knob: it exceeds what measuring
# the default numerics needs, and it is kept because it exposed an
# interface breakdown (a suspected fixed-size table) in desy1xplanck.
# Investigation order when several knobs move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time): an apparent
# CAMB sensitivity can masquerade as unresolved cosmolike-side
# resolution, so the cheap knobs must be settled before the expensive
# one is blamed. kmax_boltzmann and camb kmax are one physical cutoff
# seen from the two sides, so the scan moves them together.
ACCURACY_KNOBS = [
    ("accuracyboost -> 3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost -> 5 (stress)", {"accuracyboost": 5.0}, {}),
    ("integration_accuracy -> 10", {"integration_accuracy": 10}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost -> 2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint -> 50", {}, {"k_per_logint": 50}),
]

# The EMUL2 entries below carry three extra fields:
#   "emulator"        = an EMUL2 configuration: machine-learning
#                       emulators (the emulrdrag, emulbaosn, and
#                       emulmps theory blocks) replace the Boltzmann
#                       code. These are ADVISORY (no pass/fail; see
#                       test_emul2.py) and have no TATT variant;
#   "exact_example"   = the exact-physics configuration this entry is
#                       the emulated version of; its "nla_dataset" is
#                       reused, so the emulated chi2 and the exact
#                       reference are computed against the SAME
#                       synthetic NLA vector (against different data
#                       their difference would mix data mismatch with
#                       emulator error);
#   "exact_reference" = the exact-physics reference chi2 key the
#                       emulator's accuracy is judged against.
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "roman_kl.cosmic_shear",
        "tatt_dataset": "tatt_roman_kl_shear.dataset",
        "nla_dataset": "synthetic_roman_kl_shear.dataset",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "roman_kl.combo_3x2pt",
        "tatt_dataset": "tatt_roman_kl_3x2.dataset",
        "nla_dataset": "synthetic_roman_kl_3x2.dataset",
    },
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "roman_kl.combo_3x2pt",
        "likelihood": "roman_kl.combo_2x2pt",
        "tatt_dataset": "tatt_roman_kl_3x2.dataset",
        "nla_dataset": "synthetic_roman_kl_3x2.dataset",
    },
    "emul2_example1": {
        "frozen_module": "frozen_config_emul2_example1.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE1.yaml",
        "likelihood": "roman_kl.cosmic_shear",
        "emulator": True,
        "exact_example": "example1",
        "exact_reference": "example1_nla",
        "nla_dataset": "synthetic_roman_kl_shear.dataset",
    },
    "emul2_example2": {
        "frozen_module": "frozen_config_emul2_example2.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE2.yaml",
        "likelihood": "roman_kl.combo_3x2pt",
        "emulator": True,
        "exact_example": "example2",
        "exact_reference": "example2_nla",
        "nla_dataset": "synthetic_roman_kl_3x2.dataset",
    },
}

# Pass limit on the covariance-weighted difference of the two
# implementations at FASTPT_LOW_SETTINGS: at each point both blocks
# print their theory data vector, and the tested number is
# delta^T C^-1 delta - the chi2 OF the implementation difference,
# zero when the vectors agree. 0.2 is the house comfort band of the
# other checks, in reach since the two-grid fastpt block made the
# output-table density cheap (this project's own sweep: max delta
# chi2 0.188561 at the converged defaults; the historical
# single-grid default reached 26409 across the prior). The sweep
# also measured a residual floor near 0.175 that further density
# does not move (0.175 at eight times the density) - the one
# project where a difference beyond table density is visible,
# safely inside the band.
FASTPT_COMPARISON_TOLERANCE = 0.2

# The python FAST-PT side has numerical settings of its own, read by
# the fastpt theory block from its extra_args block
# (external_modules/code/PyFAST-PT/fastpt.py, symlinked into cobaya
# as theories/fastpt). The block computes on two grids: accuracyboost
# multiplies the density of the output table cosmolike reads with
# linear interpolation (the accuracy driver), and
# internal_accuracyboost the density of the internal grid the FFTLog
# convolutions run on; a cubic spline in log k upsamples the terms
# from one grid onto the other. Both boosts default to 1.0 = the
# converged configuration, so low IS the default; it is hard-coded
# here so the test keeps evaluating this exact configuration even if
# the defaults later move. High doubles both boosts, so the advisory
# column shows the residual grid response of low.
FASTPT_LOW_SETTINGS = {
    "accuracyboost": 1.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

FASTPT_HIGH_SETTINGS = {
    "accuracyboost": 2.0,
    "internal_accuracyboost": 2.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS
BARYON_METHODS = _cct.BARYON_METHODS
BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES

# The 30 CFASTPT-vs-FASTPT comparison points under this project's
# sampled-parameter prefix; the values are identical in every project.
FASTPT_COMPARISON_POINTS = _cct.fastpt_comparison_points("ROMAN_KL")

# ---- the harness -----------------------------------------------------------

# ONE instance binds the shared machinery to this project's data;
# everything below re-exports its surface under the historical names.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_roman_kl_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=ACCURACY_KNOBS,
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings=FASTPT_LOW_SETTINGS,
    fastpt_high_settings=FASTPT_HIGH_SETTINGS,
    fastpt_points=FASTPT_COMPARISON_POINTS,
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
_evaluate_cached = _cct._evaluate_cached
_load_datavector = _cct._load_datavector
_baryon_method = _cct._baryon_method
_baryon_dataset = _cct._baryon_dataset
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy
report_knob = _cct.report_knob
report_fastpt_comparison = _cct.report_fastpt_comparison

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
_baryon_accuracy_delta_impl = _H._baryon_accuracy_delta_impl
_baryon_drift_chi2_impl = _H._baryon_drift_chi2_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
baryon_accuracy_delta = _H.baryon_accuracy_delta
baryon_drift_chi2 = _H.baryon_drift_chi2
_worker = _H._worker
_run_isolated = _H._run_isolated
_fastpt_comparison_info = _H._fastpt_comparison_info
_fastpt_comparison_block = _H._fastpt_comparison_block
_run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s

# ---- project-only reports ---------------------------------------------------

def report_emul2_advisory(label, chi2, frozen_ref, exact_ref, limit):
    """Print one EMUL2 accuracy check: measurements and a recommendation.

    There is no pass/fail here. An emulator is an approximation, so
    the useful outputs are the numbers themselves: the change against
    the frozen emulator reference (nonzero: the installed emulator no
    longer reproduces the chi2 it gave at freeze time),
    the difference against the exact-physics chi2 at the same
    cosmology (how accurate the emulator is), and the recommendation
    derived from that accuracy. Emulated configuration and exact
    counterpart both evaluate the counterpart's synthetic NLA vector
    (see load_frozen_info), and the exact reference is 0.000000 there
    by construction, so |emulator - exact| is the emulator error at
    the same data and nothing else.

    Arguments:
      label      = one line naming the emulated configuration.
      chi2       = the emulator chi2 computed in this run.
      frozen_ref = the frozen emulator reference chi2.
      exact_ref  = the exact-physics reference chi2 (from the matching
                   example's frozen reference).
      limit      = the recommendation threshold on |chi2 - exact_ref|.

    Returns:
      |chi2 - exact_ref|, the accuracy difference the recommendation
      is based on.
    """
    drift = chi2 - frozen_ref
    delta_exact = abs(chi2 - exact_ref)
    if delta_exact < limit:
        verdict = "RECOMMENDED for actual data analysis"
    else:
        verdict = ("NOT recommended for actual data analysis "
                   f"(|delta chi2| >= {limit})")
    # one multi-line f-string; the drift prints at :+.6f (fixed six
    # decimals, sign always shown)
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  chi2 (this run, emulator)   = {chi2:.6f}
  frozen emulator reference   = {frozen_ref:.6f}  (drift {drift:+.6f})
  exact-physics reference     = {exact_ref:.6f}
  |emulator - exact| chi2     = {delta_exact:.6f}   (threshold: {limit})
  -> {verdict}
{'-' * 66}""", flush=True)
    return delta_exact

def report_emul2_race(label, fresh, tenth):
    """Print one EMUL2 race check, advisory only.

    Arguments:
      label = one line naming the emulated configuration.
      fresh = chi2 of the point evaluated first on the model.
      tenth = chi2 of the same point as the 10th of a row.

    Returns:
      |tenth - fresh|, the printed difference. A value above
      RACE_TOLERANCE is flagged as a possible race or state leak, but
      nothing fails: this file only alerts.
    """
    delta = abs(tenth - fresh)
    # the a-if-else expression picks the note from the comparison
    note = ("consistent" if delta < RACE_TOLERANCE
            else "WARNING: possible race condition or state leak")
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   ({note})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
{'-' * 66}""", flush=True)
    return delta
