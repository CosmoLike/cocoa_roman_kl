"""Baryonic feedback drift tests BD1-BD7: frozen-vector pinning.

Baryonic feedback suppresses the matter power spectrum on small scales;
the bfmt theory block (Cobaya name baryon_suppression) computes that
suppression with one of several methods (see test_accuracy_baryons.py
for the list), and the likelihood multiplies it into the nonlinear P(k).

Each test evaluates the example1 configuration (cosmic shear, NLA) with
one feedback method on, against that method's frozen data vector: the
default-settings theory prediction written at freeze time by
generate_frozen_reference.py --baryons, at the frozen fiducial point
plus the method's cosmology override
(cocoa_test_utils.BARYON_POINT_OVERRIDES). At freeze time the chi2
against that vector was zero by construction, so the assertion

    chi2 <= CHI2_TOLERANCE (0.2)

pins the whole feedback pipeline: a failure means cosmolike or the bfmt
theory block changed its prediction since the freeze (a "drift"). This
is the idea of the reference tests (test_example1.py) applied to the
feedback pipeline, and it complements test_accuracy_baryons.py: the
accuracy checks regenerate their vector on the fly in every run, so they
measure numerical settings and can never see drift; these tests hold the
frozen vector still, so they measure drift and nothing else. The seven
tests cover every method the bfmt theory block implements:

  BD1. SP(k), power-law fb relation     BD2. SP(k), Akino et al. 2022
  BD3. SP(k), double power-law relation BD4. BCEmu
  BD5. FlamingoBaryonResponseEmulator   BD6. BACCOemu
  BD7. BCemu2025

To run only this file (from the Cocoa/ folder, cocoa environment
active, start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_baryons.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
# "4" is cocoa_test_utils.REQUIRED_OMP_THREADS: the race checks need
# several threads, and the frozen references were computed with four.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The shim cocoa_test_utils.py (this project's data bound to the shared
# test machinery) lives one folder up, in tests/; insert(0, ...) puts
# that folder first on the module search path, so a direct run of this
# file and the worker subprocesses import this project's shim.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestBaryonDrift(unittest.TestCase):
    """Drift tests BD1-BD7: the feedback pipeline against its freeze.

    setUpClass runs once: it moves to ROOTDIR and verifies every
    frozen file against the SHA-256 manifest before any physics runs.
    """

    @classmethod
    def setUpClass(cls):
        """Check the Cocoa shell and verify the frozen files, once per class.

        unittest calls this once, before the first test of the class
        (@classmethod passes the class itself as cls).
        """
        u.require_cocoa_environment()
        u.verify_frozen()

    def _baryon_drift_check(self, name, baryon, label):
        """Assert that one method's chi2 against its frozen vector stays small.

        Arguments:
          name   = the test label (BD1-BD7) for the report.
          baryon = a label of cocoa_test_utils.BARYON_METHODS.
          label  = one line naming the feedback method.

        Returns:
          nothing; the printed block shows the chi2 and the tolerance.

        Raises:
          AssertionError when the chi2 exceeds CHI2_TOLERANCE.
        """
        chi2 = u.baryon_drift_chi2(baryon)
        print(f"""
{'-' * 66}
DRIFT: {name}: {label}
  chi2 against the frozen feedback vector = {chi2:.6f}
  (zero at freeze time; tolerance {u.CHI2_TOLERANCE})
{'-' * 66}""", flush=True)
        self.assertLessEqual(
            chi2, u.CHI2_TOLERANCE,
            f"{name}: chi2 {chi2:.6f} exceeds the tolerance "
            f"{u.CHI2_TOLERANCE}; cosmolike or the bfmt theory block "
            "changed its prediction since the freeze")

    def test_bd1_spk_power_law(self):
        """BD1: SP(k) with the power-law fb relation."""
        self._baryon_drift_check("BD1", "spk power law",
                                 "SP(k), power-law fb relation")

    def test_bd2_spk_akino(self):
        """BD2: SP(k) with the Akino et al. 2022 fb relation."""
        self._baryon_drift_check("BD2", "spk akino",
                                 "SP(k), Akino et al. 2022")

    def test_bd3_spk_double_power_law(self):
        """BD3: SP(k) with the double power-law fb relation."""
        self._baryon_drift_check("BD3", "spk double power law",
                                 "SP(k), double power-law fb relation")

    def test_bd4_bcemu(self):
        """BD4: BCEmu."""
        self._baryon_drift_check("BD4", "bcemu", "BCEmu")

    def test_bd5_flamingo(self):
        """BD5: FlamingoBaryonResponseEmulator."""
        self._baryon_drift_check("BD5", "flamingo",
                                 "FlamingoBaryonResponseEmulator")

    def test_bd6_baccoemu(self):
        """BD6: BACCOemu."""
        self._baryon_drift_check("BD6", "baccoemu", "BACCOemu")

    def test_bd7_bcemu2025(self):
        """BD7: BCemu2025."""
        self._baryon_drift_check("BD7", "bcemu2025", "BCemu2025")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs every test method of the
# classes above and prints one line per method (verbosity=2).
if __name__ == "__main__":
    unittest.main(verbosity=2)
