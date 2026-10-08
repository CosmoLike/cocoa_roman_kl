"""Unit test 18: the race check with the EuclidEmulator2 nonlinear P(k).

EuclidEmulator2 (EE2) emulates the nonlinear boost B(k, z) = P_nl/P_lin
measured in N-body simulations; with non_linear_emul: 1 the likelihood
multiplies it onto the linear spectrum below z = 10 (Halofit above).
Cocoa pins a modified EE2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix (the
repository's README documents them).

Test 18 is the race check with EE2 on: on one model instance, the
fiducial is evaluated fresh and again as the 10th of 10 cosmologies in
a row (the nine others are cocoa_test_utils.RACE_PERTURBATIONS). EE2's
compute is OpenMP-threaded, so leaked state or a thread race inside it
shifts the second fiducial value; the two must agree within
RACE_TOLERANCE (1e-4).

The check that builds the unmodified EE2 (commit ff59f66) next to the
installed one and compares their data vectors runs as test 18 of the
lsst_y1 project (its tests/data_vector/test_ee2.py). The emulated
physics does not depend on the project, so that comparison is not
repeated here.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_ee2.py
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


class TestEE2Race(unittest.TestCase):
    """Test 18, behind the frozen-state verification.

    setUpClass runs once before the test: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the race check compares one
    build against itself, so the frozen state only supplies the
    configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        """Check the Cocoa shell and verify the frozen files, once per class.

        unittest calls this once, before the first test of the class
        (@classmethod passes the class itself as cls).
        """
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x18_ee2_race_ten_in_a_row(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.

        Raises:
          AssertionError when |tenth - fresh| >= RACE_TOLERANCE.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2("example1", tatt=False,
                                           ee2=True)
        u.report_race_test(
            18, "example1 (cosmic shear, NLA+EE2) race check: 10 "
            "cosmologies in a row", fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs every test method of the
# classes above and prints one line per method (verbosity=2).
if __name__ == "__main__":
    unittest.main(verbosity=2)
