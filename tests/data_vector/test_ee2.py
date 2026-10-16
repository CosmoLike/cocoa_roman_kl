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
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factories (cosmolike_core/cocoa_testing_suites.py) build
# the classes; assigning each return value to the usual module-level
# name is what makes pytest collect it here, with unchanged test ids.
TestEE2Race = u.ee2_suite(
    u, name="TestEE2Race",
    race=('test_x18_ee2_race_ten_in_a_row', 18, 'example1', 'example1 (cosmic shear, NLA+EE2)'))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
