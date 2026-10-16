"""Unit test: non-Limber galaxy-galaxy lensing (adopt_limber_gs).

The galaxy-galaxy lensing (ggl) spectrum C_l^gs enters the data vector
through gamma_t(theta) in real space and directly in Fourier space. The
Limber approximation replaces the exact projection of the 3D power
spectrum onto the sky by the single wavenumber k = (l + 1/2)/chi; it
fails at low l when the redshift kernels are narrow. The likelihood
yaml key adopt_limber_gs chooses how C_l^gs is computed:

  adopt_limber_gs: 0 (the default): below l = 150 the exact
      projection, computed by cosmolike's C_gs_tomo with the split of
      Fang, Krause, Eifler & MacCrann (arXiv:1911.11947): an FFTLog
      integral of the linear power spectrum plus, in Limber, what
      linear theory misses. In Fourier space each band center takes
      the Limber value plus the non-Limber correction interpolated
      between integer multipoles; with the shipped binning the 8
      lowest of the 20 band centers (l = 23 to 146) lie below l = 150.
  adopt_limber_gs: 1: Limber approximation at every multipole.

The Limber approximation fails at low l for the lens-source pairs
whose kernels overlap in redshift (lens bin = source bin, or the
source bin in front of the lens bin, where the signal is the intrinsic
alignment of the sources times the lens density). In this project the
measured-pair cut (ggl_exclude) already removes those pairs, the first
pair [0, 0] included, so Limber costs little: DCHI2_MEASURED = 0.10,
below the 0.2 band of the reference tests, with the largest
contributions from each lens bin paired with the source bin right
behind it (adjacent bins still overlap in redshift). The frozen
fiducial has A1 = 0 and no magnification bias, so here C_l^gs is pure
galaxy-galaxy lensing. The project defaults to the exact projection all
the same (galaxy clustering has its own key, adopt_limber_gg; see
test_nonlimber_gg.py). This test measures what Limber would cost.

It evaluates the frozen 3x2pt fiducial (NLA) three times in one
process: non-Limber, Limber, non-Limber again, and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 the Limber model
would score against a data set generated with non-Limber ggl. It prints
the total and the contribution of each lens-source pair (the pair's own
block of delta, cross-covariance with other pairs ignored).

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the ggl cache notices the change (a stale cache gives zero);
  2. only ggl entries change: cosmic shear, clustering, and every other
     block are bitwise equal between the two evaluations;
  3. switching back to the default reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here;
  5. the default (non-Limber) evaluation reproduces the frozen
     reference chi2 (checked last, so a stale snapshot cannot hide
     checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_nonlimber_ggl.py
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

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
TestNonLimberGGL = u.nonlimber_suite(
    u, name="TestNonLimberGGL", kind="ggl",
    measured=0.1027, default=0)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
