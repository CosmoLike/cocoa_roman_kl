"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike caches every expensive stage behind its own key: cosmology
(distances, growth, power-spectrum tables), intrinsic alignment
(FAST-PT tables under TATT), photo-z shifts (n(z) splines, lens
efficiencies), and the shear calibrations (a pure data-vector rescale).
A partial-invalidation bug, in which one sector's update path fails to
rebuild a stored table (a C static variable) that another sector reads,
produces silently wrong data vectors only in mixed update sequences,
which the tests that evaluate one point at a time never exercise.

The test walks a deterministic ladder in one process, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps    (omegam, H0, As_1e9 together)
    3 x IA-only steps           (A1; under TATT also A2 and BTA)
    3 x source-photo-z steps    (every DZ_S shift)
    3 x lens-photo-z steps      (every DZ_L shift)
    3 x shear-calibration steps (every M)

A sector in which the configuration samples no parameter drops out of
the ladder. In roman_kl's example2 the lens shifts DZ_L are set equal to
DZ_S (the lenses are the source sample) and every IA amplitude is fixed,
so the ladder has three phases: cosmology, source photo-z, M.

It records the final data vector, then evaluates one scramble point
(every sector moved at once, galaxy bias included; the chi2 is
discarded) and returns to the ladder's final point: the pipeline must
reproduce the recorded vector bit for bit. A second model instance
walks the mirrored ladder (M -> DZ_L -> DZ_S -> IA -> cosmology) to the
same final point: the answer must depend on the point, never on the
order of the updates.

Assertions, in each intrinsic-alignment model (NLA and TATT; the TATT
ladder exercises the FAST-PT rebuilds that NLA never touches):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative: cosmic shear by
     both bins' factors, galaxy-galaxy lensing (C_l^gs here, gamma_t in
     a real-space project) by the source factor, clustering by nothing;
  3. evaluating the final point again, unchanged, leaves the vector
     bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_kl/tests/data_vector/test_cache_consistency.py
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
TestCacheConsistency = u.cache_ladder_suite(
    u, name="TestCacheConsistency")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
