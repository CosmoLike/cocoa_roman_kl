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
# "4" is cocoa_test_utils.REQUIRED_OMP_THREADS: the race checks need
# several threads, and the frozen references were computed with four.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import unittest

# The shim cocoa_test_utils.py (this project's data bound to the shared
# test machinery) lives one folder up, in tests/; insert(0, ...) puts
# that folder first on the module search path, so a direct run of this
# file and the worker subprocesses import this project's shim.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"
REFERENCE_KEY = "example2_nla"

# (report tag, adopt_limber_gs)
SETTINGS = (
    ("non-Limber (default)", 0),
    ("Limber", 1),
    ("non-Limber again (round trip)", 0),
)

# A stale ggl cache gives delta chi2 = 0 exactly; the floor is orders of
# magnitude below the measured value, so it only catches a dead flag.
DCHI2_FLOOR = 1.0e-6

# delta chi2 measured for this project (macOS, arm64, at the frozen
# settings, nonlimber_accuracyboost 4), and the relative band assertion
# 4 allows around it
DCHI2_MEASURED = 0.1027
DCHI2_RTOL = 0.05


class TestNonLimberGGL(unittest.TestCase):
    """Limber vs non-Limber ggl on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        """Check the Cocoa shell, verify the frozen files, load the references.

        unittest calls this once, before the first test of the class;
        @classmethod passes the class itself as cls, so cls.reference
        (the frozen reference chi2 values) is shared by every test.
        """
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_nonlimber_ggl(self):
        """Run the three evaluations and check assertions 1-5.

        Raises:
          AssertionError at the first failed assertion (numbered in the
          module docstring).
        """
        import numpy as np
        import cosmolike_roman_kl_interface as ci

        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
        pairs = None
        for tag, flag in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            name = u.EXAMPLES[EXAMPLE]["likelihood"]
            info["likelihood"][name]["adopt_limber_gs"] = flag
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            start = time.perf_counter()
            chi2s[tag] = u.evaluate_chi2(model, point)
            elapsed = time.perf_counter() - start
            print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                  "first evaluation of the model)", flush=True)
            # full-precision model vector (the printed one keeps 9 digits)
            vectors[tag] = np.array(ci.compute_data_vector_masked())
            if icov is None:
                # all settings share one data set (mask, covariance), so
                # the first build's masked inverse covariance serves all
                icov = np.array(ci.get_inv_cov_masked())
                like = model.likelihood[name]
                if hasattr(ci, "compute_data_vector_3x2pt_real_sizes"):
                    sizes = ci.compute_data_vector_3x2pt_real_sizes()
                    nlen = int(like.ntheta)
                else:
                    sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
                    nlen = int(like.ncl)
                # the ggl pairs in data-vector order: lens-major, the
                # pairs listed in the yaml key ggl_exclude left out (here
                # (0, 0) is cut, so block 0 is the pair (0, 1)); the set
                # comprehension makes the excluded pairs a set of tuples,
                # the list comprehension walks every (lens, source) pair
                excluded = {(int(zl), int(zs)) for zl, zs in
                            (getattr(like, "ggl_exclude", None) or [])}
                pairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                                  for zs in range(int(like.source_ntomo))
                                  if (zl, zs) not in excluded]

        dv_default = vectors[SETTINGS[0][0]]
        # delta = non-Limber minus Limber; @ is the matrix product, so
        # dchi2 = delta^T C^-1 delta
        delta = dv_default - vectors[SETTINGS[1][0]]
        dchi2 = float(delta @ icov @ delta)

        # the ggl block follows the cosmic shear block in every probe
        # combination (3x2pt, 2x2pt, 6x2pt): entries [ggl0, ggl1)
        ggl0 = int(sizes[0])
        ggl1 = ggl0 + int(sizes[1])
        npairs = int(sizes[1]) // nlen

        print(f"\n  delta chi2 report ({EXAMPLE}, NLA):")
        print(f"    {SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        print(f"    {SETTINGS[1][0]}: chi2 = {chi2s[SETTINGS[1][0]]:.6f}")
        print(f"    delta^T C^-1 delta = {dchi2:.4f} "
              f"(measured {DCHI2_MEASURED:.4f})")
        print("    per lens-source pair (the pair's block alone):")
        rows = []
        for p in range(npairs):
            block = np.zeros_like(delta)
            sl = slice(ggl0 + p*nlen, ggl0 + (p + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), p))
        # the largest contributions first; stop below 1e-3 of the total.
        # The pair label falls back to the block number when the pair
        # list and the block count disagree.
        for contribution, p in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            label = (f"(lens {pairs[p][0]}, source {pairs[p][1]})"
                     if len(pairs) == npairs else f"pair {p}")
            print(f"      {label:24s} {contribution:.4f}")

        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber ggl did not change the data vector: the "
            "adopt_limber_gs flag did not reach the C code, or the ggl "
            "cache did not rebuild")

        outside = np.concatenate((delta[:ggl0], delta[ggl1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the ggl block changed with adopt_limber_gs")

        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_default),
            "returning to the default did not reproduce the first data "
            "vector bit for bit; the ggl cache did not rebuild cleanly")

        self.assertLess(
            abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
            f"delta chi2 = {dchi2:.4f} differs from the measured "
            f"{DCHI2_MEASURED:.4f} by more than {DCHI2_RTOL:.0%}")

        # last, so that a stale frozen snapshot does not hide the four
        # checks above
        self.assertLess(
            abs(chi2s[SETTINGS[0][0]] - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE,
            f"{SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} vs frozen "
            f"reference {self.reference[REFERENCE_KEY]:.6f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs every test method of the
# classes above and prints one line per method (verbosity=2).
if __name__ == "__main__":
    unittest.main(verbosity=2)
