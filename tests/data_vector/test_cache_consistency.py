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
# "4" is cocoa_test_utils.REQUIRED_OMP_THREADS: the race checks need
# several threads, and the frozen references were computed with four.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# The shim cocoa_test_utils.py (this project's data bound to the shared
# test machinery) lives one folder up, in tests/; insert(0, ...) puts
# that folder first on the module search path, so a direct run of this
# file and the worker subprocesses import this project's shim.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"

# Sector membership by sampled-parameter name: a parameter belongs to the
# first sector whose pattern occurs in its name (_sector_of; re.compile
# builds a regular expression, ^...$ demands the whole name, and the "."
# of "other" matches any name, so every parameter lands somewhere).
# "bias" moves only in the scramble step, and only its B1 parameters (the
# one bias key of DELTAS); "other" (a name no earlier pattern claims, for
# example a B3NL parameter) never moves.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic. A
# key is an exact name (str) or a pattern searched in the name. Each
# delta is small against the parameter's range but moves the data
# vector far above rounding.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
# NSTEP = steps per phase; SCRAMBLE_STEP puts every sector one step past
# the ladder, so the scramble point differs from every ladder point in
# every sector; RESCALE_RTOL bounds assertion 2 (an m step only
# multiplies entries by (1 + m) factors, so the vector must follow to
# near double-precision rounding)
NSTEP = 3
SCRAMBLE_STEP = 4  # every sector at step 4, bias included
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    """Return the sector of one parameter name (the first SECTORS match)."""
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """Return {parameter: per-step delta} for one sector's sampled names.

    Arguments:
      sector = a SECTORS name.
      names  = the sampled parameter names that belong to the sector.

    Returns:
      a dict with one entry per name that a DELTAS key of the sector
      matches; names without a match stay fixed during the ladder.
    """
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # the conditional expression picks the test: a str key must
            # equal the name, a compiled pattern must occur in it
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial (example2)."""

    @classmethod
    def setUpClass(cls):
        """Check the Cocoa shell and verify the frozen files, once per class.

        unittest calls this once, before the first test of the class
        (@classmethod passes the class itself as cls).
        """
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """Return the ladder point with each sector at its given step count.

        Arguments:
          fid   = the frozen fiducial point, {name: value}.
          steps = {sector: step count}; a sector's parameters move by
                  step * delta from the fiducial.

        Returns:
          a new point dictionary (fid is not modified).
        """
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """Return, per data-vector entry, the source bins whose (1 + m) scale it.

        The 3x2pt vector is ordered: shear blocks (pairs i <= j, twice in
        real space for xi_+ and xi_-), galaxy-galaxy lensing blocks (the
        (lens, source) pairs ggl_exclude keeps, lens-major; in roman_kl the
        first pair kept is (0, 1), because (0, 0) is cut), clustering
        blocks, and in a 6x2pt project the CMB-lensing crosses gk, ks and kk.
        Each block has ncl entries in Fourier space (ntheta in real space).
        Cosmic shear scales with both bins, galaxy-galaxy lensing and ks
        with the source bin, every other block with none.

        Arguments:
          like = the likelihood instance (its ncl or ntheta, the bin
                 counts and ggl_exclude are read).
          np_  = the numpy module (this file imports numpy inside the test
                 methods only).

        Returns:
          an int array [n_entries, 2] of (i, j): the source bins whose
          factors multiply that entry, -1 meaning no factor.
        """
        import cosmolike_roman_kl_interface as ci
        real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
        sizes = [int(x) for x in
                 (ci.compute_data_vector_3x2pt_real_sizes() if real else
                  ci.compute_data_vector_3x2pt_fourier_sizes())]
        nlen = int(like.ntheta) if real else int(like.ncl)
        nsrc = int(like.source_ntomo)
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        k = 0
        ssrep = sspairs + sspairs if real else sspairs  # xi_plus + xi_minus
        for (i, j) in ssrep:
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        k = sizes[0] + sizes[1] + sizes[2]  # skip clustering
        if len(sizes) > 3:  # 6x2pt: gk (no factor), ks (source), kk (none)
            k += sizes[3]
            for zs in range(nsrc):
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
        return fac

    def _run_ladder(self, tatt):
        """Walk the forward and the mirrored ladder and check assertions 1-5.

        Arguments:
          tatt = True runs the TATT variant (IA_model 1) of example2,
                 False the NLA one.

        Returns:
          nothing; prints the final chi2 of each ladder.

        Raises:
          AssertionError at the first failed assertion (numbered in the
          module docstring).
        """
        import numpy as np
        import cosmolike_roman_kl_interface as ci

        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        results = {}
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            # sector_deltas = {sector: {parameter: per-step delta}} over
            # the sampled parameters: the dict comprehension makes one
            # entry per sector, the list comprehension inside picks that
            # sector's names (the _ discards the SECTORS pattern)
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            # the cosmology, source photo-z and calibration sectors must be
            # sampled, or the ladder would test nothing
            for s in ("cosmo", "dz_source", "m"):
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")
            # a sector nothing samples drops out of the ladder (in roman_kl
            # DZ_L equals DZ_S and every IA amplitude is fixed); the
            # generator expression inside tuple() keeps the phases with at
            # least one moving parameter
            active = tuple(s for s in PHASES if self.sector_deltas[s])

            phases = active if order == "forward" else tuple(reversed(active))
            # every sector starts at step 0, the fiducial; mfac holds the
            # calibration bins of every entry (_mpairs)
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    # the M values before this step, to predict its rescale
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    # assertion 2: an m step multiplies each entry by
                    # (1 + m_new)/(1 + m_old) for every calibration bin the
                    # entry carries
                    if sector == "m":
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        # sorted() orders the names as text: with 10 source
                        # bins that is M1, M10, M2, ..., M9, so mp[j] is not
                        # bin j for j > 0 (the note on the next line assumes
                        # bin order). The check holds anyway because every M
                        # has the same fiducial (0) and step, which makes all
                        # factors equal; for the same reason it cannot detect
                        # factors applied to the wrong bin.
                        mp = sorted(m_prev)  # M1..M5 in bin order
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        # compare the nonzero entries only (masked entries
                        # are 0 in both vectors)
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias included
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            print(f"  {order} ladder ({'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_nla(self):
        """Assertions 1-5 with the NLA intrinsic-alignment model."""
        self._run_ladder(tatt=False)

    def test_cache_consistency_tatt(self):
        """Assertions 1-5 with the TATT model (the FAST-PT tables rebuild)."""
        self._run_ladder(tatt=True)


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs every test method of the
# classes above and prints one line per method (verbosity=2).
if __name__ == "__main__":
    unittest.main(verbosity=2)
