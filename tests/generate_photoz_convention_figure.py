"""Regenerate the photo-z convention figures the tests README shows.

Evaluates the frozen cosmic-shear fiducial (example1, NLA) under the
four runtime photo-z settings (cspline/Z_LOW default, linear, Steffen,
Z_MID; see test_photoz_conventions.py for what they mean), one model
per setting in one process, and plots the fractional differences of
the shear spectra against the default,

    delta C_ell / C_ell = C_ell(setting)/C_ell(default) - 1,

per tomographic pair and per band. Masked bands are left out. Two
figures, written next to this script, because the two knobs live on
different scales:

    photoz_zmid_dcl.png   - the Z_LOW vs Z_MID reading of the n(z)
                            file z column (percent level),
    photoz_interp_dcl.png - linear and Steffen vs cubic spline
                            (1e-4 level).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/roman_kl/tests/generate_photoz_convention_figure.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so this
# must run before any cobaya/cosmolike import; 4 threads, as in the tests
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

import matplotlib
# Agg is matplotlib's file-only backend: no window opens, so the script
# runs without a display; it must be chosen before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# the shim cocoa_test_utils.py sits in this folder (tests/); insert(0,
# ...) puts the folder first on the module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# example1 is the cosmic-shear configuration. NTOMO, NCL, L_MIN and L_MAX
# repeat its binning (10 source bins, 20 bands log-spaced between l = 20
# and 4000, as in tests/frozen/data/*.dataset), and MASK_FILE is the mask
# of its dataset (the 55 shear blocks kept).
EXAMPLE = "example1"
NTOMO = 10
NCL = 20
L_MIN, L_MAX = 20.0, 4000.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data", "roman_kl.mask")

# (report tag, photoz_interpolation_type, photoz_zmid_convention), the
# default first
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """Return the printed theory vector under each setting, keyed by tag.

    Each setting builds its own model from the frozen example1
    configuration and evaluates the fiducial point once; the evaluation
    prints the theory vector (print_datavector) into a temporary folder,
    which is deleted at the end (the finally block runs even on error).

    Returns:
      {report tag: 1D array of the full-length theory vector, masked
      entries 0}.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({EXAMPLE}, NLA, {tag}) ...", flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def plot(curves, fname, title, scale=100.0, unit="%", ylim=None):
    """Draw one grid of panels, one panel per shear pair, and save it.

    The grid has 8 x 7 panels for the 55 pairs (i <= j) of the 10 source
    bins; the 56th panel is switched off. The panels share both axes and
    touch each other (no space between them), as in the notebook
    plotters; the bin labels inside the panels count from 1.

    Arguments:
      curves = {label: dcl}, each dcl an array [npair, NCL] of fractional
               differences with NaN at masked bands; one color per label.
      fname  = output file name, written next to this script.
      title  = figure title.
      scale  = factor applied to dcl before plotting (100 gives percent).
      unit   = the unit shown in the y-axis label.
      ylim   = None, or the half-height of the symmetric y range, in the
               plotted unit.

    Returns:
      nothing; the PNG file is written (dpi 120) and the figure closed.
    """
    edges = np.geomspace(L_MIN, L_MAX, NCL + 1)
    # the band centers where cosmolike evaluates C_l
    # (init_binning_fourier): geometric means of the log-spaced band edges
    ell = np.sqrt(edges[1:] * edges[:-1])  # geometric band centers
    pairs = [(i, j) for i in range(NTOMO) for j in range(i, NTOMO)]
    fig, axes = plt.subplots(nrows=8, ncols=7, figsize=(21, 21),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, dcl) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            ax.semilogx(ell, scale * dcl[p], color=color, lw=1.6,
                        label=label if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$", transform=ax.transAxes,
                fontsize=13)
        if p >= 48:
            ax.set_xlabel(r"$\ell$", fontsize=16)
        if p % 7 == 0:
            ax.set_ylabel(rf"$\Delta C_\ell/C_\ell$ [{unit}]", fontsize=14)
    axes.ravel()[-1].axis("off")  # 55 pairs in an 8x7 grid
    if ylim is not None:
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    """Evaluate the four settings and write the two figures.

    Raises:
      RuntimeError from require_cocoa_environment outside a started
      Cocoa shell; AssertionError from verify_frozen when a frozen file
      changed.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    # the mask file holds "index value" lines; keep the values
    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = NTOMO * (NTOMO + 1) // 2
    ncs = npair * NCL  # the cosmic-shear block leads the data vector

    def frac(tag):
        """Return the fractional shear-spectrum difference of one setting.

        Arguments:
          tag = a SETTINGS report tag.

        Returns:
          an array [npair, NCL] of C_ell(setting)/C_ell(default) - 1 over
          the cosmic-shear block, NaN where the mask is 0.
        """
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        # np.errstate silences numpy's divide-by-zero and invalid-value
        # warnings inside the block: masked entries are 0/0, and np.where
        # replaces them by NaN anyway
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(mask[:ncs] > 0, cur[:ncs] / ref[:ncs] - 1.0,
                         np.nan)
        return d.reshape(npair, NCL)

    plot({"Z_MID": frac("Z_MID")},
         "photoz_zmid_dcl.png",
         "roman_kl cosmic shear: n(z) z-column read as Z_MID "
         "instead of Z_LOW (frozen fiducial)", scale=100.0, unit="%",
         ylim=4.0)
    plot({"linear": frac("linear"), "steffen": frac("steffen")},
         "photoz_interp_dcl.png",
         "roman_kl cosmic shear: linear and Steffen n(z) "
         "interpolation vs cubic spline (frozen fiducial)",
         scale=1.0e4, unit=r"$10^{-4}$", ylim=8.0)


if __name__ == "__main__":
    main()
