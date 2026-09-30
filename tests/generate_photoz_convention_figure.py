"""Regenerate the photo-z convention figures the tests README shows.

Evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
test_photoz_conventions.py for what they mean) and plots the
fractional data-vector differences against the default,

    delta C_ell / C_ell = C_ell(setting)/C_ell(default) - 1,

per tomographic pair and per band power. Masked bands are left out.
Two figures, because the two knobs live on different scales:

    photoz_zmid_dcl.png   - the Z_LOW vs Z_MID reading of the n(z)
                            file z column (percent level),
    photoz_interp_dcl.png - linear and Steffen vs cubic spline
                            (1e-4 level).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/roman_kl/tests/generate_photoz_convention_figure.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"
NTOMO = 10
NCL = 20
L_MIN, L_MAX = 20.0, 4000.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data", "roman_kl.mask")

SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """The printed theory vector under each setting, keyed by tag."""
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
    """One 6x6 per-pair panel grid in the notebook-plotter layout.

    curves = {label: dcl}, each a (npair, NCL) fractional-difference
    array with NaN at masked bands.
    """
    edges = np.geomspace(L_MIN, L_MAX, NCL + 1)
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
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = NTOMO * (NTOMO + 1) // 2
    ncs = npair * NCL  # the cosmic-shear block leads the data vector

    def frac(tag):
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
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
