"""Survey choices of the roman_kl galaxy/shear covariance forecast.

The shared covariance code (cosmolike_notebook_utils.covariance, used by
the notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb and by the command-line
runner compute_covariance.py) asks a project for three functions:

  configuration() = the survey and the numerical settings, fully resolved;
  initialize()    = run CAMB once and install that state in the compiled
                    interface;
  compute()       = the covariance components G, SSC, cNG and their total.

The survey is the Roman kinematic-lensing (KL) forecast of Xu et al.
(arXiv:2201.00739). Kinematic lensing combines each galaxy's image with
its spectroscopically measured rotation, which separates the galaxy's
own orientation from the lensing shear: the shape noise drops to a
dispersion of 0.035, and the spectra give redshifts precise enough for
narrow tomographic bins. The 10 lens and 10 source bins read the same
n(z) file. Because the lens bins are narrow, default.yaml keeps the
Gaussian non-Limber correction of the clustering and galaxy-shear
spectra up to l = 4000 (nonlimber_lmax), the top of the measured range.

This is a forecast with massless neutrinos, linear galaxy bias and the
Gaussian intrinsic-alignment choice passed in `gaussian`; it does not
reproduce the matrices the likelihood reads (data/*.cv) or the frozen
test references. covariance/README.md lists the assumptions and their
sources.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return the resolved survey, cosmology and accuracy settings.

    Every value the shared forecast reads is written out here, and the
    numerical controls come from default.yaml (load_covariance_accuracy
    applies the boost and the overrides), so the returned mapping fully
    describes one calculation and is archived with its results. The
    redshift files fix only the shape of each n(z) (each column is
    normalized); the number densities below fix the counts.

    Arguments:
        accuracy_boost = None keeps the accuracy_boost of default.yaml;
            1, 2, 4 or 8 refines interpolation tables and cutoffs.
        gaussian = None, or a mapping that chooses the spectra of the
            Gaussian covariance: nonlimber (bool), ia ("none", "NLA" or
            "TATT") and the amplitudes A1, A2, B_TA (gaussian_model checks
            and expands it; see "Choosing the Gaussian spectra" in
            covariance/README.md).
        **accuracy_overrides = other default.yaml controls by name, for
            example integration_accuracy=1.

    Returns:
        a dict: cosmology, redshift files and their conventions, excluded
        galaxy-shear pairs, Fourier bands, halo-mass panel edges, area,
        densities [per arcmin^2], per-component shape dispersion, linear
        bias per lens bin, angular-bin edges [arcmin], radial panel edges
        in scale factor, the resolved accuracy controls (with the values
        before the boost) and the resolved gaussian model.

    Raises:
        TypeError for an unknown accuracy control, ValueError for a
        default.yaml that is not a mapping or an invalid gaussian model.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # 20 Fourier bands with integer edges log-spaced from l = 20 to 4000
    # (the n_cl, l_min and l_max of the likelihood datasets); band b holds
    # the multipoles band_first[b] .. band_last[b], both included, so every
    # integer multipole belongs to exactly one band
    band_edges = np.rint(np.geomspace(20, 4001, 21)).astype(np.int32)

    # The galaxy-shear pairs left out of the measured vector (the key is
    # named after gamma_t, the real-space galaxy-shear statistic): every
    # (lens, source) with source index <= lens index, 55 of the 100
    # pairs, the first pair (0, 0) included; the same pairs as ggl_exclude
    # in the likelihood yamls. They cut measured rows only: the Gaussian
    # covariance of the kept rows still needs every cross spectrum between
    # the fields, excluded pairs included.
    excluded_gammat = []
    for lens in range(10):
        for source in range(lens+1):
            excluded_gammat.append([lens, source])

    # The fiducial cosmology is shared by G, SSC and cNG, so CAMB runs once
    # (get_camb_cosmology reads this mapping: As_1e9 = 10^9 A_s,
    # w0pwa = w0 + wa, so wa = 0 here; AccuracyBoost scales both CAMB and
    # cosmolike, CAMBAccuracyBoost and CLAccuracyBoost only their own
    # side; kmax is in 1/Mpc; non_linear_emul = 2 takes the nonlinear power
    # from CAMB's Halofit, version takahashi)
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.049,
            "H0": 67.27,
            "ns": 0.9645,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            # massless neutrinos: initialize_forecast refuses mnu != 0
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # The redshift files give the shape of each bin's n(z); lenses and
        # sources read the same file. photoz_interpolation = 0 (cubic
        # spline) and photoz_zmid = 0 (the z column holds left bin edges)
        # are the likelihood's defaults.
        "lens_file": "data/Roman_tomo_Ntomo10_src.nz",
        "source_file": "data/Roman_tomo_Ntomo10_src.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Measured pairs, bands and halo-mass panels (ln M edges from
        # 10^-40 to 10^17 Msun/h, cov.halo_mass_edges) stay fixed when the
        # accuracy is refined.
        "excluded_gammat": excluded_gammat,
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Densities are per square arcminute: 4 galaxies per arcmin^2 in
        # each sample, split equally over the 10 bins. Xu et al. quote a
        # shape dispersion of 0.035 for both components added in
        # quadrature; the code needs it per component, 0.035/sqrt(2).
        # One linear galaxy bias per lens bin.
        "area_deg2": 2000.0,
        "lens_density_arcmin2": [4.0/10]*10,
        "source_density_arcmin2": [4.0/10]*10,
        "sigma_e_component": [0.035/np.sqrt(2.0)]*10,
        "bias": [1.24, 1.36, 1.47, 1.60, 1.76, 1.47, 1.60, 1.76, 1.76, 1.76],

        # theta_edges_arcmin: 20 log-spaced angular bins from 2.5 to
        # 250 arcmin, used only by the real-space layout (space="real").
        # a_edges: the panels, in scale factor, of the radial
        # (line-of-sight) integrals of the Limber spectra, with Gaussian
        # quadrature nodes inside each panel; the edges z = 3.1, 2, 1.5, 1,
        # 0.7, 0.4, 0.2 and 1e-5 are listed so that a increases from the
        # distant boundary to the observer.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=21),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the forecast state in the compiled interface.

    Arguments:
        interface = the imported cosmolike_roman_kl_interface module.
        settings = the mapping returned by configuration().

    Returns:
        a dict of the CAMB tables handed to the interface (the
        set_cosmology arrays), suitable for saving beside the results.

    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood's covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the forecast covariance with its G, SSC, cNG and total parts.

    Arguments:
        interface = the compiled interface after initialize().
        settings = the mapping returned by configuration().
        space = "real" (angular bins, theta_edges_arcmin) or "fourier"
            (the integer multipole bands; the layout of the likelihood).
        rows = None for every measured row, or an int32 array [n_rows, 3]
            of (observable type, bin A, bin B) to compute a subset; the
            internal cross spectra are always complete.
        progress = None, or a function called as progress(stage,
            elapsed_seconds) while the calculation runs.
        backend = None to use the notebook wrappers, or
            interface.covariance to call the production C++ bindings
            directly (what compute_covariance.py does).

    Returns:
        the dict of compute_forecast: the matrices "gaussian", "ssc",
        "cng" and "total", the mean signals, the coordinates and the
        resolved settings. The full real layout has 3300 rows (55 xi_+
        and 55 xi_- shear blocks, 45 galaxy-shear and 10 clustering
        blocks, 20 angular bins each); the Fourier layout has 2200
        (55 + 45 + 10 blocks of 20 bands).
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
