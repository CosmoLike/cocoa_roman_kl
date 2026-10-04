"""Project choices for the shared galaxy/shear covariance notebook.

This module initializes 10 lens and 10 source distributions from the
project. Numerical algorithms live in cosmolike_notebook_utils.covariance.
The example is a massless-neutrino, zero-IA forecast with explicitly chosen
number densities; it does not reproduce the project's frozen likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    The covariance README records the catalog assumptions and their sources.
    Redshift-file normalization sets a shape, not a catalog number density.
    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it.
        accuracy_overrides = named internal controls from default.yaml.
    Returns:
        Fully resolved settings, including the unboosted accuracy parameters.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # Inclusive bands count every integer multipole once.
    band_edges = np.rint(np.geomspace(20, 4001, 21)).astype(np.int32)

    # These are the project's measured-pair cuts, not cuts on internal
    # spectra. Covariance still needs every crossed field correlation.
    excluded_gammat = []
    for lens in range(10):
        for source in range(lens+1):
            excluded_gammat.append([lens, source])

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once.
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.049,
            "H0": 67.27,
            "ns": 0.9645,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
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

        # File columns describe radial shapes; these flags fix their z convention.
        "lens_file": "data/Roman_tomo_Ntomo10_src.nz",
        "source_file": "data/Roman_tomo_Ntomo10_src.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Measured pair and band choices are fixed during accuracy refinement.
        "excluded_gammat": excluded_gammat,
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": np.linspace(np.log(1.e6), np.log(1.e17), 9),

        # Densities are per square arcminute. Shape noise is per component.
        "area_deg2": 2000.0,
        "lens_density_arcmin2": [4.0/10]*10,
        "source_density_arcmin2": [4.0/10]*10,
        "sigma_e_component": [0.035/np.sqrt(2.0)]*10,
        "bias": [1.24, 1.36, 1.47, 1.60, 1.76, 1.47, 1.60, 1.76, 1.76, 1.76],

        # Shell edges increase in a, from the distant boundary to the observer.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=21),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_roman_kl_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments: interface = initialized compiled project; settings = configuration();
        space = "real" or "fourier"; rows = optional measured row subset;
        progress = optional (stage, elapsed_seconds) callback;
        backend = None for notebook wrappers, interface.covariance for CLI.
    Returns: shared forecast dict, including resolved settings and coordinates.
    The full real layout has 3300 entries; Fourier has 2200 entries.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
