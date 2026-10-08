"""Cobaya likelihood roman_kl.combo_xi_ggl: cosmic shear plus galaxy-galaxy lensing.

The data vector holds the shear spectra C_l^ss of every source-bin
pair, then the galaxy-shear spectra C_l^gs (galaxy-galaxy lensing:
the shear of background sources around foreground lens galaxies) of
the lens-source pairs the yaml key ggl_exclude keeps.

Everything else (data files, theory inputs, chi2) is the shared
machinery of _cosmolike_prototype_base.py; this class only selects
the probe. Cobaya finds it through the likelihood name roman_kl.combo_xi_ggl
(Cocoa links this folder into cobaya/likelihoods/roman_kl, hence the
import path below) and reads its default options from combo_xi_ggl.yaml
next to this file.
"""

from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  """Cosmic shear plus galaxy-galaxy lensing: probe "xi_ggl"."""
  def initialize(self):
    """Initialize the shared base class with this probe.

    "xi_ggl" selects the shear spectra C_l^ss ("xi", the shared code's
    name for cosmic shear) and the galaxy-shear spectra C_l^gs ("ggl").
    super(...).initialize calls the base-class method.
    """
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")