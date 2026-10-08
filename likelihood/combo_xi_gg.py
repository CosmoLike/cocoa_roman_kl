"""Cobaya likelihood roman_kl.combo_xi_gg: cosmic shear plus galaxy clustering.

The data vector holds the shear spectra C_l^ss of every source-bin
pair and the galaxy clustering spectra C_l^gg, one per lens bin. The
galaxy-shear block keeps its place in the data-vector layout (its size
follows ggl_exclude), but this probe does not compute it.

Everything else (data files, theory inputs, chi2) is the shared
machinery of _cosmolike_prototype_base.py; this class only selects
the probe. Cobaya finds it through the likelihood name roman_kl.combo_xi_gg
(Cocoa links this folder into cobaya/likelihoods/roman_kl, hence the
import path below) and reads its default options from combo_xi_gg.yaml
next to this file.
"""

from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class combo_xi_gg(_cosmolike_prototype_base):
  """Cosmic shear plus galaxy clustering: probe "xi_gg"."""
  def initialize(self):
    """Initialize the shared base class with this probe.

    "xi_gg" selects the shear spectra C_l^ss ("xi", the shared code's
    name for cosmic shear) and the clustering spectra C_l^gg ("gg").
    super(...).initialize calls the base-class method.
    """
    super(combo_xi_gg,self).initialize(probe="xi_gg")
