"""Cobaya likelihood roman_kl.combo_2x2pt: galaxy-galaxy lensing plus galaxy clustering.

2x2pt is the pair of galaxy probes without cosmic shear: the
galaxy-shear spectra C_l^gs of the lens-source pairs the yaml key
ggl_exclude keeps (galaxy-galaxy lensing) and the clustering spectra
C_l^gg, one per lens bin.

Everything else (data files, theory inputs, chi2) is the shared
machinery of _cosmolike_prototype_base.py; this class only selects
the probe. Cobaya finds it through the likelihood name roman_kl.combo_2x2pt
(Cocoa links this folder into cobaya/likelihoods/roman_kl, hence the
import path below) and reads its default options from combo_2x2pt.yaml
next to this file.
"""

from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """Galaxy-galaxy lensing plus clustering: probe "2x2pt"."""
  def initialize(self):
    """Initialize the shared base class with this probe.

    "2x2pt" selects the galaxy-shear spectra C_l^gs and the clustering
    spectra C_l^gg; the shear spectra C_l^ss are left out.
    super(...).initialize calls the base-class method.
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")