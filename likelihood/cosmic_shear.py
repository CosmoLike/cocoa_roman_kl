"""Cobaya likelihood roman_kl.cosmic_shear: cosmic shear alone.

Cosmic shear is the correlated distortion of galaxy shapes by weak
gravitational lensing; its data vector here holds the shear angular
power spectra C_l^ss of every source-bin pair i <= j.

Everything else (data files, theory inputs, chi2) is the shared
machinery of _cosmolike_prototype_base.py; this class only selects
the probe. Cobaya finds it through the likelihood name roman_kl.cosmic_shear
(Cocoa links this folder into cobaya/likelihoods/roman_kl, hence the
import path below) and reads its default options from cosmic_shear.yaml
next to this file.
"""

from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Cosmic-shear likelihood: the shared base class with probe "xi"."""
  def initialize(self):
    """Initialize the shared base class with this probe.

    "xi" is the shared code's name for cosmic shear (from the real-space
    shear correlation functions xi_+/-); in this Fourier-space project it
    selects the spectra C_l^ss.
    super(...).initialize calls the base-class method.
    """
    super(cosmic_shear,self).initialize(probe="xi")