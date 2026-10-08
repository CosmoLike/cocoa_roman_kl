"""Cobaya likelihood roman_kl.combo_3x2pt: the three two-point probes together.

3x2pt combines cosmic shear (C_l^ss of every source-bin pair),
galaxy-galaxy lensing (C_l^gs of the lens-source pairs the yaml key
ggl_exclude keeps: in combo_3x2pt.yaml, the 45 pairs whose source bin
is behind its lens bin) and galaxy clustering (C_l^gg, one per lens
bin).

Everything else (data files, theory inputs, chi2) is the shared
machinery of _cosmolike_prototype_base.py; this class only selects
the probe. Cobaya finds it through the likelihood name roman_kl.combo_3x2pt
(Cocoa links this folder into cobaya/likelihoods/roman_kl, hence the
import path below) and reads its default options from combo_3x2pt.yaml
next to this file.
"""

from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class combo_3x2pt(_cosmolike_prototype_base):
  """The 3x2pt likelihood: the shared base class with probe "3x2pt"."""
  def initialize(self):
    """Initialize the shared base class with this probe.

    "3x2pt" selects all three blocks of the data vector: C_l^ss, C_l^gs
    and C_l^gg.
    super(...).initialize calls the base-class method.
    """
    super(combo_3x2pt,self).initialize(probe="3x2pt")