"""Shared Cobaya likelihood of roman_kl, the Roman kinematic-lensing project.

roman_kl forecasts a Roman Space Telescope weak-lensing analysis that uses
kinematic lensing (KL): each galaxy's image is combined with its
spectroscopically measured rotation (velocity field), which separates the
galaxy's own orientation from the lensing shear and lowers the shape noise
(Xu et al., arXiv:2201.00739; covariance/README.md). The spectra also fix
each galaxy's redshift, so the 10 tomographic bins (redshift slices of the
sample) are narrow, and one n(z) file serves as both lens and source
sample. KL here means kinematic lensing, not a Karhunen-Loeve compression:
the likelihood applies no data compression.

The data vector lives in Fourier space: angular power spectra C_l in
n_cl multipole bands, log-spaced between l_min and l_max (20 bands from
l = 20 to 4000 in the shipped datasets), in this order:
  shear-shear C_l^ss    (cosmic shear), every source-bin pair i <= j;
  galaxy-shear C_l^gs   (galaxy-galaxy lensing, ggl), every lens-source
                        pair not listed in the yaml key ggl_exclude;
  galaxy-galaxy C_l^gg  (galaxy clustering), one per lens bin.
The yaml files of this folder exclude every ggl pair whose source bin is
not behind its lens bin (source index <= lens index), the first pair
[0, 0] included; EXAMPLE_EVALUATE1.yaml keeps all 100 pairs
(ggl_exclude: []) to match the layout of its cosmic-shear dataset, whose
mask removes them instead.

At every point the theory vector t comes from cosmolike, the C library
compiled into cosmolike_roman_kl_interface (imported as ci), and the
likelihood returns ln L = -chi2/2 with chi2 = (t - d)^T C^-1 (t - d), a sum
over the entries the mask keeps (d = data vector, C = covariance).

The five likelihood classes of this folder (cosmic_shear, combo_xi_ggl,
combo_xi_gg, combo_2x2pt, combo_3x2pt) subclass _cosmolike_prototype_base
and only choose the probe; the yaml file of the same name next to each
class holds its default options.

Terms used below:
  Cobaya provider = the object through which a likelihood reads what the
      theory codes (CAMB, the emulators, FAST-PT, and bfmt, the
      baryonic-feedback block) computed at the current point;
  dataset file    = the small ini-style text file (data/*.dataset) naming
      the data vector, covariance, mask and n(z) files and the binning;
  mask            = one 0 or 1 per data-vector entry; 0 removes the entry
      from chi2;
  Fortran order   = the flattening of an [nz, nk] table in which the z
      index runs fastest, the layout ci.set_cosmology expects.
"""

# __future__ imports must come before any other statement (only the
# docstring, comments and blank lines may precede them). Under Python 3
# these three are already the default behavior and change nothing.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools

# Cobaya's likelihood base class and error type, and getdist's reader of
# the ini-style .dataset files
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2, the nonlinear-power emulator of non_linear_emul = 1
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print how long the body of a with-block took (a debugging aid).

  @contextmanager turns this generator into a context manager: in
  `with timer("label"):` the code before `yield` runs on entry, the
  block's body runs at the `yield`, and the print after it runs on a
  normal exit (an exception skips it: there is no try/finally).

  Arguments:
    label = text printed before the elapsed time.

  Returns:
    a context manager; on normal exit it prints "label: <seconds>s".
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

import cosmolike_roman_kl_interface as ci

# OpenMP team size of cosmolike, read once at import from OMP_NUM_THREADS
# (1 when the variable is unset); with_omp_threads re-applies it before
# every wrapped call
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Wrap a likelihood method so it first restores cosmolike's thread count.

    cosmolike's hot loops are OpenMP parallel regions, whose team size is
    what omp_get_max_threads() returns at that moment. That value starts
    as OMP_NUM_THREADS, but some Python libraries call
    omp_set_num_threads(1) without saying so, which lowers it for the
    process and leaves every later cosmolike region on one core. The
    returned wrapper calls ci.set_omp_threads(COSMOLIKE_OMP_THREADS) and
    then runs the method.

    It is used as a decorator: @with_omp_threads written above a def
    replaces that method by with_omp_threads(method). functools.wraps
    copies the method's name and docstring onto the wrapper, so error
    messages and help() still show the original.

    Arguments:
      fn = the method to wrap (any callable).

    Returns:
      wrapper, a function that takes the same arguments as fn and
      returns what fn returns.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name of this project: ROMAN_KL_M1 is
# the shear calibration of source bin 1, ROMAN_KL_DZ_S1 its photo-z shift,
# ROMAN_KL_B1_1 the linear bias of lens bin 1 (params_source.yaml,
# params_lens.yaml)
survey = "ROMAN_KL"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Cobaya likelihood shared by every roman_kl probe combination.

  Cobaya builds one instance per likelihood block of the yaml, copies
  every option of that block (and of the class's default yaml) onto the
  instance as an attribute (self.accuracyboost, self.IA_model, ...),
  calls initialize once, then get_requirements, and then logp at every
  point. One evaluation runs logp -> get_datavector ->
  internal_get_datavector: set_cosmo_related hands the matter power
  spectra, the growth factor and the distances of the current cosmology
  to cosmolike, set_lens_related and set_source_related hand over the
  nuisance parameters, cosmolike returns the theory vector, and
  compute_logp turns it into -chi2/2.
  """

  def initialize(self, probe):
    """Read the dataset file, build the table grids and configure cosmolike.

    Runs once, when Cobaya builds the model. The dataset file
    (self.data_file inside self.path, for example data/roman_kl_3x2.dataset)
    names the data vector, covariance, mask and n(z) files and the binning:
    n_cl bands log-spaced between l_min and l_max, and l_max_shear (a shear
    band whose center is not below it is dropped on top of the mask).

    It then builds the redshift and wavenumber grids of the tables handed
    to cosmolike at every evaluation (z_interp_1D for distances and growth,
    z_interp_2D and log10k_interp_2D for the power spectra) and makes the
    init_* calls of the compiled interface. Their order is the reference
    init sequence that the notebook wrappers copy.

    Arguments:
      probe = the probe combination: "xi" (cosmic shear alone; the name
              comes from the real-space correlation functions xi_+/-, here
              it selects C_l^ss), "xi_ggl", "xi_gg", "2x2pt" or "3x2pt".

    Returns:
      nothing; sets attributes of self and the C global state behind ci.

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.
    """
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ncl = ini.int("n_cl")
    self.lmin = ini.int("l_min")
    self.lmax = ini.int("l_max")
    self.lmax_shear = ini.int("l_max_shear")
    # ------------------------------------------------------------------------   
    # z nodes of the 1D tables (comoving distance, growth): dense below
    # z = 3, where the galaxy kernels live, sparser up to z = 50.1, plus
    # 1070-1100 around recombination (z near 1090) for the distance to the
    # CMB last-scattering surface. tmp = 1000 + 250*boost sets the counts.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # The z nodes of the 2D power-spectrum tables handed to cosmolike,
    # which interpolates linearly in z between exactly these nodes (direct
    # indexing into the uniform blocks; there is no internal regridding).
    # Linear interpolation leaves a sawtooth-shaped O(dz^2) residual that
    # vanishes at the nodes, so two grids that do not share nodes disagree
    # by the full residual amplitude: a grid whose nodes move with the
    # boost makes the chi2 jitter (order unity in the clustering vector,
    # measured on this project) instead of converging. The dyadic factor
    # m = 2^ceil(log2(boost)) below (capped at 16) refines each uniform
    # block by an integer factor with the same endpoints, so (a) every
    # block stays uniform (cosmolike keeps its two-segment direct
    # indexing, no search), (b) the nodes of every coarser grid are a
    # subset of the nodes of every finer grid, making a boost increase a
    # true refinement (the error falls like 1/m^2), and (c) boost 1 gives
    # the 140-node grid that CAMB receives (z_interp_2D_camb below).
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/(105 m)); the high block multiplies its interval count
    # (endpoint=True: 35 nodes = 34 intervals -> 34 m + 1 nodes). The last
    # node, z = 49.99, stays inside the hybrid emulators' range (zmax = 50);
    # redshifts this high matter only for CMB lensing.
    #
    # pk_z_refinement multiplies m on top of the factor the accuracy boost
    # sets. A Fourier-space data vector reads P(k, z) at fixed multipoles,
    # where the residual of the linear z interpolation does not average
    # out as it does in a real-space vector: roman_fourier's 3x2pt chi2
    # moves by 0.25, 0.030, 0.002 from m = 1 to 2, 4, 8, roman_real's and
    # lsst_y1's by <= 0.004 from m = 1 to 2.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement stays at this boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes only re-evaluate the
    # smooth z-spline CAMB builds from these transfer redshifts when
    # the cosmolike tables are filled, so raising the boost refines
    # exactly the table resampling that produced the jitter, and the
    # CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    # log10 of k in 1/Mpc, from 1.02e-5 to 100/Mpc, 1500 nodes at boost 1;
    # set_cosmo_related subtracts log10 h to hand cosmolike k in h/Mpc
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # cosmolike keeps its configuration in C global variables, which the
    # init_* calls below fill once
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ncl), int(self.lmin), int(self.lmax), int(self.lmax_shear))
    # ggl_exclude = the (lens, source) pairs left out of galaxy-galaxy
    # lensing, flattened for the C side ([[0, 0], [1, 0]] -> [0, 0, 1, 0]).
    # It must describe the same layout as the dataset's data vector, mask
    # and covariance (the 3x2pt yamls keep 45 of the 100 pairs).
    ci.init_ggl_exclude(np.array(self.ggl_exclude).flatten())

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # how the n(z) tables are interpolated (0 = cubic spline, 1 = linear,
    # 2 = Steffen, monotone) and how their z column is read (0 = left bin
    # edges, 1 = sample points); tests/data_vector/test_photoz_conventions.py
    # measures both
    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    # C-FAST-PT (cosmolike's C version of the FAST-PT perturbation-theory
    # integrals) runs its FFTLog convolutions on an internal grid of this
    # factor times the density of its output table; 1.0 = equal grids
    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # the chi grid of the non-Limber FFTLog integrals, refined on top of
    # the accuracy boost: the narrow lens bins need it (combo_3x2pt.yaml
    # sets 4, NL_Nchi = 2048; see init_nonlimber_accuracy_boost)
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # 0 = exact (non-Limber) projection below l = 150 for galaxy-galaxy
    # lensing (gs) and clustering (gg), Limber above it; 1 = Limber at
    # every multipole (tests/data_vector/test_nonlimber_ggl.py and
    # test_nonlimber_gg.py measure the difference)
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 0)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power;
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics use the cold dark matter + baryon spectrum P_cb. The
    # emulators provide no P_cb, so the hybrid path (use_emulator = 2)
    # uses the small-scale ratio P_lin/(1 - f_nu)^2 (get_neutrino_inputs).
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # use_emulator: 0 = CAMB supplies P(k), growth and distances; 2 =
    # hybrid, trained emulators supply them (the EXAMPLE_EMUL2 yamls); 1 =
    # emulators would supply the whole data vector, a mode this Fourier
    # project does not complete (get_datavector returns 0.0 for it)
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_fourier(self.cov_file, self.mask_file, self.data_vector_file)
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      # is_linear=False: cosmolike uses the nonlinear P(k) it is handed
      # (True would force the linear spectrum everywhere)
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling = 1: Python keeps the n(z) tables
      # (self.lens_nz, self.source_nz) and hands a copy to cosmolike at
      # every evaluation (set_lens_related, set_source_related), so a user
      # function can change them per point; 0: cosmolike reads the files
      # once
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 

      # the dataset's covariance, mask and data vector; cosmolike inverts
      # the masked covariance once, here
      ci.init_data_fourier(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA needs no Python FAST-PT tables: use the C path (IA_code 0),
        # so get_requirements does not request the fastpt theory block
        self.IA_code = 0
      # IA_model: 0 = NLA, 1 = TATT. IA_redshift_evolution = 3 (the yamls)
      # makes A1(z) = A1_1 ((1 + z)/(1 + z0))^A1_2, a power law in
      # redshift, and likewise for A2. IA_code: 0 = C-FAST-PT, 1 = Python
      # FAST-PT.
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # bias_model = one model code per galaxy-bias term, in the order
        # [b1, b2, bs2, b3, bmag, bK] (cosmolike bias.h): 0 = one amplitude
        # per lens bin; the yamls set b3 to 1 (B3_FROM_B1, b3 follows from
        # b1). Cosmic shear alone ("xi") has no galaxies, hence no bias.
        ci.init_bias(bias_model=self.bias_model)

      # non_linear_emul = 1: EuclidEmulator2 corrects the nonlinear power
      # (set_cosmo_related); the emulator object is built once, here
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # external_baryon_suppression = 1: the bfmt theory block supplies the
      # baryonic suppression of the nonlinear P(k) (set_cosmo_related
      # applies it). It excludes the two other baryon treatments below:
      # the PCA marginalization (use_baryon_pca) and the contamination of
      # P(k) with one simulation (add_baryons_on_dv).
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      # create_baryon_pca = 1: compute the baryon principal components of
      # the simulations named in baryon_pca_select_sims (written to
      # filename_baryon_pca by internal_get_datavector).
      # add_baryons_on_dv = 1: contaminate the matter power spectrum with
      # the baryonic effect of the simulation which_bsims_add_on_dv, read
      # from the dataset's all_sims_hdf5_file.
      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    # use_baryon_pca = 1: marginalize over baryonic feedback with principal
    # components (PCs). baryon_pca_file holds one PC per column and one row
    # per data-vector entry; the theory vector gains sum_i Q_i PC_i over
    # the first npcs = 4 columns, with amplitudes ROMAN_KL_BARYON_Q1..Q4.
    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Tell Cobaya which theory products every evaluation needs.

    Cobaya calls this once, after initialize, and makes the theory codes
    compute these products at every point; set_cosmo_related reads them
    through self.provider.

    use_emulator = 0 (CAMB): H0, As, omegam, omegab, omnuh2 (Omega_nu h^2
      of the massive neutrinos), the linear and nonlinear matter power
      spectra (delta_tot) and the linear cold dark matter + baryon
      spectrum (delta_nonu) at the redshifts z_interp_2D_camb up to
      k_max = kmax_boltzmann*accuracyboost [1/Mpc], and the comoving
      distance on z_interp_1D [Mpc]; plus the FAST-PT tables when
      IA_code = 1 and the baryon suppression S(k, z) when
      external_baryon_suppression is on. It also requests Cl {'tt': 0}:
      no CMB multipole is used, but the inline note records that CAMB
      misbehaves without that request.
    use_emulator = 2 (hybrid): the same with mnu in place of omnuh2,
      without delta_nonu (the emulators predict only delta_tot), without
      the Cl request and without the baryon suppression.
    use_emulator = 1: the emulated data-vector blocks of the probe ('ss',
      'sg', 'gg') and, for galaxy probes, H0 and the distances.
    non_linear_emul = 1 adds the parameters EuclidEmulator2 reads
    (omegab, mnu, w, wa).

    Returns:
      a dict {product name: options or None}, Cobaya's requirement format.
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'ss': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'ss': None,
          'sg': None,
          'gg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'ss': None,
          'gg': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'ss': None,
          'sg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'sg': None,
          'gg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
      # IA_code = 1: the Python FAST-PT theory block supplies the
      # intrinsic-alignment and one-loop galaxy-bias tables
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # mnu gives Omega_nu h^2 on this path (get_neutrino_inputs)
      _requirements_["mnu"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The bfmt theory block computes the baryonic suppression on the grid
      # requested here: the table redshifts z_interp_2D and the wavenumbers
      # k = 10^log10k_interp_2D in 1/Mpc (the block converts to h/Mpc
      # itself when its model needs it).
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code = 1: the Python FAST-PT theory block supplies the
      # intrinsic-alignment and one-loop galaxy-bias tables
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Keep both fields available to the likelihood and direct halo readers.
      # CAMB obtains them from the same transfer-function calculation.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Hand the current cosmology's power spectra, growth and distances to cosmolike.

    Runs at every evaluation (a hot path: numpy operations on whole
    tables). For use_emulator != 1 it builds, on the (z_interp_2D,
    log10k_interp_2D) grid,
      lnPL  = ln P_lin(k, z), the linear matter power;
      lnPNL = ln P_nl(k, z), the nonlinear matter power: CAMB's Halofit
              (non_linear_emul = 2), or the EuclidEmulator2 boost times
              P_lin below z = 10 and Halofit above (non_linear_emul = 1),
              times the bfmt baryon suppression when
              external_baryon_suppression is on;
    plus the linear growth G(z) on z_interp_1D (up to the last
    z_interp_2D node) and the comoving distance chi(z) on z_interp_1D.

    Units and layout handed to ci.set_cosmology: k in h/Mpc, P in
    (Mpc/h)^3 (adding ln h^3 converts from Mpc^3), chi in Mpc/h; the 2D
    tables are flattened in Fortran order (flatten(order='F')), so entry
    iz + nz*ik holds (z[iz], k[ik]). With IA_code = 1 it also hands over
    the Python FAST-PT tables.

    Returns:
      nothing; changes the C global state behind ci.

    Raises:
      LoggedError for a non_linear_emul other than 1 or 2.
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      
      
      # ln P_lin on the (z, k) table grid: logP takes k in 1/Mpc and gives
      # ln P in Mpc^3; + ln h^3 converts to (Mpc/h)^3
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EuclidEmulator2 covers z < 10 and 8.73e-3 <= k <= 9.4 h/Mpc: its
        # boost B = P_nl/P_lin is requested on that k range (10^-2.0589 to
        # 10^0.973 h/Mpc), then interpolated in log10 k onto the table grid
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        # below EE2's lowest k the boost is 1 (ln B = 0); above its highest
        # k the linear extrapolation of ln B in log10 k continues
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # start from CAMB's Halofit, which covers every redshift, ...
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # ... and use P_lin*B at z < 10: np.where with the [nz, 1]
        # condition picks, row by row, the emulated or the Halofit table,
        # and ravel(order='F') flattens the result back with z fastest
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      # G on the dense 1D z grid (clipped to the P(k) interpolator range):
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 9e-5 and the
      # growth rate f = 1 - (1+z) dlnG/dz (the slope of the table) by
      # 1%; on the 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a cubic
      # spline in z through CAMB's transfer redshifts, so this asks CAMB
      # for no extra redshifts (about 0.1 ms per evaluation). The table
      # is divided by G at the last z_2D node (z_growth ends below it);
      # cosmolike's growfac divides by G(0), so D(z=0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is sampled at growth_k (default 0.05/Mpc), a sub-horizon scale.
      # At k = 5e-4/Mpc (about 2 H0/c) CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every
      # reader of G (IA amplitudes, one-loop D^4, sigma(M, z), the growth
      # rate f) describes sub-horizon modes; with 0.06 eV neutrinos the
      # growth varies by 0.03% above 0.05/Mpc (measured in
      # cosmolike_core/.claude/skills/cosmolike-dev/references/
      # growth_factor_measurements.md).
      growth_k = float(getattr(self, "growth_k", 0.05))
      # G(z) = D(z) (1 + z), with D(z)/D(0) = sqrt(P_lin(z, k)/P_lin(0, k))
      # at k = growth_k
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # external_baryon_suppression: the bfmt theory block returns
      # {z: S(k)}, the ratio of the power with baryonic feedback to the
      # dark-matter-only power on the requested k grid (after its own
      # handling of the ranges its model is not calibrated for). ln S is
      # added to ln P_nl at that z: entries i, i + nz, i + 2 nz, ... of the
      # z-fastest table (the Python slice i::nz, from i in steps of nz).
      # A z missing from the result, or
      # any error while reading it, is logged and skipped, which leaves
      # P_nl without suppression at that z (or at every z).
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: hand the Python FAST-PT tables to cosmolike. This must
      # follow ci.set_cosmology, which resets cosmolike's cosmology cache
      # key (cosmology.random). FPTIA rows: ten intrinsic-alignment terms,
      # then k [h/Mpc] (row -2) and P_lin (row -1), one column per k;
      # FPTbias ends with the same two rows.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra far above the neutrino free-streaming
        scale, an approximation on cluster scales. Its measured size is
        in projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Hand the source-sample nuisance parameters to cosmolike.

    Runs at every evaluation. Per source bin i = 1..source_ntomo it reads
    ROMAN_KL_M<i>, the multiplicative shear calibration m (the shear of
    bin i is scaled by 1 + m), ROMAN_KL_DZ_S<i>, the photo-z shift (n_i(z)
    is evaluated at z - Delta z_i), and the intrinsic-alignment
    parameters ROMAN_KL_A1_<i>, ROMAN_KL_A2_<i> and ROMAN_KL_BTA_<i>.
    Their meaning follows IA_redshift_evolution: with the power law of
    the yamls (3), A1_1 and A1_2 are the amplitude and the redshift
    exponent of A1, likewise for A2, and BTA_1 is the density weighting
    b_TA of TATT. A parameter absent from params counts as 0. With
    use_emulator = 1 only m is set.

    Arguments:
      **params = {name: value} of the point, as Cobaya passes them to logp.

    Returns:
      nothing; changes the nuisance state behind ci.
    """
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # n(z) is sent at every evaluation, so a user function can change
        # it per point (for example to add outliers): copy the stored
        # table (self.source_nz keeps the unmodified one), change the copy,
        # then hand it over with set_source_sample
        source_nz_local = self.source_nz.copy()

        # a user modification goes here, for example
        # source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # the photo-z shifts still apply on top of the handed n(z); a user
        # function that models them itself would drop this call
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Hand the lens-sample nuisance parameters to cosmolike.

    Runs at every evaluation of a probe with galaxies (not for "xi"). Per
    lens bin i = 1..lens_ntomo it reads ROMAN_KL_B1_<i> (linear galaxy
    bias, default 1), ROMAN_KL_B2_<i> (quadratic bias), ROMAN_KL_BMAG_<i>
    (magnification bias), ROMAN_KL_B3NL_<i> (third-order nonlocal bias),
    ROMAN_KL_BK_<i> (the bias bK) and ROMAN_KL_DZ_L<i> (photo-z shift of
    the lens n(z)). A parameter absent from params counts as 0 unless a
    default is named here. params_lens.yaml sets DZ_L<i> equal to DZ_S<i>:
    the lenses are the source sample.

    Arguments:
      **params = {name: value} of the point, as Cobaya passes them to logp.

    Returns:
      nothing; changes the nuisance state behind ci.
    """
    ntomo = self.lens_ntomo
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # n(z) is sent at every evaluation, so a user function can change
        # it per point (for example to add outliers): copy the stored
        # table (self.lens_nz keeps the unmodified one), change the copy,
        # then hand it over with set_lens_sample
        lens_nz_local = self.lens_nz.copy()

        # a user modification goes here, for example
        # lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # the photo-z shifts still apply on top of the handed n(z); a user
        # function that models them itself would drop this call
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def compute_logp(self, datavector):
    """Return ln L = -chi2/2 for a theory data vector.

    chi2 = (t - d)^T C^-1 (t - d), summed over the entries the mask keeps,
    with t the theory vector, d the dataset's data vector and C^-1 the
    inverse of the masked covariance (computed once, in initialize).
    Every unmasked entry enters: no data compression is applied.

    Arguments:
      datavector = theory vector at full length (masked entries included;
                   their values are ignored), as get_datavector returns it.

    Returns:
      ln L as a float, without the constant normalization term.
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def logp(self, **params):
    """Return ln L at one point: Cobaya's entry point for every evaluation.

    Arguments:
      **params = {name: value} of the sampled and fixed parameters.

    Returns:
      compute_logp(get_datavector(**params)), a float.
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Compute the theory data vector at one point.

    Arguments:
      **params = {name: value} of the sampled and fixed parameters.

    Returns:
      a 1D float64 numpy array at full length, masked entries 0
      (internal_get_datavector). For use_emulator = 1 the emulator path
      is not connected and the result is the 0-d array 0.0.
    """
    if self.use_emulator == 1:
      #dv = self.internal_get_datavector_emulator(**params)
      dv = 0.0
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    """Assemble the data vector from emulated blocks (use_emulator = 1).

    get_datavector does not call this method, and it cannot run with this
    Fourier-space interface: it calls real-space bindings
    (compute_data_vector_3x2pt_real_sizes and the
    compute_add_fpm_3x2pt_real_any_order family) that
    cosmolike_roman_kl_interface does not define. The intended flow: the
    emulator theory blocks return the cosmic-shear, ggl and clustering
    blocks of the probe, which are copied into one vector in that order;
    cosmolike then adds the point-mass terms (ROMAN_KL_PM<i>) and, with
    use_baryon_pca, the baryon principal components.

    Arguments:
      **params = {name: value} of the point.

    Returns:
      the assembled vector, a float64 numpy array.

    Raises:
      ValueError when an emulated block has the wrong length or the probe
      is unknown.
    """
    # ---------------------------------------------------------------
    # fast parameters: m's and pm's are never emulated
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    if not self.use_baryon_pca: 
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the masked theory vector with cosmolike (use_emulator 0 or 2).

    The cosmology goes first (power spectra, growth, distances), then the
    lens nuisance (probes with galaxies), then the source nuisance. Then
    one of three branches:
      create_baryon_pca: compute the baryon principal components, save
        them to filename_baryon_pca, and return the plain vector;
      use_baryon_pca: return the vector plus sum_i Q_i PC_i, with
        Q_i = ROMAN_KL_BARYON_Q<i>;
      otherwise: return the plain vector.
    print_datavector = True also writes the vector to
    print_datavector_file, one "index value" line per entry.

    Arguments:
      **params = {name: value} of the point.

    Returns:
      the theory vector at full length with masked entries 0, as the
      list of floats the interface returns (get_datavector converts it).
    """
    self.set_cosmo_related()
    if self.probe != "xi":
      self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else: 
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
