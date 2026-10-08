"""Archived script: cosmic-shear spectra contaminated by baryonic feedback.

For each of eleven hydrodynamical simulations (the names in the
__main__ block: TNG100, Horizon-AGN, MassiveBlack-II, Illustris, EAGLE,
three OWLS-AGN and three BAHAMAS heating temperatures) this script
computes the Limber shear spectra C_l^ss of every source-bin pair with
the matter power spectrum contaminated by that simulation's baryonic
effect (ci.init_baryons_contamination reads it from
data/baryons_logPkR.h5) and saves each as a .npy file; it then saves the
dark-matter-only spectra as the reference. plot_C_ss_tomo_limber draws
the fractional differences (its call is commented out).

The file is kept in old/ for reference and does not run as it is:
  - ci.init_IA is called without the ia_code argument the current
    interface requires, so the script stops there with a TypeError;
  - the paths of the __main__ block are absolute paths of one cluster
    account, and its roman_kl.dataset is not shipped in data/;
  - CAMB is imported from a Linux build folder
    (lib.linux-x86_64-<PYTHON_VERSION>);
  - with non_linear_emul = 1 it calls euclidemu2.get_boost, the older
    EuclidEmulator2 interface (the likelihood calls get_boost2).
Its cosmology set-up also predates the likelihood's: set_cosmology gets
no omegab, no massive-neutrino inputs and no separate growth grid, and
the growth is sampled at k = 5e-4/Mpc instead of growth_k = 0.05/Mpc.
likelihood/_cosmolike_prototype_base.py is the current reference for
these steps.
"""

import sys, platform, os
import matplotlib
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy
import euclidemu2
import cosmolike_roman_kl_interface as ci
from getdist import IniFile
from scipy.interpolate import interp1d
import itertools
import iminuit
import functools
import warnings
print(sys.version)
print(os.getcwd())

# CAMB's Python package from Cocoa's build folder (a Linux folder name)
sys.path.insert(0, os.environ['ROOTDIR']+'/external_modules/code/CAMB/build/lib.linux-x86_64-'+os.environ['PYTHON_VERSION'])
import camb
from camb import model
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

# figure style: STIX fonts, LaTeX text, light grid
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['text.usetex'] = True

# settings: all_sims_file holds the baryonic effect of every simulation;
# CAMBAccuracyBoost is CAMB's accuracy factor of this script;
# non_linear_emul 2 = CAMB's Halofit; CLprobe is not used below;
# IA_model 0 = NLA, and IA_redshift_evolution 0 (NO_IA) sets every
# intrinsic-alignment amplitude to zero
all_sims_file = os.path.join(os.environ['ROOTDIR'], 'projects/roman_kl/data/baryons_logPkR.h5')
CAMBAccuracyBoost = 1.05
non_linear_emul = 2
CLprobe = '3x2pt'
IA_model = 0
IA_redshift_evolution = 0

# the evaluation point: cosmology (As_1e9 = 10^9 A_s, H0 in km/s/Mpc,
# mnu in eV, w0pwa = w0 + wa) and the photo-z shifts and shear
# calibrations of the 10 source bins, all zero
As_1e9 = 2.128048
ns = 0.9645
H0 = 67.67
omegab = 0.0491685
omegam = 0.3156
mnu = 0.06
ROMAN_KL_DZ_S1 = 0.0
ROMAN_KL_DZ_S2 = 0.0
ROMAN_KL_DZ_S3 = 0.0
ROMAN_KL_DZ_S4 = 0.0
ROMAN_KL_DZ_S5 = 0.0
ROMAN_KL_DZ_S6 = 0.0
ROMAN_KL_DZ_S7 = 0.0
ROMAN_KL_DZ_S8 = 0.0
ROMAN_KL_DZ_S9 = 0.0
ROMAN_KL_DZ_S10 = 0.0
ROMAN_KL_M1 = 0.0
ROMAN_KL_M2 = 0.0
ROMAN_KL_M3 = 0.0
ROMAN_KL_M4 = 0.0
ROMAN_KL_M5 = 0.0
ROMAN_KL_M6 = 0.0
ROMAN_KL_M7 = 0.0
ROMAN_KL_M8 = 0.0
ROMAN_KL_M9 = 0.0
ROMAN_KL_M10 = 0.0
w0pwa = -1.0
w = -1.0

# functions
def get_camb_cosmology(omegam = omegam, omegab = omegab, H0 = H0, ns = ns, 
                       As_1e9 = As_1e9, w = w, w0pwa = w0pwa, AccuracyBoost=1.0, 
                       kmax=10.0, k_per_logint=10, CAMBAccuracyBoost=CAMBAccuracyBoost,
                       non_linear_emul=non_linear_emul):
    """Run CAMB and return the tables of the old ci.set_cosmology call.

    The keyword defaults are the module-level values above; mnu (0.06
    eV, one massive neutrino) comes from the module level too. The CAMB
    settings grow with the boost: CAMBAccuracyBoost is multiplied by
    AccuracyBoost, kmax by 1 + 3 (boost - 1), and k_per_logint rises by
    3 (boost - 1).

    Arguments:
      omegam, omegab, H0, ns, As_1e9, w, w0pwa = the cosmology.
      AccuracyBoost, CAMBAccuracyBoost = accuracy multipliers.
      kmax         = CAMB's largest k [1/Mpc] before the boost.
      k_per_logint = CAMB's k samples per log interval before the boost.
      non_linear_emul = 1 for the EuclidEmulator2 boost below z = 10
                        (Halofit above), 2 for Halofit everywhere.

    Returns:
      (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D,
      chi): log10 of k in h/Mpc, the power-spectrum redshifts, ln P_lin
      and ln P_nl in (Mpc/h)^3 flattened with z fastest, the growth
      G = D (1 + z) on z_interp_2D divided by its last value, the
      distance redshifts, and the comoving distance in Mpc/h.
    """

    # one-line helper formulas (lambda = an unnamed function): A_s from
    # As_1e9, wa from w0pwa = w0 + wa, and the physical densities
    # omega_b h^2, omega_c h^2 (matter minus baryons minus the neutrino
    # density mnu (3.046/3)^0.75/94.0708) and omega_m h^2
    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    CAMBAccuracyBoost = CAMBAccuracyBoost*AccuracyBoost
    kmax = kmax*(1.0 + 3*(CAMBAccuracyBoost-1))
    k_per_logint = int(k_per_logint) + int(3*(CAMBAccuracyBoost-1))
    extrap_kmax=2.5e2*CAMBAccuracyBoost
    # the z and k grids of the cosmolike tables, a fixed older version of
    # the likelihood's grids
    tmp=1250
    z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp))),
                                  np.linspace(3.0,50.1,max(100,int(0.40*tmp)))),axis=0)
    len_z_interp_1D = len(z_interp_1D)
    tmp=140
    z_interp_2D = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
                                  np.linspace(3.01,50.1,max(30,int(0.25*tmp)))),axis=0)
    len_z_interp_2D = len(z_interp_2D)
    tmp=1500
    log10k_interp_2D = np.linspace(-4.99,2.0,tmp)
    len_log10k_interp_2D = len(log10k_interp_2D)
    
    pars = camb.set_params(H0=H0, 
                           ombh2=omegabh2(omegab, H0), 
                           omch2=omegach2(omegam, omegab, mnu, H0), 
                           mnu=mnu, 
                           omk=0, 
                           tau=0.06,  
                           As=As(As_1e9), 
                           ns=ns, 
                           halofit_version='takahashi', 
                           lmax=10,
                           AccuracyBoost=CAMBAccuracyBoost,
                           lens_potential_accuracy=1.0,
                           num_massive_neutrinos=1,
                           nnu=3.046,
                           accurate_massive_neutrino_transfers=False,
                           k_per_logint=k_per_logint,
                           kmax = kmax);
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');    
    pars.NonLinear = model.NonLinear_both
    pars.set_matter_power(redshifts = z_interp_2D, kmax = kmax, silent = True);
    results = camb.get_results(pars)
    PKL  = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot", nonlinear = False, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    PKNL = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot",  nonlinear = True, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    lnPL = np.log(PKL.P(z_interp_2D,np.power(10.0,log10k_interp_2D)).flatten(order='F'))+np.log((H0/100.0)**3) 
    if non_linear_emul == 1:
        params = { 'Omm'  : omegam, 
                   'As'   : As(As_1e9), 
                   'Omb'  : omegab,
                   'ns'   : ns, 
                   'h'    : H0/100., 
                   'mnu'  : mnu,  
                   'w'    : w, 
                   'wa'   : wa(w0pwa, w)
                 }
        # get_boost is the older EuclidEmulator2 interface (the likelihood
        # calls get_boost2 with a pre-built emulator)
        kbt, tmp_bt = euclidemu2.get_boost(params,z_interp_2D[z_interp_2D < 10.0],10**np.linspace(-2.0589,0.973,len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')  
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(log10k_interp_2D-np.log10(H0/100.)) #h/Mpc
        tmp[:,10**(log10k_interp_2D-np.log10(H0/100)) < 8.73e-3] = 0.0
        lnbt = np.zeros((len_z_interp_2D, len_log10k_interp_2D))
        lnbt[z_interp_2D < 10.0, :] = tmp
        # start from CAMB's Halofit, which covers every redshift, ...
        lnPNL = np.log(PKNL.P(z_interp_2D,np.power(10.0,log10k_interp_2D)).flatten(order='F'))+np.log((H0/100.0)**3) 
        # ... and use P_lin times the EE2 boost at z < 10
        lnPNL = np.where((z_interp_2D<10)[:,None], lnPL.reshape(len_z_interp_2D,len_log10k_interp_2D,order='F')+lnbt, 
                                                   lnPNL.reshape(len_z_interp_2D,len_log10k_interp_2D,order='F')).ravel(order='F')
    elif non_linear_emul == 2:
        lnPNL = np.log(PKNL.P(z_interp_2D,np.power(10.0,log10k_interp_2D)).flatten(order='F'))+np.log((H0/100.0)**3)  
    log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
    # growth G = D (1 + z) from the linear power at k = 5e-4/Mpc, divided
    # by its value at the last z node (the likelihood samples it at
    # growth_k = 0.05/Mpc on its dense 1D grid instead)
    G_growth = np.sqrt(PKL.P(z_interp_2D,0.0005)/PKL.P(0,0.0005))*(1 + z_interp_2D)
    G_growth = G_growth/G_growth[len(G_growth)-1]
    chi = results.comoving_radial_distance(z_interp_1D) * (H0/100.)
    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi)

def C_ss_tomo_limber(ell, 
                     omegam = omegam, 
                     omegab = omegab, 
                     H0 = H0, 
                     ns = ns, 
                     As_1e9 = As_1e9, 
                     w = w, 
                     w0pwa = w0pwa,
                     A1  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                     A2  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     BTA = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     shear_photoz_bias = [ROMAN_KL_DZ_S1, ROMAN_KL_DZ_S2, ROMAN_KL_DZ_S3, ROMAN_KL_DZ_S4, ROMAN_KL_DZ_S5,
                                          ROMAN_KL_DZ_S6, ROMAN_KL_DZ_S7, ROMAN_KL_DZ_S8, ROMAN_KL_DZ_S9, ROMAN_KL_DZ_S10],
                     M = [ROMAN_KL_M1, ROMAN_KL_M2, ROMAN_KL_M3, ROMAN_KL_M4, ROMAN_KL_M5,
                          ROMAN_KL_M6, ROMAN_KL_M7, ROMAN_KL_M8, ROMAN_KL_M9, ROMAN_KL_M10],
                     baryon_sims = None,
                     AccuracyBoost = 1.0, 
                     kmax = 10, 
                     k_per_logint = 10, 
                     CAMBAccuracyBoost = CAMBAccuracyBoost,
                     CLAccuracyBoost = 1.0, 
                     CLIntegrationAccuracy=0,
                     non_linear_emul=non_linear_emul):
    """Return the Limber shear spectra C_l^ss of every source-bin pair.

    Runs CAMB (get_camb_cosmology), hands the tables and the nuisance
    parameters to cosmolike, then contaminates the matter power spectrum
    with one simulation's baryonic effect, or removes any contamination
    when baryon_sims is None.

    Arguments:
      ell = 1D array of multipoles.
      omegam, omegab, H0, ns, As_1e9, w, w0pwa = the cosmology.
      A1, A2, BTA = intrinsic-alignment amplitudes, one per source bin.
      shear_photoz_bias = photo-z shift of each source bin.
      M = multiplicative shear calibration of each source bin.
      baryon_sims = None (dark matter only) or a simulation name of
                    baryons_logPkR.h5.
      AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
      non_linear_emul = the CAMB settings (see get_camb_cosmology).
      CLAccuracyBoost, CLIntegrationAccuracy = cosmolike accuracy; both
                    grow with AccuracyBoost (the integration level by
                    3 (boost - 1)).

    Returns:
      the array ci.C_ss_tomo_limber returns, [n_ell, n_source, n_source].
    """

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi) = get_camb_cosmology(omegam=omegam, 
                                                                                                  omegab=omegab, 
                                                                                                  H0=H0, 
                                                                                                  ns=ns, 
                                                                                                  As_1e9=As_1e9, 
                                                                                                  w=w, 
                                                                                                  w0pwa=w0pwa,
                                                                                                  AccuracyBoost=AccuracyBoost,
                                                                                                  kmax=kmax,
                                                                                                  k_per_logint=k_per_logint,
                                                                                                  CAMBAccuracyBoost=CAMBAccuracyBoost,
                                                                                                  non_linear_emul=non_linear_emul)
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost-1.0)))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))

    ci.set_cosmology(omegam=omegam, 
                     H0 = H0, 
                     log10k_2D = log10k_interp_2D, 
                     z_2D = z_interp_2D, 
                     lnP_linear = lnPL,
                     lnP_nonlinear = lnPNL,
                     G = G_growth,
                     z_1D = z_interp_1D,
                     chi = chi)
    ci.set_nuisance_shear_calib(M = M)
    ci.set_nuisance_shear_photoz(bias = shear_photoz_bias)
    ci.set_nuisance_ia(A1 = A1, A2 = A2, B_TA = BTA)

    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        print('Baryon sim = ', baryon_sims)
        ci.init_baryons_contamination(sim = baryon_sims, allsims = all_sims_file)        
    return ci.C_ss_tomo_limber(l = ell)
def plot_C_ss_tomo_limber(ell, C_ss, C_ss_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, 
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = (0.6,0.78), yaxislabelsize = 16, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 15, figsize = (12, 12), 
                          save = None, colorbar=1):
    """Plot C_l^ss of every source-bin pair, or its ratio to a reference.

    One panel per pair (i <= j) in an ntomo x ntomo grid, the panels
    below the diagonal switched off; one curve per entry of C_ss, colored
    along cmap. Without C_ss_ref the panels show l (l + 1) C_l/(2 pi) on
    log axes; with it they show C_ss/C_ss_ref - 1 on one shared linear
    axis. The x-axis label sits on row index 4 only (a five-bin layout).

    Arguments:
      ell = 1D array of multipoles [n_ell].
      C_ss = list of arrays [n_ell, ntomo, ntomo], one per curve.
      C_ss_ref = None, or the reference array of the same shape.
      param = None, or one value per curve for the colorbar.
      colorbarlabel = colorbar title.
      lmin, lmax = x-axis range.
      cmap = colormap of the curves.
      ylim = (lower, upper): factors on the curve range without a
             reference; with one, the band (lower - 1, upper - 1).
      linestyle, linewidth = lists cycled over the curves.
      legend = None, or one label per curve; legendloc = its position.
      yaxislabelsize, yaxisticklabelsize, xaxisticklabelsize = font sizes.
      bintextpos, bintextsize = position (axes fraction) and size of the
             pair label.
      figsize = figure size in inches.
      save = None, or the file name the figure is saved to.
      colorbar = None to omit the colorbar when param is given.

    Returns:
      0 after printing "Bad Input" for inconsistent inputs; None after
      saving; (fig, axes) when save is None.
    """

    nell, ntomo, ntomo2 = C_ss[0].shape
    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
      
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    if not (C_ss_ref is None):
        nell2, ntomo3, ntomo4 = C_ss_ref.shape
        if (ntomo3 != ntomo4) or (nell != nell2):
            print(f"notomo = {ntomo}, ntomo_REF = {ntomo3}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   
        
    if C_ss_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.5)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(C_ss):
            print("Bad Input")
            return 0

    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])

    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
    
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                clmin = []
                clmax = []
                for Cl in C_ss:  
                    tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi)
                    clmin.append(np.min(tmp))
                    clmax.append(np.max(tmp))
     
                axes[j,i].set_xlim([lmin, lmax])
                
                if C_ss_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                    axes[j,i].set_yscale('log')
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                    axes[j,i].set_yscale('linear')
                    
                axes[j,i].set_xscale('log')
                
                if i == 0:
                    if C_ss_ref is None:
                        axes[j,i].set_ylabel("$\ell (\ell+1) C_{\ell}^{EE}/(2 \pi)$", fontsize=yaxislabelsize)
                    else:
                        axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)
                
                if j == 4:
                    axes[j,i].set_xlabel(r"$\ell$", fontsize=16)
                
                axes[j,i].text(bintextpos[0], bintextpos[1], 
                    "$(" +  str(i) + "," +  str(j) + ")$", 
                    horizontalalignment = 'center', 
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = axes[j,i].transAxes)
                
                for x, Cl in enumerate(C_ss):
                    if C_ss_ref is None:
                        tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi)
                    else:
                        tmp = Cl[:,i,j] / C_ss_ref[:,i,j] - 1
                    lines = axes[j,i].plot(ell, tmp, 
                                           color=cm(x/len(C_ss)), 
                                           linewidth=next(linewidthcycler), 
                                           linestyle=next(linestylecycler))
    
    if not (legend is None):
        if len(legend) != len(C_ss):
            print("Bad Input")
            return 0
        fig.legend(
            legend, 
            loc=legendloc,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    if not (save is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(save)
    else:
        return (fig, axes)
    
if __name__ == "__main__":
    # absolute paths of one cluster account (they exist nowhere else) and
    # a roman_kl.dataset that data/ does not ship: edit them before use
    path = '/groups/timeifler/yhhuang/CosmoLike/cocoa/Cocoa/projects/roman_kl/data/'
    data_file = 'roman_kl.dataset'
    data_path = '/xdisk/timeifler/yhhuang/roman_kl/data/'
    basename = 'roman_kl_%s'
    figname = '/xdisk/timeifler/yhhuang/roman_kl/figures/baryon_contamination_roman_kl.pdf'

    ini = IniFile(os.path.join(path, data_file))
    lens_file = ini.relativeFileName('nz_lens_file')
    source_file = ini.relativeFileName('nz_source_file')
    lens_ntomo = ini.int('lens_ntomo')
    source_ntomo = ini.int('source_ntomo')
    n_cl = ini.int('n_cl')
    l_min = ini.float('l_min')
    l_max = ini.float('l_max')
    
    # an older cosmolike init sequence: init_IA below lacks the ia_code
    # argument the current interface requires, so the run stops there
    ci.initial_setup()
    ci.init_accuracy_boost(1.0, int(1))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=lens_file,
        lens_ntomo=int(lens_ntomo), 
        source_multihisto_file=source_file,
        source_ntomo=int(source_ntomo))
    ci.init_IA(ia_model=int(IA_model), ia_redshift_evolution=int(IA_redshift_evolution))

    # n_cl band centers, log-spaced between l_min and l_max, as in
    # cosmolike's Fourier binning
    dlogl = (np.log(l_max) - np.log(l_min)) / n_cl
    ell = np.exp(np.arange(np.log(l_min), np.log(l_max), dlogl) + 0.5*dlogl)
    param = ('TNG100-1','HzAGN-1','mb2-1','illustris-1','eagle-1','owls_AGN_t80','owls_AGN_t85',
             'owls_AGN_t87', 'BAHAMAS_t76','BAHAMAS_t78','BAHAMAS_t80')

    # one C_ss per simulation, saved as roman_kl_<simulation>.npy, then
    # the dark-matter-only reference, roman_kl_dmo.npy
    C_ss = []
    for x in param:
        dv = C_ss_tomo_limber(ell=ell, baryon_sims=x)
        C_ss.append(dv)
        fname = os.path.join(data_path, basename % x)
        np.save(fname, dv)
        print('Saved ', fname)

    C_ss_ref = C_ss_tomo_limber(ell=ell)
    fname = os.path.join(data_path, basename % 'dmo')
    np.save(fname, C_ss_ref)
    print('Saved ', fname)
    # plot figures: the call below is disabled; uncomment it to draw
    # C_ss/C_ss_ref - 1 for every simulation
    #plot_C_ss_tomo_limber(ell=ell, C_ss=C_ss, C_ss_ref=C_ss_ref, lmin=ell[0], lmax=ell[len(ell)-1], 
    #                      cmap="twilight_shifted",  bintextpos = [0.15, 0.2], ylim = [0.61,1.07], 
    #                      legend = param, legendloc=(0.9,0.55), 
    #                      linewidth=[1.0, 1.3, 1.6, 1.9], linestyle = ['solid', 'dashed', 'dashdot', 'dotted'],
    #                      figsize = (18, 12), bintextsize = 20, yaxislabelsize = 17, 
    #                      yaxisticklabelsize = 14, xaxisticklabelsize = 20, 
    #                      save=figname)
    #print('Figure saved at ', figname)
