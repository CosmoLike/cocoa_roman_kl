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

# import CAMB
sys.path.insert(0, os.environ['ROOTDIR']+'/external_modules/code/CAMB/build/lib.linux-x86_64-'+os.environ['PYTHON_VERSION'])
import camb
from camb import model
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

# general matplot settings
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

# settings
CAMBAccuracyBoost = 1.0
non_linear_emul = 2
CLprobe = '3x2pt'
IA_model = 0
IA_redshift_evolution = 0

# evaluate parameters
As_1e9 = 2.128048
ns = 0.96605
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
                       kmax=5.0, k_per_logint=10, CAMBAccuracyBoost=1.0,
                       non_linear_emul=non_linear_emul):

    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    CAMBAccuracyBoost = CAMBAccuracyBoost*AccuracyBoost
    kmax = kmax*(1.0 + 3*(CAMBAccuracyBoost-1))
    k_per_logint = int(k_per_logint) + int(3*(CAMBAccuracyBoost-1))
    extrap_kmax=2.5e2*CAMBAccuracyBoost
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
        # Use Halofit first that works on all redshifts
        lnPNL = np.log(PKNL.P(z_interp_2D,np.power(10.0,log10k_interp_2D)).flatten(order='F'))+np.log((H0/100.0)**3) 
        # on z < 10.0, replace it with EE2
        lnPNL = np.where((z_interp_2D<10)[:,None], lnPL.reshape(len_z_interp_2D,len_log10k_interp_2D,order='F')+lnbt, 
                                                   lnPNL.reshape(len_z_interp_2D,len_log10k_interp_2D,order='F')).ravel(order='F')
    elif non_linear_emul == 2:
        lnPNL = np.log(PKNL.P(z_interp_2D,np.power(10.0,log10k_interp_2D)).flatten(order='F'))+np.log((H0/100.0)**3)  
    log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
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
                     A1  = [0, 0, 0, 0, 0], 
                     A2  = [0, 0, 0, 0, 0],
                     BTA = [0, 0, 0, 0, 0],
                     shear_photoz_bias = [ROMAN_KL_DZ_S1, ROMAN_KL_DZ_S2, ROMAN_KL_DZ_S3, ROMAN_KL_DZ_S4, ROMAN_KL_DZ_S5,
                                          ROMAN_KL_DZ_S6, ROMAN_KL_DZ_S7, ROMAN_KL_DZ_S8, ROMAN_KL_DZ_S9, ROMAN_KL_DZ_S10],
                     M = [ROMAN_KL_M1, ROMAN_KL_M2, ROMAN_KL_M3, ROMAN_KL_M4, ROMAN_KL_M5,
                          ROMAN_KL_M6, ROMAN_KL_M7, ROMAN_KL_M8, ROMAN_KL_M9, ROMAN_KL_M10],
                     baryon_sims = None,
                     AccuracyBoost = 1.0, 
                     kmax = 7.5, 
                     k_per_logint = 10, 
                     CAMBAccuracyBoost = CAMBAccuracyBoost,
                     CLAccuracyBoost = 1.0, 
                     CLIntegrationAccuracy=0,
                     non_linear_emul=non_linear_emul):

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
        ci.init_baryons_contamination(sim = baryon_sims)        
    return ci.C_ss_tomo_limber(l = ell)
def plot_C_ss_tomo_limber(ell, C_ss, C_ss_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, 
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = (0.6,0.78), yaxislabelsize = 16, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 15, figsize = (12, 12), 
                          save = None, colorbar=1):

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
    path = '/groups/timeifler/yhhuang/CosmoLike/cocoa/Cocoa/projects/roman_kl/data/'
    data_file = 'roman_kl.dataset'
    data_path = '/xdisk/timeifler/yhhuang/roman_kl/data/'
    basename = 'roman_kl_%s.modelvector'
    figname = '/xdisk/timeifler/yhhuang/roman_kl/figures/baryon_contamination_roman_kl.pdf'

    ini = IniFile(os.path.join(path, data_file))
    lens_file = ini.relativeFileName('nz_lens_file')
    source_file = ini.relativeFileName('nz_source_file')
    lens_ntomo = ini.int('lens_ntomo')
    source_ntomo = ini.int('source_ntomo')
    n_cl = ini.int('n_cl')
    l_min = ini.float('l_min')
    l_max = ini.float('l_max')
    
    ci.initial_setup()
    ci.init_accuracy_boost(1,0, int(1))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=lens_file,
        lens_ntomo=int(lens_ntomo), 
        source_multihisto_file=source_file,
        source_ntomo=int(source_ntomo))
    ci.init_IA(ia_model=int(IA_model), IA_redshift_evolution=int(IA_redshift_evolution))

    ell = np.logspace(np.log10(l_min), np.log10(l_max), n_cl)
    param = ('TNG100','HzAGN','mb2','illustris','eagle','owls_AGN_T80','owls_AGN_T85',
             'owls_AGN_T87', 'BAHAMAS_T76','BAHAMAS_T78','BAHAMAS_T80')

    C_ss = []
    for x in param:
        print('Calculating for ', x)
        dv = C_ss_tomo_limber(ell=ell, baryon_sims=x)
        C_ss.append(dv)
        fname = os.path.join(data_path, basename % x)
        np.savetxt(fname, np.column_stack((np.arange(0, len(dv)), dv)), fmt='%d %.6e')
        print('Saved ', fname)

    C_ss_ref = C_ss_tomo_limber(ell=ell)
    fname = os.path.join(data_path, basename % 'dmo')
    np.savetxt(fname, np.column_stack((np.arange(0, len(C_ss_ref)), C_ss_ref)), fmt='%d %.6e')
    print('Saved ', fname)
    # plot figures
    plot_C_ss_tomo_limber(ell=ell, C_gs=C_ss, C_gs_ref=C_ss_ref, lmin=ell[0], lmax=ell[len(ell)-1], 
                          cmap="twilight_shifted",  bintextpos = [0.15, 0.2], ylim = [0.61,1.07], 
                          legend = param, legendloc=(0.9,0.55), 
                          linewidth=[1.0, 1.3, 1.6, 1.9], linestyle = ['solid', 'dashed', 'dashdot', 'dotted'],
                          figsize = (18, 12), bintextsize = 20, yaxislabelsize = 17, 
                          yaxisticklabelsize = 14, xaxisticklabelsize = 20, 
                          save=figname)
    print('Figure saved at ', figname)