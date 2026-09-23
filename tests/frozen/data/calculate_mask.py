import numpy as np

#VM Fourier-space mask generator for the roman_kl project.
#VM Regenerates data/roman_kl.mask, data/ones.mask and the frozen
#VM ones_shear.mask exactly. The project uses two data-vector shapes
#VM (n_cl = 20 log-spaced C_ell bins in [20, 4000], 10 lens = 10 source
#VM tomographic bins, no per-ell scale cuts):
#VM   shear family (roman_kl_mcmc.dataset, ggl_exclude = []):
#VM     55 shear + 100 ggl + 10 clustering = 165 blocks x 20 = 3300
#VM   3x2pt family (roman_kl_3x2.dataset, ggl_exclude = lens >= source):
#VM     55 shear + 45 ggl + 10 clustering = 110 blocks x 20 = 2200

#VM INPUT (from data/*.dataset and likelihood/*.yaml) ----------------------
N_CL   = 20  # Number of C_ell bins (log-spaced in [l_min, l_max])
N_LENS = 10  # Number of lens tomographic bins
N_SRC  = 10  # Number of source tomographic bins

#VM GLOBAL VARIABLES -------------------------------------------------------
N_SHEAR = int(N_SRC * (N_SRC + 1) / 2)  # 55 shear blocks

def nblocks(ggl_exclude):
  "Total number of [shear, ggl, clustering] blocks given the ggl exclusions"
  return N_SHEAR + (N_LENS * N_SRC - len(ggl_exclude)) + N_LENS

def save_mask(filename, mask):
  np.savetxt(filename,
    np.column_stack((np.arange(0, len(mask)), mask)),
    fmt='%d %1.1f')

#VM SHEAR FAMILY (3300 = 165 blocks x 20 ells, ggl_exclude = []) -----------
NDATA_SHEAR_FAMILY = nblocks([]) * N_CL

# roman_kl.mask: cosmic-shear-only analysis on the full-shape vector; the
# 55 shear blocks are kept and every ggl and clustering block is zeroed
mask = np.zeros(NDATA_SHEAR_FAMILY)
mask[0 : N_SHEAR * N_CL] = 1.0
save_mask("roman_kl.mask", mask)

save_mask("ones_shear.mask", np.ones(NDATA_SHEAR_FAMILY))

#VM 3x2PT FAMILY (2200 = 110 blocks x 20 ells) -----------------------------
# ggl_exclude (likelihood yaml) drops every lens >= source pair (55 pairs)
GGL_EXCLUDE = [[i, j] for i in range(N_LENS) for j in range(i + 1)]
NDATA_3X2PT_FAMILY = nblocks(GGL_EXCLUDE) * N_CL

save_mask("ones.mask", np.ones(NDATA_3X2PT_FAMILY))
