"""Write the Fourier-space mask files of the roman_kl datasets.

A mask file has one line per data-vector entry, "index value", with
value 1.0 for an entry the likelihood uses and 0.0 for one it ignores
(chi2 sums over the 1.0 entries only). Run in data/, this script
rewrites roman_kl.mask, ones_shear.mask and ones.mask byte for byte;
the copies under tests/frozen/data/ are pinned by the test manifest and
change only when the frozen state is regenerated.

The project uses two data-vector layouts. Both have n_cl = 20 multipole
bands, log-spaced between l = 20 and 4000, 10 lens and 10 source
tomographic bins, and no per-band scale cuts. The entries are ordered
shear blocks, then galaxy-galaxy lensing (ggl) blocks, then clustering
blocks, 20 bands per block:

  shear family (roman_kl_mcmc.dataset, ggl_exclude = []):
    55 shear + 100 ggl + 10 clustering = 165 blocks x 20 = 3300 entries
  3x2pt family (roman_kl_3x2.dataset, ggl_exclude = every pair whose
  lens index >= source index):
    55 shear + 45 ggl + 10 clustering = 110 blocks x 20 = 2200 entries

To run (from projects/roman_kl/data/, any Python with numpy):

    python calculate_mask.py
"""
import numpy as np

# ---- input: must equal n_cl, lens_ntomo and source_ntomo of data/*.dataset
N_CL   = 20  # Number of C_ell bins (log-spaced in [l_min, l_max])
N_LENS = 10  # Number of lens tomographic bins
N_SRC  = 10  # Number of source tomographic bins

# ---- derived sizes: the shear pairs (i, j) with i <= j number
# N_SRC (N_SRC + 1)/2
N_SHEAR = int(N_SRC * (N_SRC + 1) / 2)  # 55 shear blocks

def nblocks(ggl_exclude):
  """Count the blocks of one layout: shear, then ggl, then clustering.

  Arguments:
    ggl_exclude = the excluded [lens, source] pairs, as in the yaml key.

  Returns:
    N_SHEAR + (N_LENS*N_SRC - len(ggl_exclude)) + N_LENS, an int.
  """
  return N_SHEAR + (N_LENS * N_SRC - len(ggl_exclude)) + N_LENS

def save_mask(filename, mask):
  """Write one mask file: "index value" lines, the value printed as 1.0 or 0.0.

  np.column_stack pairs the entry indices 0, 1, 2, ... with the mask
  values as the two columns of the file.

  Arguments:
    filename = output path, relative to the working directory.
    mask     = 1D array of 0.0 and 1.0, one value per data-vector entry.

  Returns:
    nothing; the file is overwritten.
  """
  np.savetxt(filename,
    np.column_stack((np.arange(0, len(mask)), mask)),
    fmt='%d %1.1f')

# ---- shear family: 3300 entries = 165 blocks x 20 bands, ggl_exclude = []
NDATA_SHEAR_FAMILY = nblocks([]) * N_CL

# roman_kl.mask: the cosmic-shear-only analysis on the shear-family layout;
# the 55 shear blocks are kept and every ggl and clustering block is zeroed
mask = np.zeros(NDATA_SHEAR_FAMILY)
mask[0 : N_SHEAR * N_CL] = 1.0
save_mask("roman_kl.mask", mask)

# ones_shear.mask: every entry of the shear-family layout kept (the
# no-scale-cut variant the tests read through --mask=ones)
save_mask("ones_shear.mask", np.ones(NDATA_SHEAR_FAMILY))

# ---- 3x2pt family: 2200 entries = 110 blocks x 20 bands
# ggl_exclude of the likelihood yamls: the list comprehension gives [i, j]
# for every lens i and every source j <= i, 1 + 2 + ... + 10 = 55 pairs
GGL_EXCLUDE = [[i, j] for i in range(N_LENS) for j in range(i + 1)]
NDATA_3X2PT_FAMILY = nblocks(GGL_EXCLUDE) * N_CL

# ones.mask: every entry of the 3x2pt layout kept (roman_kl_3x2.dataset)
save_mask("ones.mask", np.ones(NDATA_3X2PT_FAMILY))
