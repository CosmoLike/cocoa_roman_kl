"""Embed the Roman KL cosmic-shear data in the shear-family layout.

The Roman kinematic-lensing (KL) forecast supplies a cosmic-shear data
vector (Roman_Ntomo10_KL.datavector) and its covariance
(Roman_ssss_cov_Ncl20_Ntomo10). The cosmolike likelihood of this project
reads the full layout of the shear family instead: 55 shear, 100
galaxy-galaxy lensing and 10 clustering blocks of 20 multipole bands,
3300 entries in that order. This script writes

  roman_kl.datavector    = the shear values first, zeros after them;
  roman_kl.mask          = 1.0 on the shear entries, 0.0 on the padding,
                           so the likelihood ignores the padding;
  Roman_cov_Ncl20_Ntomo10 = every (i, j) with i <= j: the input line
                           where the shear covariance has one, otherwise
                           a placeholder line with variance 1.0 on the
                           diagonal and 0.0 off it.

The placeholders never enter chi2 (the mask removes those entries); the
unit diagonal only keeps the matrix invertible. Each covariance line has
10 columns, "i j ell_i ell_j" then four tomographic-bin indices, then the
Gaussian and the non-Gaussian parts, which cosmolike adds. The covariance
names below carry no suffix; the shipped copies end in .cv.

To run (from the folder holding the two input files):

    python resize.py
"""

import numpy as np

# tomographic bins and multipole bands of the Roman KL data
Ntomo = 10
Ncl = 20

# layout length: (55 shear + 10 clustering + 100 ggl blocks) x 20 bands
size = int((Ntomo * (Ntomo + 1)/2 + Ntomo + Ntomo * Ntomo ) * Ncl)
print('Size = ', size)
datav = np.zeros(size)
mask = np.zeros(size)

index = np.arange(size)
# usecols=(1): the input has "index value" lines; keep the values
dv_KL = np.loadtxt('Roman_Ntomo10_KL.datavector', usecols=(1))
size_KL = len(dv_KL)
datav[:size_KL] = dv_KL
mask[:size_KL] = 1
np.savetxt('roman_kl.datavector', np.column_stack((index, datav)), fmt='%d %1.6e')
np.savetxt('roman_kl.mask', np.column_stack((index, mask)), fmt='%d %1.1f')

cov_fname = 'Roman_ssss_cov_Ncl20_Ntomo10'
# each input line split into its 10 columns, kept as text so the copied
# lines reproduce the input digits
rows = []
with open(cov_fname, 'r') as f:
    for line in f:
        parts = line.strip().split()
        rows.append(parts[:10])
# the index columns as integer arrays (i_in and j_in are not used below)
arr = np.array(rows, dtype=object)
i_in = arr[:, 0].astype(int)
j_in = arr[:, 1].astype(int)

# (i, j) -> the input line, so the loop below can copy each shear pair
data = {}
for r in rows:
    i = int(r[0])
    j = int(r[1])
    data[(i, j)] = r

# placeholder ell and tomographic-bin columns of the padding lines; the
# likelihood reads only i, j and the last two columns
ell1, ell2, tomo1, tomo2, tomo3, tomo4 = (1.0, 1.0, 0, 0, 0, 0)
def fmt_float(x: float) -> str:
    """Format one number as the covariance file does: 6 decimals, exponent form."""
    return f"{float(x):.6e}"

# the output covariance: every pair i <= j of the 3300 entries, once
out_fname = 'Roman_cov_Ncl20_Ntomo10'
with open(out_fname, 'w') as f:
    for i in range(size):
        for j in range(i, size):
            key = (i, j)
            if key in data:
                f.write(' '.join(data[key]) + '\n')
            else:
                if j != i:
                    f.write(f'{i:d} {j:d} {fmt_float(ell1)} {fmt_float(ell2)} {tomo1:d} {tomo2:d} {tomo3:d} {tomo4:d} ' +
                            f'{fmt_float(0.0)} {fmt_float(0.0)}\n')
                else:
                    f.write(f'{i:d} {j:d} {fmt_float(ell1)} {fmt_float(ell2)} {tomo1:d} {tomo2:d} {tomo3:d} {tomo4:d} ' +
                            f'{fmt_float(1.0)} {fmt_float(0.0)}\n')

