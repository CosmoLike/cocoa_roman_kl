"""Keep the leading parameters of an MCMC proposal covariance file.

A .covmat file is the parameter covariance a Cobaya MCMC run uses to
shape its proposed steps: a header line "# name1 name2 ..." and then a
square matrix whose rows and columns follow that order. This script keeps
the leading ndim x ndim block and the first ndim names, for a run that
samples only those parameters.

The input and output names are fixed below. They name a roman_real file
(dc1_3x2_roman_real.covmat) that this project does not ship: edit them
before use. To run (from the folder holding the input file):

    python cut_covmat.py
"""

import numpy as np

# input covariance, and the file the cut copy is written to
fname = 'dc1_3x2_roman_real.covmat'
output_fname = 'dc1_3x2_roman_real_cut.covmat'

cov = np.loadtxt(fname, comments='#')
# the header line split at spaces: ["#", name1, name2, ...]
first_line = open(fname).readline().split()

# number of leading parameters kept (rows, columns and header names)
ndim = 5
cov = cov[:ndim, :ndim]
newline = ' '.join(first_line[1:ndim+1])

# np.savetxt writes the kept names back as a "# ..." header line
np.savetxt(output_fname, cov, header=newline, comments='#')