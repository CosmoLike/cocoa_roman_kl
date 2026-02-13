import numpy as np

fname = 'dc1_3x2_roman_real.covmat'
output_fname = 'dc1_3x2_roman_real_cut.covmat'

cov = np.loadtxt(fname, comments='#')
first_line = open(fname).readline().split()

ndim = 5
cov = cov[:ndim, :ndim]
newline = ' '.join(first_line[1:ndim+1])

np.savetxt(output_fname, cov, header=newline, comments='#')