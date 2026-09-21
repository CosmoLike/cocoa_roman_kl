import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import math as mt

#HB adapted from VM & Surpanta's code

redshift_file = 'example1.nz' #redshift file - change to your own file
ggl_skip_combos = [[6,0],[7,0],[7,1]] #skip these combinations of lens and source bins

ξp_CUTOFF = 0  # cutoff scale in arcminutes
ξm_CUTOFF = 0 # cutoff scale in arcminutes
gc_CUTOFF = 1.5 # Galaxy clustering cutoff in Mpc/h

THETA_MIN  = 2.5    # Minimum angular scale (in arcminutes)
THETA_MAX  = 250.  # Maximum angular scale (in arcminutes)
N_ANG_BINS = 15    # Number of angular bins
N_LENS = 8  # Number of lens tomographic bins
N_SRC  = 8  # Number of source tomographic bins
N_XI_PS = int(N_SRC * (N_SRC + 1) / 2) 
N_XI    = int(N_XI_PS * N_ANG_BINS)

def calculate_average_redshift(filename):
    """
    Reads a text file with redshift bins and corresponding number of data points,
    then calculates the average redshift weighted by the number of data points.

    Parameters:
        filename (str): Path to the input text file.

    Returns:
        float: The weighted average redshift.
    """
    # Load data from the file
    data = np.loadtxt(filename)
    avg_redshifts = []

    # Extract columns
    redshifts = data[:, 0]  # First column: redshift bins
    for i in np.arange(1,9):
        counts = data[:, i]      # Second column: number of data points

        # Compute the weighted average redshift
        avg_redshifts.append(np.sum(redshifts * counts) / np.sum(counts))

    return avg_redshifts

# Calculate average redshifts
zavg = calculate_average_redshift(redshift_file)
    
# COMPUTE SHEAR SCALE CUTS
vtmin = THETA_MIN * 2.90888208665721580e-4;
vtmax = THETA_MAX * 2.90888208665721580e-4;
logdt = (mt.log(vtmax) - mt.log(vtmin))/N_ANG_BINS;
theta = np.zeros(N_ANG_BINS+1)

for i in range(N_ANG_BINS):
  tmin = mt.exp(mt.log(vtmin) + (i + 0.0) * logdt);
  tmax = mt.exp(mt.log(vtmin) + (i + 1.0) * logdt);
  x = 2./ 3.
  theta[i] = x * (tmax**3 - tmin**3) / (tmax**2- tmin**2)
  theta[i] = theta[i]/2.90888208665721580e-4

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
def ang_cut(z):
  "Get Angular Cutoff from redshit z"
  theta_rad = gc_CUTOFF / cosmo.angular_diameter_distance(z).value
  return theta_rad * 180. / np.pi * 60.


#VM COSMIC SHEAR SCALE CUT -------------------------------------------------
ξp_mask = np.hstack([(theta[:-1] > ξp_CUTOFF) for i in range(N_XI_PS)])
ξm_mask = np.hstack([(theta[:-1] > ξm_CUTOFF) for i in range(N_XI_PS)])   

## GGL mask ---------------------------------------------------------------
γt_mask = []  #initialize empty list for γt_mask
for j in range(N_LENS): 
    for k in range(N_SRC):
        if [j,k] in ggl_skip_combos:
            continue
        else:
            γt_mask.append((theta[:-1] > ang_cut(zavg[j])))
γt_mask = np.hstack(γt_mask) 

## w_theta mask -----------------------------------------------------------
w_mask = np.hstack([(theta[:-1] > ang_cut(zavg[j])) for j in range(N_LENS)])
mask = np.hstack([ξp_mask, ξm_mask, γt_mask, w_mask])

## Output mask -------------------------------------------------------------
np.savetxt("example1.mask",
 np.column_stack((np.arange(0,len(mask)),
 mask.astype(int))),fmt='%d %1.1f')

