"""Compute and save the roman_kl galaxy/shear covariance from the command line.

A covariance matrix C holds the expected scatter of the data vector and
the correlations between its entries, C_ij = <(d_i - <d_i>)(d_j - <d_j>)>.
The likelihood reads a supplied matrix (data/*.cv); this runner computes
a forecast matrix for the same Fourier-space layout from the survey
choices in roman_kl_covariance.py, kept as separate components:
Gaussian (G), super-sample (SSC, the response to density modes larger
than the survey) and connected non-Gaussian (cNG), plus their total. No
likelihood file is changed.

run_covariance (cosmolike_notebook_utils/covariance/command_line.py)
does the work through the production bindings of the compiled
interface; the notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb computes the
same matrices through the slower notebook wrappers.

From the Cocoa/ folder (cocoa environment active, start_cocoa.sh
sourced, OMP_NUM_THREADS set to the thread count for cosmolike):

    python ./projects/roman_kl/covariance/compute_covariance.py \\
        ./projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.yaml

--help lists the options (--output, --overwrite, ...); covariance/README.md
explains the measurement spaces and the accuracy settings.
"""

import os
from pathlib import Path
import sys

# Set the thread pools of the numerical libraries (OpenBLAS, MKL, Apple
# Accelerate) to one worker before their first import, so they do not
# compete with cosmolike for cores. cosmolike's own OpenMP team follows
# OMP_NUM_THREADS from the environment.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/roman_kl (parents[1] of this file climbs from
# covariance/ to the project folder); the shared covariance package lives
# in cosmolike_core, the compiled interface in interface/, and this
# folder's roman_kl_covariance.py is found because Python puts the folder
# of the running script on its module search path
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_roman_kl_interface as ci
import roman_kl_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# __name__ equals "__main__" only when this file runs as a script.
# run_covariance reads the yaml path and options from the command line,
# computes G, SSC, cNG and total (Fourier space unless the yaml chooses
# real space; joint=False selects the galaxy/shear layout, True is the
# cluster adapter's) and writes one .npz archive.
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="fourier", joint=False,
    )
