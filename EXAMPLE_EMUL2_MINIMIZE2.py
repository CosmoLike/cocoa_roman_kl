"""Search for the best-fit point of a roman_kl hybrid example.

A hybrid example (use_emulator: 2 in the likelihood block) replaces
the Boltzmann code CAMB with trained emulators, neural networks that
predict the expansion history and the matter power spectra from the
cosmological parameters; cosmolike still computes the survey's
angular power spectra.

Example 2 is the 3x2pt likelihood (roman_kl.combo_3x2pt: cosmic shear,
galaxy-galaxy lensing and galaxy clustering) of
EXAMPLE_EMUL2_EVALUATE2.yaml.

This file is a thin entry point: run() of
external_modules/code/cosmolike_core/cocoa_hybrid_sampling.py does
the work, and that module's docstring explains the method and every
command-line option.

mode="minimize" searches for the smallest -2 log posterior (the
best-fit point) with an annealed ensemble of emcee walkers (points
that move through parameter space together).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced), first a check that evaluates the yaml's
fiducial point (the parameter values of its sampler.evaluate.override
block) and stops, then a run on two MPI processes (mpirun
starts two copies of this script that share the work; --bind-to none
lets each copy use several cores for cosmolike's OpenMP threads):

    python ./projects/roman_kl/EXAMPLE_EMUL2_MINIMIZE2.py --check
    mpirun -n 2 --bind-to none python ./projects/roman_kl/EXAMPLE_EMUL2_MINIMIZE2.py

Results go to projects/roman_kl/chains/ (--outroot names them; an
existing record is never overwritten). The project README lists the
remaining options.
"""

from pathlib import Path
import sys

# project = the folder of this file (projects/roman_kl); parents[1]
# climbs two levels to Cocoa/, and the / operator of a Path joins
# path pieces
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
# put cosmolike_core first on Python's module search path, so the
# import below finds the shared sampler module there
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# __name__ equals "__main__" only when this file runs as a script, so
# importing it starts nothing
if __name__ == "__main__":
    run(mode="minimize", project=project, example=2)
