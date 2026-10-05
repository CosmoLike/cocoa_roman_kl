# Table of contents <a name="table_of_contents"></a>

1. [Running Cosmolike projects (Basic instructions)](#roman_kl_running_cosmolike_projects)
2. [Baryonic feedback on EXAMPLE_EVALUATE1](#roman_kl_baryonic_feedback)
3. [Running Hybrid Cosmolike-ML emulators](#roman_kl_examples_emul2)
4. [Unit tests](#unit_tests)
5. [Computing covariances](#computing_covariances)

## Running Cosmolike projects (Basic instructions) <a name="roman_kl_running_cosmolike_projects"></a> 

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.**
>
> Run production and HPC calculations from YAML through the optimized
> `_interface` bindings. Notebook `_wrapper` APIs expose intermediate
> quantities for exploration; copying and rearranging their arrays adds
> overhead. Both routes call the same C kernels.
>
> In a matched **LSST Y1 covariance** test on an M2 Pro with eight threads,
> the CLI averaged **68.34 s** (three runs); one wrapper run took **177.74 s**.
> The CLI was **2.60× faster**, with bitwise-identical covariance components.
> See [the production covariance CLI](#computing_covariances).

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_KL_CODE=1
>
>     (...)
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export ROMAN_KL_URL="https://github.com/CosmoLike/cocoa_roman_kl.git"
>     export ROMAN_KL_NAME="roman_kl"
>     #Pin the project version with at most one of the keys below (COMMIT, BRANCH, or TAG).
>     #If more than one is set, COMMIT wins over BRANCH, and BRANCH wins over TAG.
>     #If none is set, Cocoa loads the latest commit on the repository default branch.
>     #export ROMAN_KL_GIT_BRANCH="main"
>     #export ROMAN_KL_GIT_COMMIT="abc"
>     export ROMAN_KL_GIT_TAG="v4.11.0"

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile roman_kl project)
>
>       source ./projects/roman_kl/scripts/compile_roman_kl.sh

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/roman_kl` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning] 
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_kl/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/roman_kl/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_kl/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/roman_kl/EXAMPLE_MCMC1.yaml -f


> [!Warning]
> CosmoLike supports the optimized strict-IEEE default build and
> `COSMOLIKE_DEBUG_MODE`. The compiler mode `COSMOLIKE_AGGRESSIVE_MODE`
> is retired because its fast-math configuration produced incorrect
> covariance inverses. Unset that variable before compiling.
> Do not enable `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`,
> `-fassociative-math`, `-ffinite-math-only`, `-freciprocal-math`,
> `-fno-signed-zeros`, or `-fno-trapping-math` in CosmoLike builds.
> This does not change Cocoa's separate `--aggressive` download option.


# Baryonic feedback on EXAMPLE_EVALUATE1 <a name="roman_kl_baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback suppression to the
matter power spectrum via the `bfmt` theory block (SP(k), BCEmu, Flamingo, BACCOemu,
or BCemu2025). By default, the example runs without feedback.

**Step :one:**: ensure the lines below are commented out in `set_installation_options.sh`
before running `setup_cocoa.sh` and `compile_cocoa.sh`. *By default, these lines should
be commented out, but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt` theory block and select
the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the `roman_kl.cosmic_shear`
likelihood block.

**Step :four:**: uncomment the selected model's parameters in the `params` block and in
the `sampler: evaluate: override` block (the example carries a commented block for each
model).

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and the `bfmt`
> options, see `Cocoa/external_modules/code/baryon_suppression/README.md`.

# Running Hybrid Cosmolike-ML emulators <a name="roman_kl_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

While our data vector emulators are incredibly fast, there is an intermediate 
approach that emulates only the Boltzmann outputs (comoving distance, linear and 
nonlinear matter power spectrum). This hybrid-ML case can offer greater flexibility, 
especially in the initial phases of a research project, as changes to the modeling 
of nuisance parameters or to the assumed galaxy distributions do not require 
retraining of the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags on `set_installation_options.sh` are similar to what we showed in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: Remove GPU (idea is to run emulators on the CPU!)

  - Linux

        export CUDA_VISIBLE_DEVICES=""

 **Step :four:** Run `cobaya-run` on the first emulator example, following the commands below.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_kl/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/roman_kl/EXAMPLE_EMUL2_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_kl/EXAMPLE_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/roman_kl/EXAMPLE_EMUL2_MCMC1.yaml -r

> [!NOTE]
> **Running on more than one node.** The flag `--mca btl vader,tcp,self` works unchanged across
> nodes: Open MPI picks the transport per pair of ranks, using shared memory (`vader`) within a
> node and TCP between nodes. Three things deserve attention on multi-node runs:
>
> 1. **Network interface.** The TCP layer must not select an interface that is not routable
>    between compute nodes. The flag `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0` excludes
>    the common offenders. TCP bandwidth is not a limitation for our workloads, which exchange
>    small, infrequent MPI messages.
>
> 2. **Environment.** Ranks on remote nodes must see Cocoa's environment (`ROOTDIR`, `PATH`,
>    `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, the OpenMP/BLAS thread settings, and
>    `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN`). Slurm forwards the submitting environment
>    automatically; the explicit `-x` flags in our sbatch templates repeat this so the
>    scripts also work under ssh-based launchers. No other Cocoa installation flags are read at runtime.
>
> 3. **Slurm geometry.** Keep `ntasks-per-node` × `cpus-per-task` no larger than the cores per
>    node, and use `--map-by numa:pe=${OMP_NUM_THREADS}` so each rank reserves the cores its
>    OpenMP threads will use.

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

## Historical Notes

- The corresponding `cosmolike_core` branch for the `generic_interface.cpp` parser fix is `nonlimber-dev`.
- A typical failure signature is:
  `[critical] read_table: failed to parse file external_modules/data/roman_kl/Roman_3x2pt_cov_Ncl20_Ntomo10 at data line 12401, column 9: token='1.630861e-316' (out of range)`
- `Roman_3x2pt_cov_Ncl20_Ntomo10` can contain extremely small subnormal entries such as `1.630861e-316`.
- The original `read_table` implementation in `cosmolike_core/cosmolike/generic_interface.cpp` used `std::stod`, which raised a range error on these finite underflowed values and stopped likelihood initialization.
- The fix is to parse table values with `std::strtod` and only treat range errors as fatal when the parsed result is non-finite. Finite underflowed values are accepted.
- If you see this error, switch `Cocoa/external_modules/code/cosmolike_core` to branch `nonlimber-dev` or apply the same `generic_interface.cpp` patch, then rebuild `projects/roman_kl/interface/cosmolike_roman_kl_interface.so`.

# Unit tests <a name="unit_tests"></a>

The `tests/` folder holds unit tests for the likelihoods of this
project: they compare each likelihood against stored reference
values, check for race conditions from OpenMP threading, and measure
the numerical error of the default accuracy settings, and the accuracy of the
hybrid emulated pipelines. The
tests read nothing from the live project;
[tests/README.md](tests/README.md) describes every test, the tests'
own data snapshot, and how to refresh it.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/roman_kl/tests/data_vector

## Minimum accuracy parameters

The advisory checks in `tests/data_vector/test_accuracy.py` measure the
numerical error of the default accuracy settings: each setting is
raised one at a time on the 3x2pt configuration, so a large
$\Delta\chi^2$ can be attributed to the setting causing it, and
then every setting at once.

Each check prints the $\Delta\chi^2$
between the high-accuracy and the default evaluations, to compare
against the 0.2 band the reference tests allow. No measured values
are quoted here: rerun the checks to measure them on the current
code, and see [tests/README.md](tests/README.md) for each check,
the settings raised, and what each setting controls.

The examples default to `k_per_logint: 50`: the old default
undersampled the CAMB transfer functions, and that error masqueraded
as an apparent CAMB `AccuracyBoost` sensitivity until the transfer
sampling was raised.

# Computing covariances <a name="computing_covariances"></a>

[EXAMPLE_EVALUATE_COVARIANCE.ipynb](EXAMPLE_EVALUATE_COVARIANCE.ipynb)
computes a covariance with this project's 3×2pt measurement layout.
It keeps G, SSC and cNG separately, applies the supplied likelihood mask,
and plots the computed and supplied totals together.

| Measurement choice | Notebook example |
| --- | --- |
| Dataset | [data/roman_kl_3x2.dataset](data/roman_kl_3x2.dataset) |
| Primary space | Fourier-space 3×2pt |
| Lens bins | 10 |
| Source bins | 10 |
| Bins per two-point observable | 20, multipoles 20–4000 |
| Generated entries before cuts | 2,200 |
| Entries after the dataset mask | 2,200 |

The primary example is the 20-band Fourier 3×2pt configuration in
`roman_kl_3x2.dataset`, also used by `EXAMPLE_EVALUATE2.yaml`. It has ten lens
and ten source bins and retains only galaxy–shear pairs with source index
greater than lens index. The low shape dispersion is the kinematic-lensing
forecast choice. The cosmic-shear-only `roman_kl_mcmc.dataset` is a different
configuration.

The default [installation options](../../set_installation_options.sh) set
`IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE=1`. This leaves covariance-generation
kernels and notebook bindings out of the compiled interface. Likelihoods still
read and invert their supplied covariance matrices. The steps below enable
covariance generation for this build; comment out that export in
`set_installation_options.sh` to keep it enabled in later sessions.
Recompile after changing the option, then restart any running notebook kernel.

We assume Cocoa and this project are installed, users have run
`conda activate cocoa`, the shell is Bash, and the current folder is
`cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable covariance generation and compile the project interface.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE
    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE
    source ./projects/roman_kl/scripts/compile_roman_kl.sh

**Step :three:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :four:**: open the printed URL and select
`projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :five:**: inspect the survey inputs and keep `boosts = [1]` for the
first calculation, then select **Kernel → Restart Kernel and Run All Cells**.
Set `boosts = [1, 2]` to add the accuracy comparison.

The notebook writes `covariance/forecast_fourier.npz`,
`covariance/forecast_camb.npz` and
`covariance/forecast_likelihood_selection.npz`. The last archive retains
both cut totals and the original data-vector indices.
Set `spaces = ["real", "fourier"]` to compute both transformations; only the native space is compared with the supplied likelihood.
The [covariance guide](covariance/README.md) describes the physical inputs,
component plots, accuracy controls and covariance-only tests.

> [!NOTE]
> The generated matrix is an analogous forecast, not a reproduction of the
> supplied likelihood covariance. Gaussian spectra can include non-Limber
> gg/gs and NLA/TATT; SSC/cNG retain zero-IA Limber physics. The forecast
> uses massless neutrinos and a spherical-cap footprint.
> `accuracy_boost` refines
> tables and cutoffs; `integration_accuracy` separately selects precomputed
> GSL rules from [covariance/default.yaml](covariance/default.yaml).

## Command-line calculation

The Python runner computes the full galaxy–shear covariance in Fourier space,
using the optimized production interface. It saves G, SSC, cNG and their
sum without plotting or opening a notebook. Numerical kernels and survey
settings are shared with the notebook calculation.

From Bash in `cocoa/Cocoa`, with `conda activate cocoa`:

**Step :one:**: activate Cocoa and enable covariance generation.

    source start_cocoa.sh
    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE
    unset IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE

**Step :two:**: compile the project interface.

    source ./projects/roman_kl/scripts/compile_roman_kl.sh

**Step :three:**: inspect the YAML cosmology and compute the matrix components.

    export OMP_NUM_THREADS=8
    python ./projects/roman_kl/covariance/compute_covariance.py \
        ./projects/roman_kl/EXAMPLE_EVALUATE_COVARIANCE.yaml

The `.npz` archive contains the full matrix before likelihood scale cuts,
its components, measurement ordering, resolved settings and stage timings.
Existing output files require `--overwrite`; likelihood inputs are separate.

Set `covariance.space` to `real` or `fourier` to select the measurement.

The [evaluate YAML](EXAMPLE_EVALUATE_COVARIANCE.yaml) uses Cobaya's YAML reader, with familiar
`theory`, `params`, `sampler: evaluate` and `output` blocks. Fixed parameter
values specify one cosmology; a parameter with a prior must be supplied
explicitly in `sampler.evaluate.override`. No MCMC or random prior draw runs.

In its `covariance` block, `accuracy_boost: 2` refines the project's
`default.yaml` baseline. `integration_accuracy: 1` changes the quadrature
level independently. Internal accuracy controls can also be set there.
Use `space` for the measurement space. Set the OpenMP team with
`OMP_NUM_THREADS` in the shell; no thread count belongs in the YAML.

`theory.camb.extra_args` supports `AccuracyBoost`, `kmax`, `k_per_logint`,
`lens_potential_accuracy` and `halofit_version`. CAMB's boost controls
CAMB; the covariance boost controls its own tables and cutoffs.

Paths in the YAML are relative to the working directory, `cocoa/Cocoa`.
`output` names the `.npz` archive; `--output` can override it for an HPC
job. Set `OMP_NUM_THREADS` in that job’s environment. `--help` lists the
command options.

To return to a data-vector-only build, use the following steps from
`cocoa/Cocoa` with `conda activate cocoa` and Bash.

**Step :one:**: activate Cocoa.

    source start_cocoa.sh

**Step :two:**: omit covariance generation and rebuild the interface.

    unset IGNORE_COSMOLIKE_ROMAN_KL_CODE
    export IGNORE_COSMOLIKE_ROMAN_KL_COVARIANCE=1
    source ./projects/roman_kl/scripts/compile_roman_kl.sh

Gaussian non-Limber and NLA/TATT options are documented in the
[covariance guide](covariance/README.md#choosing-the-gaussian-spectra).
The YAML keeps these Gaussian choices separate from SSC/cNG. OpenMP
threads come exclusively from `OMP_NUM_THREADS`, not from a YAML key.
