#!/bin/bash
#SBATCH --job-name=proposal
#SBATCH --output=/xdisk/timeifler/yhhuang/log/%x-%A.out
#SBATCH --error=/xdisk/timeifler/yhhuang/log/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=4
#SBATCH --export=None
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu
#SBATCH --partition=standard
#SBATCH --account=timeifler

# path
if [ -z "$1"]; then
    echo "Error: missing YAML inout."
    echo "Usage: sbatch puma_run_MCMC.sh <YAML>"
    exit 1
fi
export MCMC_YAML=$1
export RUN_MODE_FLAG="-r"

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo MCMC YAML is ${MCMC_YAML}
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
module purge > /dev/null 2>&1
module load anaconda
conda init bash
source ~/.bashrc
conda activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPTION_MPI_OS="--oversubscribe"
export OPTION_MPI_PML="--mca pml ob1"
export OPTION_MPI_BTL="--mca btl vader,tcp,self"
export OPTION_MPI_BIND="--bind-to core:overload-allowed"
export OPTION_MPI_RANK="--rank-by slot"
export OPTION_MPI_MAP="--map-by numa:pe=${OMP_NUM_THREADS}"
echo $OMP_NUM_THREADS

mpirun -n ${SLURM_NTASKS} ${OPTION_MPI_BTL} ${OPTION_MPI_PML} \
    ${OPTION_MPI_OS} ${OPTION_MPI_BIND} ${OPTION_MPI_RANK} ${OPTION_MPI_MAP} \
    cobaya-run ${MCMC_YAML} ${RUN_MODE_FLAG}
