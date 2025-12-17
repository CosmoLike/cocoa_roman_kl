#!/bin/bash
#SBATCH --job-name=DC1_real
#SBATCH --output=/xdisk/timeifler/yhhuang/log/kl-%A.out
#SBATCH --error=/groups/timeifler/yhhuang/log/kl-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5GB
#SBATCH --export=None
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

# path
export MCMC_YAML=./projects/roman_real/EXAMPLE_MCMC1.yaml
export RUN_MODE_FLAG="-r"

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
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
