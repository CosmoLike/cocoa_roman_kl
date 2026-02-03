#!/bin/bash
#SBATCH --job-name=dv_bary
#SBATCH --output=/xdisk/timeifler/yhhuang/log/dv_bary-%A.out
#SBATCH --error=/xdisk/timeifler/yhhuang/log/dv_bary-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5GB
#SBATCH --export=None
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo $OMP_NUM_THREADS

python ./projects/roman_kl/evaluate_baryon.py

source stop_cocoa.sh
