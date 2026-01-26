#!/bin/sh 
#SBATCH --signal=SIGUSR1@180
#SBATCH --qos=serial
#SBATCH --nodes 1
#SBATCH --ntasks 7
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=6G
#SBATCH --time 24:00:00
#SBATCH --job-name="opt spectre_qa_example"
#SBATCH --output out.txt

# An example slurm submission script that is run with `sbatch slurm_job.sh init.sh`
# The exact slurm paramaters will be cluster-dependent

ulimit -s unlimited # important, otherwise risk getting seg fault due to stack overflow

export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/src_spectre.sh

srun calc_spectre_eq.py $1 

rm slurmid

