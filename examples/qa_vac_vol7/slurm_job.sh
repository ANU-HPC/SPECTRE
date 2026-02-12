#!/bin/sh 
#SBATCH --signal=SIGUSR1@300
#SBATCH --qos=serial
#SBATCH --nodes 1
#SBATCH --ntasks 7
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=6G
#SBATCH --time 24:00:00
#SBATCH --job-name="spectre_eq"
#SBATCH --output out.txt
# #SBATCH --partition dcgp_fua_prod # Only for Pitagora/Cineca

# Usage: ./slurm_job_new.sh init.toml   (do NOT prefix with sbatch)

# The exact slurm parameters will be cluster-dependent

# If not inside a Slurm allocation, submit myself and exit
if [ -z "$SLURM_JOB_ID" ]; then
    # Job name follows the .toml file name
    STEM="${1%.toml}"

    # Submit the slurm job, which is this script again
    sbatch --parsable --job-name="spectre_eq ${STEM}" "$0" "$1" > slurmid
    exit $?
fi

ulimit -s unlimited # important, otherwise risk getting seg fault due to stack overflow

export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/src_spectre.sh

srun calc_spectre_eq.py $1 

rm slurmid
