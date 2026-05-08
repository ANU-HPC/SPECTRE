#!/bin/sh 

# Usage: ./local_job.sh init.toml   (do NOT prefix with sbatch)

NUM_RANKS=7
export OMP_NUM_THREADS=1

#source ~/src_spectre.sh

nohup mpirun -n $NUM_RANKS calc_spectre_eq.py $1 > out.txt & 
