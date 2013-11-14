#!/bin/bash -l
mkdir -p tmp_mini_data
mkdir -p tmp_data

module load R

#SBATCH --job-name=blb_pre_process
#SBATCH --output=dump/blb_pre_process.out
#SBATCH --error=dump/blb_pre_process.err

srun R --no-save --vanilla < BLB_pre_process.R



