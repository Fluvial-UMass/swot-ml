#!/bin/bash
# SLURM directives
#SBATCH -c 4 # Number of Cores per Task
#SBATCH --mem=128G  # Requested Memory
#SBATCH -p gpu # Partition
#SBATCH --gpus=m40:1     
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o /work/pi_kandread_umass_edu/tss-ml/runs/_slurm_out/%j.out

module load miniconda/22.11.1-1
conda activate tss-ml

# Pass config file as the argument
python /work/pi_kandread_umass_edu/tss-ml/src/run.py $1