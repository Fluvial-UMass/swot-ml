#!/bin/bash
#SBATCH --job-name=train_ensemble
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/swot_q_lumped_era5/ensemble/_slurm_outputs/seed_%a.out
#SBATCH --array=0-7
#SBATCH -t 7-00:00:00
#SBATCH -p gpu
#SBATCH -q long
#SBATCH -c 1
#SBATCH --gpus=1 # Request access to 1 GPU
#SBATCH --constraint=sm_61&vram11
#SBATCH --mem=16G

module load conda/latest
conda activate tss-ml

module load cuda/12.6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

SCRIPT_PATH="/work/pi_kandread_umass_edu/tss-ml/src/run.py"
CONFIG_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/swot_q_lumped_era5/test_ensemble.yml"

# Run the Python script with the current array task ID as the grid search index
python $SCRIPT_PATH --train_ensemble $CONFIG_PATH --ensemble_seed $SLURM_ARRAY_TASK_ID