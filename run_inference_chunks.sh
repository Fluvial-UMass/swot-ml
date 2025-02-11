#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/CONUS/_slurm_outputs/inference_chunks_%A_%a.out
#SBATCH --array=0-15
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus=1 # Request access to 1 GPU
#SBATCH --constraint=sm_61&vram11
#SBATCH --mem=32G

module load conda/latest
conda activate tss-ml

module load cuda/12.6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

SCRIPT_PATH="/work/pi_kandread_umass_edu/tss-ml/src/run.py"
MODEL_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/CONUS/train_all_sites_20241119_011923"

# Run the Python script with the current array task ID as the grid search index
python $SCRIPT_PATH --prediction_model $MODEL_PATH --basin_chunk_index $SLURM_ARRAY_TASK_ID
