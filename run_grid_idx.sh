#!/bin/bash
#SBATCH --job-name=manual_search
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/flexible_hybrid_hyperparam_grid/_slurm_outputs/manual_search_%j.out
#SBATCH -t 14-00:00:00
#SBATCH -p gpu-long
#SBATCH -c 1
#SBATCH --gpus=1 # Request access to 1 GPU
#SBATCH --constraint=sm_61&vram11
#SBATCH --mem=16G

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a task_id as an argument."
    exit 1
fi

TASK_ID=$1

module load miniconda/22.11.1-1
conda activate tss-ml

module load cuda/12.4.0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

SCRIPT_PATH="/work/pi_kandread_umass_edu/tss-ml/src/run.py"
CONFIG_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/flexible_hybrid_hyperparam_grid/base_config.yml"

# Run the Python script with the provided task_id as the grid search index
python $SCRIPT_PATH --grid_search $CONFIG_PATH --grid_index $TASK_ID