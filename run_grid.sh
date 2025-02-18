#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/swot_q_lumped_era5/grid_search/_slurm_outputs/grid_search_%A_%a.out
#SBATCH --array=101,104,105,106,108,109,183,185,186,189,190,194,198,199,202,203,204,221,222,225,227,228,229,232,233,235,237,240,242,243,244,245
#SBATCH -t 14-00:00:00
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
CONFIG_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/swot_q_lumped_era5/grid_search/grid_base.yml"

# Run the Python script with the current array task ID as the grid search index
python $SCRIPT_PATH --grid_search $CONFIG_PATH --grid_index $SLURM_ARRAY_TASK_ID


# # #SBATCH --array=93-250%64