#!/bin/bash
#SBATCH --job-name=dask_bayes_opt
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/ssf_smac_opt/_slurm_outputs/smac_opt.out
#SBATCH -t 14-00:00:00
#SBATCH -p ceewater_kandread-cpu
#SBATCH -q long
#SBATCH -c 1
#SBATCH --mem=16G

module load conda/latest
conda activate tss-ml

SCRIPT_PATH="/work/pi_kandread_umass_edu/tss-ml/src/run.py"
CONFIG_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/ssf_smac_opt/search.yml"

# Run the Python script with the provided task_id as the grid search index
python $SCRIPT_PATH --smac_optimize $CONFIG_PATH --smac_runs 150 --smac_workers 60