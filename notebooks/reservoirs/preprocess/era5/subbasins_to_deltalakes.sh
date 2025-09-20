#!/bin/bash
#SBATCH --job-name=extract_era5_to_lakehouse
#SBATCH --output=%j.out
#SBATCH -t 7-00:00:00
#SBATCH -p ceewater_cjgleason-cpu
#SBATCH -q long
#SBATCH -A pi_cjgleason_umass_edu
#SBATCH -c 1
#SBATCH --mem=32G
set -e

cd /nas/cee-water/cjgleason/ted/swot-ml
source .venv/bin/activate


python /nas/cee-water/cjgleason/ted/swot-ml/notebooks/reservoirs/preprocess/era5/flipped_extract_sub_basins.py \
    --basin-file /nas/cee-water/cjgleason/ted/swot-ml/data/reservoirs/metadata/All_MERIT_matchups.gpkg \
    --save-dir /nas/cee-water/cjgleason/ted/swot-ml/data/reservoirs/deltalakes/training \
    --n-workers 32 \
    --batch-size 32 
