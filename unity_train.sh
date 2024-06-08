#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path

# Check if argument is provided
if [ "$#" -ne 1 ]; then
    echo "Must provide config file as the first argument."
    echo "Usage: $0 <config-file>"
    exit 1
fi

# Extract directory and base name from the config file
config_dir=$(dirname "$1")
config_basename=$(basename "$1" .yml)


config_dir=$(realpath "$config_dir")
# Define the output directory and create it if it doesn't exist
output_dir="${config_dir}/slurm_outputs"
mkdir -p "$output_dir"

# Create the SBATCH script with dynamic output path
sbatch_script=$(mktemp)
cat << EOF > "$sbatch_script"
#!/bin/bash
#SBATCH -c 4 # Number of Cores per Task
#SBATCH --mem=64G  # Requested Memory
#SBATCH -p cpu # Partition     
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ${output_dir}/${config_basename}_%j.out

module load miniconda/22.11.1-1
conda activate tss-ml

python /work/pi_kandread_umass_edu/tss-ml/src/run.py "$1"
EOF

# Submit the job
sbatch "$sbatch_script"