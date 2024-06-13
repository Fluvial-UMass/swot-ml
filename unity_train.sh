#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path based on named argument

# Initialize variables
train_config=""
continue_dir=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train) train_config="$2"; shift ;;
        --continue) continue_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate inputs
if [[ -n "$train_config" && -n "$continue_dir" ]]; then
    echo "Error: Cannot specify both --train and --continue."
    exit 1
elif [[ -n "$train_config" ]]; then
    if [[ ! -f "$train_config" ]]; then
        echo "Error: Training configuration file does not exist: $train_config"
        exit 1
    fi
    config_path="$train_config"
    config_basename=$(basename "$train_config" .yml)
    python_flag="--train"
elif [[ -n "$continue_dir" ]]; then
    if [[ ! -d "$continue_dir" ]]; then
        echo "Error: Directory does not exist: $continue_dir"
        exit 1
    fi
    config_path="$continue_dir"
    config_basename=$(basename "$continue_dir")
    python_flag="--continue"
else
    echo "Error: Must specify either --train <config-file> or --continue <directory>."
    exit 1
fi

# Define the output directory and create it if it doesn't exist
config_dir=$(dirname "$config_path")
config_dir=$(realpath "$config_dir")
output_dir="${config_dir}/_slurm_outputs"
mkdir -p "$output_dir"

# Create the SBATCH script with dynamic output path
sbatch_script=$(mktemp)
cat << EOF > "$sbatch_script"
#!/bin/bash
#SBATCH --job-name=${config_basename}
#SBATCH -c 4 # Number of Cores per Task
#SBATCH --mem=64G  # Requested Memory
#SBATCH -p ceewater_kandread-cpu # Partition    
#SBATCH -t 14-00:00:00  # Job time limit
#SBATCH -o ${output_dir}/${config_basename}.out

module load miniconda/22.11.1-1
conda activate tss-ml

# Run Python script with provided path
python /work/pi_kandread_umass_edu/tss-ml/src/run.py $python_flag "$config_path"
EOF

# Submit the job
# cat $sbatch_script #for viewing the script.
sbatch "$sbatch_script"
