#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path based on named argument

# Initialize variables
train_config=""
continue_dir=""
use_gpu=false  # Default to CPU

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train) train_config="$2"; shift ;;
        --continue) continue_dir="$2"; shift ;;
        --test) test_dir="$2"; shift ;; 
        --gpu) use_gpu=true ;;
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
    config_parent=$(basename "$(dirname "$train_config")")
    python_flag="--train"
elif [[ -n "$continue_dir" ]]; then
    if [[ ! -d "$continue_dir" ]]; then
        echo "Error: Directory does not exist: $continue_dir"
        exit 1
    fi
    config_path="$continue_dir"
    config_basename=$(basename "$continue_dir")
    config_parent=$(basename "$(dirname "$continue_dir")")
    python_flag="--continue"
elif [[ -n "$test_dir" ]]; then
    if [[ ! -d "$test_dir" ]]; then
        echo "Error: Directory does not exist: $test_dir"
        exit 1
    fi
    config_path="$test_dir"
    config_basename=$(basename "$test_dir")
    config_parent=$(basename "$(dirname "$test_dir")")
    python_flag="--test"
else
    echo "Error: Must specify either --train <config-file> or --continue <directory>."
    exit 1
fi

# Define the output directory and create it if it doesn't exist
config_dir=$(dirname "$config_path")
config_dir=$(realpath "$config_dir")
output_dir="${config_dir}/_slurm_outputs"
mkdir -p "$output_dir"

# Create the SBATCH script with dynamic output path and partition
sbatch_script=$(mktemp)
cat << EOF > "$sbatch_script"
#!/bin/bash
#SBATCH --job-name="${config_parent}_${config_basename}"
#SBATCH -c 4 # Number of Cores per Task
#SBATCH --mem=64G  # Requested Memory
#SBATCH -t 7-00:00:00  # Job time limit
#SBATCH -o ${output_dir}/${config_basename}.out
EOF

if [ "$use_gpu" = true ]; then
    echo "#SBATCH -p gpu-long # Partition" >> "$sbatch_script"
    echo "#SBATCH --gpus=2080ti:1" >> "$sbatch_script"
else
    echo "#SBATCH -p cpu # Partition" >> "$sbatch_script"
fi

cat << EOF >> "$sbatch_script"
module load miniconda/22.11.1-1
conda activate tss-ml

# Run Python script with provided path
python /work/pi_kandread_umass_edu/tss-ml/src/run.py $python_flag "$config_path"
EOF

# Submit the job
# cat $sbatch_script #for viewing the script.
sbatch "$sbatch_script"
