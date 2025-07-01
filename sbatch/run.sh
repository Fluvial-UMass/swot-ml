#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path based on named argument

# Initialize variables
partition_name="cpu"  # Default to CPU
flag=""
config_path=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train|--continue|--finetune|--test|--attribution)
            if [[ -n "$flag" ]]; then
                echo "Error: Cannot specify multiple modes (--train, --continue, --finetune, --test, --attribution)."
                exit 1
            fi
            flag="$1"
            config_path="$2"
            shift
            ;;
        --cpu|--ceewater|--gpu|--gpu-long|--gpupod)
            # Strip the initial '--' and use the remainder as partition_name
            partition_name="${1:2}"  
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Validate input
if [[ -z "$flag" ]]; then
    echo "Error: Must specify one of --train <config-file.yml>, --continue <directory>, or --test <directory>."
    exit 1
fi

case "$flag" in
    --train)
        if [[ ! -f "$config_path" ]]; then
            echo "Error: Training configuration file does not exist: $config_path"
            exit 1
        fi 
        ;;
    *)
        if [[ ! -d "$config_path" ]]; then
            echo "Error: Directory does not exist: $config_path"
            exit 1
        fi
        ;;
esac


# Set the partition and runtime args based on partition_name
n_workers=1
SBATCH_DIRECTIVES=""
ENVIRONMENT_LINES=""
case $partition_name in
    cpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+1))\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p cpu\n"
        ENVIRONMENT_LINES+="export JAX_PLATFORMS=cpu\n"
        ;;
    ceewater)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+1))\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 7-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p ceewater_kandread-cpu\n"
        ENVIRONMENT_LINES+="export JAX_PLATFORMS=cpu\n"
        ;;
    gpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=1\n"
        SBATCH_DIRECTIVES+="#SBATCH --constraint=sm_61&vram11\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8\n"
        ;;
    gpu-long)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu\n"
        SBATCH_DIRECTIVES+="#SBATCH -q long\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=2080ti:1\n"
        # SBATCH_DIRECTIVES+="#SBATCH --constraint=sm_61&vram11\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8\n"
        ENVIRONMENT_LINES+="nvidia-smi -L\n"
        ;;
    gpupod)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers\n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpupod-l40s\n"
        SBATCH_DIRECTIVES+="#SBATCH -q gpu-quota-16\n"
        SBATCH_DIRECTIVES+="#SBATCH -A pi_cjgleason_umass_edu\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=l40s:1\n"
        ENVIRONMENT_LINES+="module load cuda/12.6\n"
        ENVIRONMENT_LINES+="export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8\n"
        ENVIRONMENT_LINES+="nvidia-smi -L\n"
        ;;
    *)
        echo "Unknown partition type: $partition_name"
        exit 1
        ;;
esac

# Define the output directory and create it if it doesn't exist
config_dir=$(dirname "$config_path")
config_dir=$(realpath "$config_dir")
output_dir="${config_dir}/_slurm_outputs"
mkdir -p "$output_dir"

config_basename=$(basename "$config_path" .yml)
config_parent=$(basename "$(dirname "$config_path")")

# Create the SBATCH script with dynamic output path and partition
sbatch_script=$(mktemp)
cat << EOF > "$sbatch_script"
#!/bin/bash
#SBATCH --job-name="${config_parent}_${config_basename}"
#SBATCH --mem=64G  # Requested Memory
#SBATCH -o ${output_dir}/${config_basename}.out
$(echo -e "$SBATCH_DIRECTIVES")

source .venv/bin/activate

$(echo -e "$ENVIRONMENT_LINES")

cd "$(dirname "$0")/../src"
python run.py $flag $config_path
EOF

# Submit the job.
sbatch "$sbatch_script"

# # View the job script.
# cat $sbatch_script

