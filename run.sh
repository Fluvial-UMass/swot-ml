#!/bin/bash
# Wrapper script to submit SLURM job with dynamic output path based on named argument

# Initialize variables
finetune_config=""
partition_name="cpu"  # Default to CPU
flag=""
config_path=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train|--continue|--finetune|--test)
            if [[ -n "$flag" ]]; then
                echo "Error: Cannot specify multiple modes (--train, --continue, --finetune, --test)."
                exit 1
            fi
            flag="$1"
            config_path="$2"
            shift
            ;;
        --cpu|--ceewater|--gpu|--gpu-long)
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

# Check if the path exists and matches flag type
if [[ "$flag" == "--train" ]]; then
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Training configuration file does not exist: $config_path"
        exit 1
    fi
elif [[ "$flag" == "--continue" || "$flag" == "--test" ]]; then
    if [[ ! -d "$config_path" ]]; then
        echo "Error: Directory does not exist: $config_path"
        exit 1
    fi
fi

# Set the partition and runtime args based on partition_name
n_workers=2
SBATCH_DIRECTIVES=""
case $partition_name in
    cpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+2)) # Number of Cores per Task \n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p cpu\n"
        ;;
    ceewater)
        SBATCH_DIRECTIVES+="#SBATCH -c $((n_workers+2)) # Number of Cores per Task \n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p ceewater\n"
        ;;
    gpu)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers # Number of Cores per Task \n"
        SBATCH_DIRECTIVES+="#SBATCH -t 1-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=2080ti:1\n"
        ;;
    gpu-long)
        SBATCH_DIRECTIVES+="#SBATCH -c $n_workers # Number of Cores per Task \n"
        SBATCH_DIRECTIVES+="#SBATCH -t 14-00:00:00\n"
        SBATCH_DIRECTIVES+="#SBATCH -p gpu-long\n"
        SBATCH_DIRECTIVES+="#SBATCH --gpus=2080ti:1\n"
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

module load cuda/12.4.0
module load miniconda/22.11.1-1
conda activate tss-ml
    
# Run Python script with provided path
python /work/pi_kandread_umass_edu/tss-ml/src/run.py $flag $config_path ${finetune_config:+--finetune "$finetune_config"}
EOF

# Submit the job
# cat $sbatch_script #for viewing the script.
sbatch "$sbatch_script"
