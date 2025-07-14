#!/bin/bash
# Batch submit script for new Typer CLI hydrological model
#
# Usage:
#   ./run_dir.sh <partition> <mode> <config_dir> [extra_args]
#
# Modes and arguments:
#   train <config_dir> [partition]
#   train_ensemble <config_dir> <ensemble_seed> [partition]
#   grid_search <config_dir> <grid_index> [partition]
#   smac_optimize <config_dir> <smac_runs> <smac_workers> [partition]
#   test <training_dir> [partition]
#   attribute <training_dir> [partition]
#   predict <model_dir> <basin_chunk_index> [partition]
#
# Examples:
#   ./run_dir.sh gpu train config/
#   ./run_dir.sh cpu train_ensemble config/ 42
#   ./run_dir.sh gpu-long grid_search config/ 3
#   ./run_dir.sh gpupod smac_optimize config/ 10 4
#   ./run_dir.sh cpu test runs/
#   ./run_dir.sh ceewater attribute runs/
#   ./run_dir.sh gpu predict runs/ 5


# Parse arguments: <partition> <mode> <config_path> [extra_args]
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <partition> <mode> <config_dir> [extra_args]"
  exit 1
fi


partition="$1"
mode="$2"
config_path="$3"
# Capture all extra args after the third positional argument
shift 3
extra_args=("$@")



process_and_submit() {
  local items=("$@")
  local count=${#items[@]}

  if [ $count -eq 0 ]; then
    echo "Error: No valid items found in the specified directory."
    exit 1
  fi

  read -p "Found $count item(s). Do you want to submit all jobs? (y/n) " answer
  if [ "$answer" != "y" ]; then
    echo "Job submission canceled."
    exit 0
  fi

  # If confirmed, submit each job
  for item in "${items[@]}"; do
    ./sbatch/run.sh "$partition" "$mode" "$item" "${extra_args[@]}"
  done
}


# Check if the argument is a directory
if [ -d "$config_path" ]; then
  case $mode in
    train|train_ensemble|grid_search|smac_optimize)
      yml_files=($(find "$config_path" -maxdepth 1 -type f -name '*.yml'))
      process_and_submit "${yml_files[@]}"
      ;;
    test|attribute|predict)
      subdirs=($(find "$config_path" -mindepth 1 -maxdepth 1 -type d -not -name '_*'))
      process_and_submit "${subdirs[@]}"
      ;;
    *)
      echo "Unimplemented batch mode: $mode"
      exit 1
      ;;
  esac
else
  echo "Error: Invalid input. Please provide a valid directory"
  exit 1
fi
