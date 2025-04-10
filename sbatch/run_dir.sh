#!/bin/bash

# Initialize variables
partition="cpu"  # Default to CPU
flag=""
config_path=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --train|--continue|--finetune|--test|--plot)
      if [[ -n "$flag" ]]; then
          echo "Error: Cannot specify multiple modes (--train, --continue, --finetune, --test, --plot)."
          exit 1
      fi
      flag="$1"
      config_path="$2"
      shift
      ;;
    --cpu|--ceewater|--gpu|--gpu-long|--gpupod)
      partition="$1"  
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done


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
    ./run.sh "$partition" "$flag" "$item"
  done
}


# Check if the argument is a directory
if [ -d "$config_path" ]; then
  # Next we need to scan and submit directories and yml files differently.
  case $flag in
    --continue|--test|--plot)
      subdirs=($(find "$config_path" -mindepth 1 -maxdepth 1 -type d -not -name '_*'))
      process_and_submit "${subdirs[@]}"
      ;;
    --train|--finetune)
      yml_files=($(find "$config_path" -maxdepth 1 -type f -name '*.yml'))
      process_and_submit "${yml_files[@]}"
      ;;
    *)
      echo "Unimplemented batch mode: $flag"
      exit 1
      ;;
  esac
else
  echo "Error: Invalid input. Please provide a valid directory"
  exit 1
fi
