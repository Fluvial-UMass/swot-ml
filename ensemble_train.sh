#!/bin/bash

# Check if the argument is a directory or file
if [ -d "$1" ]; then
  # It's a directory, initialize a count for valid files
  valid_files=0

  # Loop over each yml file
  for config_file in "$1"/*.yml; do
    if [ -f "$config_file" ]; then
      sbatch unity_train.sh "$config_file"  # Submit a new job for each file using the submission script
      ((valid_files++))
    fi
  done

  # Check if no valid files were found
  if [ $valid_files -eq 0 ]; then
    echo "Error: No valid .yml files found in the directory."
    exit 1
  fi

elif [ -f "$1" ]; then
  # It's a file, check if it exists
  if [ ! -f "$1" ]; then
    echo "Error: Configuration file does not exist."
    exit 1
  fi

  sbatch job_submit.sh "$1"  # Submit the job using the submission script
  
else
  echo "Error: Invalid input. Please provide a valid path to yml file or directory of yml file(s)."
  exit 1
fi
