#!/bin/bash

# Check if the argument is a directory
if [ -d "$1" ]; then
  # Initialize a count for valid files
  valid_files=0

  # Loop over each yml file
  for config_file in "$1"/*.yml; do
    if [ -f "$config_file" ]; then
      ((valid_files++))
    fi
  done

  # Check if no valid files were found
  if [ $valid_files -eq 0 ]; then
    echo "Error: No valid .yml file found in the directory."
    exit 1
  fi

  read -p "Found $valid_files .yml file(s). Do you want to submit all jobs? (y/n) " answer
  if [ "$answer" != "y" ]; then
    echo "Job submission canceled."
    exit 0
  fi

  # If confirmed, submit each job
  for config_file in "$1"/*.yml; do
      ./unity_train.sh --train "$config_file"
  done
  
else
  echo "Error: Invalid input. Please provide a valid directory"
  exit 1
fi
