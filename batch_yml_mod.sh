#!/bin/bash

# --- Argument Parsing ---
# Script requires 3 arguments: Directory, Property Path, New Value
if [ "$#" -ne 3 ]; then
    # $0 is the name of the script itself
    echo "Usage: $0 <config_dir> <property_path> <new_value>"
    echo "Description: Modifies a specific property in all .yml/.yaml files within the specified directory."
    echo "Arguments:"
    echo "  <config_dir>      Directory containing the YAML files."
    echo "  <property_path>   Yq path to the property (e.g., '.initial_lr', '.model_args.hidden_size')."
    echo "  <new_value>       The new value to set (e.g., 0.005, 128, '\"string val\"', true)."
    echo "  DELETE            Use the literal keyword DELETE to remove the property."
    echo ""
    echo "Example: $0 ./configs .model_args.hidden_size 128"
    echo "Example: $0 ./configs .data_dir \"/new/path/string\""
    echo "Example (Delete): $0 ./configs .optimizer DELETE"
    echo ""
    echo "WARNING: This script modifies files IN-PLACE."
    exit 1
fi

CONFIG_DIR="$1"
PROPERTY_PATH="$2"
ACTION_VALUE="$3" 

# --- Input Validation ---
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configuration directory '$CONFIG_DIR' not found."
    exit 1
fi

# --- yq Value Handling (using environment variable) ---
OPERATION="Modify" # Default operation
YQ_EXPRESSION=""   # Initialize expression variable

if [ "$ACTION_VALUE" = "DELETE" ]; then
    OPERATION="Delete"
    YQ_EXPRESSION="del($PROPERTY_PATH)"
else
    # Safest way to pass command-line args to yq, avoiding shell quoting issues.
    export NEW_VALUE_VAR="$ACTION_VALUE"
    YQ_EXPRESSION="$PROPERTY_PATH = env(NEW_VALUE_VAR)"
    # Uncomment below if you always want the value treated as a string in YAML:
    # YQ_EXPRESSION="$PROPERTY_PATH = strenv(NEW_VALUE_VAR)"
fi


# --- Script Logic ---
# --- Script Logic ---
echo "--- YAML Modification Script ---"
echo "Directory:      $CONFIG_DIR"
echo "Property Path:  $PROPERTY_PATH"
if [ "$OPERATION" = "Delete" ]; then
    echo "Action:         Delete"
else
    echo "Action:         Modify"
    echo "New Value:      '$ACTION_VALUE'" # Display the raw value passed
fi
echo "yq Expression:  $YQ_EXPRESSION"
echo "--------------------------------"
echo "WARNING: Modifying files in-place!"

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: yq command not found. Please install yq (e.g., the Go version from https://github.com/mikefarah/yq)."
    exit 1
fi

# Find YAML files (.yml or .yaml) in the specified directory
shopt -s nullglob # Prevent loop from running if no files match
# Ensure CONFIG_DIR doesn't have a trailing slash for consistency, then add one
config_path="${CONFIG_DIR%/}/" 
files=("$config_path"*.yml "$config_path"*.yaml)
shopt -u nullglob # Turn off nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No *.yml or *.yaml files found in '$CONFIG_DIR'."
    exit 0
fi

echo "Found ${#files[@]} YAML files to process."

success_count=0
error_count=0

for file in "${files[@]}"; do
    echo "$file"

    # Modify the file in-place using eval -i
    # 'eval' helps ensure the expression with environment variables is correctly interpreted by yq.
    # '-i' modifies the file directly.
    yq eval -i "$YQ_EXPRESSION" "$file"

    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        echo "Error modifying '$file'."
        ((error_count++))
    fi
done

echo "--- Script Finished ---"
echo "Processed: ${#files[@]} files in '$CONFIG_DIR'"
echo "Successful modifications: $success_count"
echo "Errors: $error_count"

# Unset the temporary env var
unset NEW_VALUE_VAR

# Exit with 0 if no errors, >0 otherwise (useful for scripting)
exit $error_count