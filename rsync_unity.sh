#!/bin/bash

SOURCE_DIR="/Users/Ted/Documents/GitHub/tss-ml"
DEST_DIR="/work/pi_kandread_umass_edu/tss-ml"

# Define folders to exclude
EXCLUDE=(
    ".git"
    ".gitignore"
    ".github*"
    ".DS_Store"
    "__pycache__"
    ".ipynb_checkpoints"
)

# Build the exclude options
EXCLUDE_OPTIONS=""
for ex in "${EXCLUDE[@]}"; do
    EXCLUDE_OPTIONS+=" --exclude $ex"
done

# rsync -avp --delete --dry-run $EXCLUDE_OPTIONS "$SOURCE_DIR/" unity:$DEST_DIR
rsync -avp $EXCLUDE_OPTIONS "$SOURCE_DIR/" unity:$DEST_DIR