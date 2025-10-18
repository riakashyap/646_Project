#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOKS_DIR="$ROOT_DIR/notebooks"
SCRIPTS_DIR="$ROOT_DIR/src"

mkdir -p $SCRIPTS_DIR
for notebook in $NOTEBOOKS_DIR/*.ipynb; do
    base_name=$(basename "$notebook" .ipynb)
    echo "tangling $notebook into $SCRIPTS_DIR/$base_name.py..."
    jupyter nbconvert --no-prompt --to python --output "$SCRIPTS_DIR/$base_name" "$notebook"

    # delete code marked as export ignored
    sed -i '/# EXPORT-IGNORE-START/,/# EXPORT-IGNORE-END/d' "$SCRIPTS_DIR/$base_name.py"
    # delete full-line comments (makes diffing harder)
    sed -i '/^\s*#/d' "$SCRIPTS_DIR/$base_name.py"
    # remove extra empty lines
    sed -i '/^$/N;/^\n$/D' "$SCRIPTS_DIR/$base_name.py"
    # add copyright
    sed -i '1i\
# Copyright (c) 2025 bdunahu, Ria. All rights reserved.\
# Use of this source code is governed by an MIT license\
# that can be found in the LICENSE file.' "$SCRIPTS_DIR/$base_name.py"
done

echo "all notebooks tangled."
