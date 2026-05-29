#!/bin/bash
# =============================================================================
# File        : setup_venv.sh
# Author      : Zhenyu Xu, zxu3@clemson.edu
# Created     : 2025-03-27
# Description : This is a script to setup the virtual environment for the Quark project.
# =============================================================================

# Set environment name and Python version (if needed)
VENV_DIR="venv"
PYTHON_CMD="python3" # Change to python3.x if necessary

# Check if Python is installed
if ! command -v $PYTHON_CMD &>/dev/null; then
  echo "Error: $PYTHON_CMD is not installed. Please install Python and try again."
  exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment already exists in $VENV_DIR. Skipping creation."
  source $VENV_DIR/bin/activate
else
  # Create virtual environment
  echo "Creating virtual environment in $VENV_DIR..."
  uv venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade
echo "Upgrading uv pip..."
uv pip install --upgrade pip

# Install dependencies if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
  echo "Installing dependencies from requirements.txt..."
  uv pip install -r requirements.txt
else
  echo "No requirements.txt found. Skipping dependency installation."
  uv pip install torch
  uv pip install einops
  uv pip install matplotlib
  uv pip install ipykernel
  uv pip install amd-quark
  uv pip install accelerate
  uv pip install "huggingface-hub[cli]"
  uv pip install git+https://github.com/huggingface/transformers.git

  # GGUF must use llama.cpp version
  git clone https://github.com/ggml-org/llama.cpp.git
  cp -r llama.cpp/gguf-py/gguf ./venv/lib/python3.*/site-packages/
  rm -rf llama.cpp
fi

# Print success message
echo "Virtual environment setup complete. Activate it using:"
echo "source $VENV_DIR/bin/activate"

mkdir ../gguf_files
mkdir ../q4nx_files
