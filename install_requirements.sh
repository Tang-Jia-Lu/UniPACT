#!/bin/bash

# Install all packages in one command with specific CUDA support for torch
echo "Installing all requirements with PyTorch CUDA 11.8 support..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118