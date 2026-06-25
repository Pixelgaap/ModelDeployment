#!/usr/bin/env bash
set -euo pipefail

# Base image: PyTorch 2.7.0 + CUDA 11.8.
# Run this inside the offline environment with its internal package index configured.

python -m pip install --upgrade pip
python -m pip install \
  "transformers>=4.41.0" \
  "accelerate>=0.30.0" \
  "fastapi>=0.111.0" \
  "uvicorn[standard]>=0.30.0" \
  "pydantic>=2.0.0" \
  "librosa>=0.10.0" \
  "soundfile>=0.12.1"

# MODEL_PATH is optional. By default the service loads the model from this directory.
# export DEVICE=auto
# export BATCH_SIZE=8
#
# Start service:
# uvicorn fastapi_app:app --host 0.0.0.0 --port 8080
