#!/bin/bash
# Runs inside the container on a GPU execute node.
set -x
nvidia-smi 2>/dev/null || echo "nvidia-smi not in container (fine)"
tar -xzf code.tar.gz
python smoke_env.py 2>&1
cp /opt/pip-freeze.txt pip-freeze.txt
