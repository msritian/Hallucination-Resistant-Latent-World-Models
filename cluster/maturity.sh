#!/bin/bash
# Training-maturity study: record 50 episodes with one model snapshot and dump the detector features (incl. the critic
# floor C). Args: <task> <models_tarball_name> <model_file>   (models tarball: run/snap_*.pt and run/final_model.pt)
TASK=$1; M=$2; F=$3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p m
tar -xzf "$M.tar.gz" -C m
python -m src.preflight.run_preflight --task "$TASK" --grounded_model "m/run/$F" --out maturity --dump_features
status=$?
mkdir -p maturity
tar -czf maturity.tar.gz maturity
echo "done $(date)  exit status $status"
exit $status
