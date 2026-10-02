#!/bin/bash
# Phase 2 preflight on two trained runs. Args: <task> <grounded_result_name> <stock_result_name> [extra run_preflight flags]
TASK=$1; G=$2; S=$3; shift 3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p g s
tar -xzf "$G.tar.gz" -C g
tar -xzf "$S.tar.gz" -C s
python -m src.preflight.run_preflight --task "$TASK" \
    --grounded_model g/run/final_model.pt --stock_model s/run/final_model.pt --out preflight "$@"
status=$?
mkdir -p preflight
tar -czf preflight.tar.gz preflight
echo "done $(date)  exit status $status"
exit $status
