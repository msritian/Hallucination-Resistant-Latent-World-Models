#!/bin/bash
# Value-error diagnostic on a preflighted pair. Args: <task> <grounded_result> <stock_result> <preflight_result>
TASK=$1; G=$2; S=$3; P=$4
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
mkdir -p g s p
tar -xzf "$G.tar.gz" -C g
tar -xzf "$S.tar.gz" -C s
tar -xzf "$P.tar.gz" -C p
python -m src.preflight.diagnose --task "$TASK" --grounded_model g/run/final_model.pt --stock_model s/run/final_model.pt \
    --episodes_dir p/preflight --out diagnostic
status=$?
mkdir -p diagnostic
tar -czf diagnostic.tar.gz diagnostic
echo "done $(date)  exit status $status"
exit $status
