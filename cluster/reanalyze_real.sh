#!/bin/bash
# Re-analyse the simulator-labelled episodes (saved in results/real_<grounded>.tar.gz) with the current code.
# Args: <task> <grounded_result_name> <stock_result_name>
TASK=$1; G=$2; S=$3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p g s old
tar -xzf "$G.tar.gz" -C g
tar -xzf "$S.tar.gz" -C s
tar -xzf "real_$G.tar.gz" -C old
python -m src.preflight.run_preflight --task "$TASK" --reuse_episodes old/preflight \
    --grounded_model g/run/final_model.pt --stock_model s/run/final_model.pt --out reanalysis
status=$?
mkdir -p reanalysis
tar -czf reanalysis.tar.gz reanalysis
echo "done $(date)  exit status $status"
exit $status
