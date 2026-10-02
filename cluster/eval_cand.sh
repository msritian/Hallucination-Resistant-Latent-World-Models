#!/bin/bash
# Evaluate planner arms with thresholds calibrated on MPPI candidates (from a plan_diag result).
# Args: <result_name> <run> <task> <preset> [extra args]
NAME=$1; RUN=$2; TASK=$3; PRESET=$4; shift 4
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p m d
tar -xzf "$NAME.tar.gz" -C m
tar -xzf "plandiag_$NAME.tar.gz" -C d
python -m src.evaluate --model m/run/final_model.pt --run "$RUN" --task "$TASK" --preset "$PRESET" --out evalout \
    --taus_file d/diag/taus_candidates.json "$@"
status=$?
mkdir -p evalout
tar -czf evalout.tar.gz evalout
echo "done $(date)  exit status $status"
exit $status
