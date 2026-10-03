#!/bin/bash
# Model rollout vs simulator rollout (same start state, same actions). Args: <result_name> <run> <task>
NAME=$1; RUN=$2; TASK=$3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p m
tar -xzf "$NAME.tar.gz" -C m
python -m src.tools.rollout_vs_sim --model m/run/final_model.pt --run "$RUN" --task "$TASK" --out rvs
status=$?
mkdir -p rvs
tar -czf rvs.tar.gz rvs
echo "done $(date)  exit status $status"
exit $status
