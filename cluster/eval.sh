#!/bin/bash
# Evaluate planner arms on one trained run. Args: <result_name> <run: stock|grounded> <task> <preset> [extra args]
NAME=$1; RUN=$2; TASK=$3; PRESET=$4; shift 4
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p m
tar -xzf "$NAME.tar.gz" -C m
python -m src.evaluate --model m/run/final_model.pt --run "$RUN" --task "$TASK" --preset "$PRESET" --out evalout "$@"
status=$?
mkdir -p evalout
tar -czf evalout.tar.gz evalout
echo "done $(date)  exit status $status"
exit $status
