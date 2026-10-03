#!/bin/bash
# Ground-truth planning diagnostics on one trained run. Args: <result_name> <run: stock|grounded> <task> [extra args]
NAME=$1; RUN=$2; TASK=$3; shift 3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
mkdir -p m
tar -xzf "$NAME.tar.gz" -C m
python -m src.tools.plan_diag --model m/run/final_model.pt --run "$RUN" --task "$TASK" --out diag "$@"
status=$?
mkdir -p diag
tar -czf diag.tar.gz diag
echo "done $(date)  exit status $status"
exit $status
