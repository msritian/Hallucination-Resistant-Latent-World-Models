#!/bin/bash
# Pessimistic planning for one offline-trained robot (src/offline_plan.py). Args: <task> <offline_run_name> <episodes>
TASK=$1; R=$2; EPS=$3; shift 3
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p r && tar -xzf "$R.tar.gz" -C r
python -m src.offline_plan --task "$TASK" --model r/run/final_model.pt --run stock_aux --dataset r/run/offline_episodes.pt \
    --episodes "$EPS" --out op "$@"
status=$?
mkdir -p op && tar -czf op.tar.gz op
echo "done $(date)  exit status $status"
exit $status
