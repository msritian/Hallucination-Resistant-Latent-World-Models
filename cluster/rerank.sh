#!/bin/bash
# Idea 1 pilot on one robot (src/rerank.py). Args: <task> <grounded_result_name> <episodes> <cal_episodes>
TASK=$1; G=$2; EPS=$3; CAL=$4
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "$G.tar.gz" -C g
tar -xzf "real_$G.tar.gz" -C old
python -m src.rerank --task "$TASK" --model g/run/final_model.pt --episodes_file old/preflight/episodes_grounded.pt \
    --out rr --episodes "$EPS" --cal_episodes "$CAL" --max_hours 8
status=$?
[ "$status" -eq 85 ] && exit 85
mkdir -p rr && tar -czf rerank.tar.gz rr
echo "done $(date)  exit status $status"
exit $status
