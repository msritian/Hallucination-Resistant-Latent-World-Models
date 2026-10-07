#!/bin/bash
# Phase A shift detection on one robot (src/shift.py)
TASK=$1; G=$2; EPS=$3; CAL=$4; shift 4  # CAL unused
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "$G.tar.gz" -C g
tar -xzf "real_$G.tar.gz" -C old
python -m src.shift --task "$TASK" --model g/run/final_model.pt --episodes_file old/preflight/episodes_grounded.pt \
    --out sh --episodes "$EPS" "$@"
status=$?
[ "$status" -eq 85 ] && exit 85
mkdir -p sh && tar -czf shift.tar.gz sh
echo "done $(date)  exit status $status"
exit $status
