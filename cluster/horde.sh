#!/bin/bash
# Track 2: Horde per-signal audits on one robot (src/horde.py)
TASK=$1; G=$2; EPS=$3; CAL=$4; shift 4  # CAL unused
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "$G.tar.gz" -C g
tar -xzf "real_$G.tar.gz" -C old
python -m src.horde --task "$TASK" --model g/run/final_model.pt --episodes_file old/preflight/episodes_grounded.pt \
    --out hd --train_ckpt g/run/checkpoint.pt "$@"
status=$?
[ "$status" -eq 85 ] && exit 85
mkdir -p hd && tar -czf shift.tar.gz hd
echo "done $(date)  exit status $status"
exit $status
