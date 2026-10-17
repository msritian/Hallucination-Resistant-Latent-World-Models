#!/bin/bash
# Offline TD-MPC2 training from the stored episodes of an earlier run (src/train.py --offline_data).
# Args: <task> <source_result_name> <offline_episodes> <updates> <seed> [extra train args]
# Exit 85 = checkpoint (./run is saved by HTCondor and the job restarts).
TASK=$1; SRC=$2; N=$3; U=$4; SEED=$5; shift 5
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p data/$SRC
[ -f data/$SRC/run/checkpoint.pt ] || tar -xzf "$SRC.tar.gz" -C data/$SRC run/checkpoint.pt
python -m src.train --run stock --task "$TASK" --seed "$SEED" --steps "$U" --out_dir run \
    --offline_data data/$SRC/run/checkpoint.pt --offline_episodes "$N" --max_hours 3 --compile_mode default \
    --video_freq 100000000 --ckpt_freq 25000 "$@"
status=$?
[ "$status" -eq 85 ] && exit 85
rm -f run/checkpoint.pt
mkdir -p run && tar -czf run.tar.gz run
echo "done $(date)  exit status $status"
exit $status
