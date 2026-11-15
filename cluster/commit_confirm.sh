#!/bin/bash
# Commitment confirmation study on one robot (src/commit.py --study confirm, docs/preregistration_commitment_confirm.md).
# Args: <task> <grounded_result_name> <real_archive> <episodes> <cal_episodes> <dba_episodes>
# <real_archive> holds preflight/episodes_grounded.pt, or nothing (real_none.tar.gz): then 50 episodes are collected.
# Exit 85 = checkpoint (./com is saved by HTCondor and the job restarts); any other exit returns com as commit.tar.gz.
TASK=$1; G=$2; REAL=$3; EPS=$4; CAL=$5; DBA=$6
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "/staging/s/smittal39/wm-results/$G.tar.gz" -C g run/final_model.pt
tar -xzf "$REAL" -C old
EPF=""
[ -f old/preflight/episodes_grounded.pt ] && EPF="--episodes_file old/preflight/episodes_grounded.pt"
python -m src.commit --study confirm --task "$TASK" --run grounded --model g/run/final_model.pt $EPF \
    --out com --episodes "$EPS" --cal_episodes "$CAL" --dba_episodes "$DBA" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
mkdir -p com
tar -czf commit.tar.gz com
echo "done $(date)  exit status $status"
exit $status
