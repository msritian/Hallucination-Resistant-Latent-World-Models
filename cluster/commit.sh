#!/bin/bash
# Commitment study on one trained robot (src/commit.py, docs/preregistration_commitment.md).
# Args: <task> <grounded_result_name> <episodes> <cal_episodes>
# Exit 85 = checkpoint (./com is saved by HTCondor and the job restarts); any other exit returns com as commit.tar.gz.
TASK=$1; G=$2; EPS=$3; CAL=$4
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "/staging/s/smittal39/wm-results/$G.tar.gz" -C g run/final_model.pt
tar -xzf "real_$G.tar.gz" -C old
python -m src.commit --task "$TASK" --run grounded --model g/run/final_model.pt \
    --episodes_file old/preflight/episodes_grounded.pt --out com --episodes "$EPS" --cal_episodes "$CAL" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
mkdir -p com
tar -czf commit.tar.gz com
echo "done $(date)  exit status $status"
exit $status
