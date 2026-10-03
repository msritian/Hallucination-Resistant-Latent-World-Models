#!/bin/bash
# Run-time warning-light experiment on one trained robot (src/monitor.py).
# Args: <task> <grounded_result_name> <episodes> <cal_episodes>
# Exit 85 = checkpoint (./mon is saved by HTCondor and the job restarts); any other exit returns mon as monitor.tar.gz.
TASK=$1; G=$2; EPS=$3; CAL=$4
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "$G.tar.gz" -C g
tar -xzf "real_$G.tar.gz" -C old
python -m src.monitor --task "$TASK" --run grounded --model g/run/final_model.pt \
    --episodes_file old/preflight/episodes_grounded.pt --out mon --episodes "$EPS" --cal_episodes "$CAL" --max_hours 3
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
mkdir -p mon
tar -czf monitor.tar.gz mon
echo "done $(date)  exit status $status"
exit $status
