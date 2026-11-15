#!/bin/bash
# Frontier study on one robot (src/commit.py --study frontier, docs/preregistration_frontier.md).
# Args: <task> <grounded_result_name> <confirm_archive> <episodes> <cal_episodes>
# <confirm_archive> is the robot's confirmation result (com/ with lba.pkl, thresholds.json and the budget-4 arms);
# the new arms are added to that directory, so the output holds both studies.
TASK=$1; G=$2; CC=$3; EPS=$4; CAL=$5
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g
tar -xzf "/staging/s/smittal39/wm-results/$G.tar.gz" -C g run/final_model.pt
[ -d com ] || tar -xzf "$CC"
python -m src.commit --study frontier --task "$TASK" --run grounded --model g/run/final_model.pt \
    --out com --episodes "$EPS" --cal_episodes "$CAL" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
tar -czf commit.tar.gz com
echo "done $(date)  exit status $status"
exit $status
