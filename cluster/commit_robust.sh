#!/bin/bash
# Robust imagined regret study on one robot (src/commit.py --study robust, src/robust_regret.py).
# Args: <task> <grounded_result_name> <frontier_archive> <episodes> <cal_episodes>
# <frontier_archive> holds the robot's confirmation + frontier directory (com/: lba.pkl, thresholds, all earlier arms);
# the new arms are added to it (same evaluation seeds 6000+, same calibration seeds 6900+).
TASK=$1; G=$2; FR=$3; EPS=$4; CAL=$5
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g
if [ -f "$G.tar.gz" ]; then M="$G.tar.gz"; else M="/staging/s/smittal39/wm-results/$G.tar.gz"; fi
tar -xzf "$M" -C g run/final_model.pt
[ -d com ] || tar -xzf "$FR"
python -m src.commit --study robust --task "$TASK" --run grounded --model g/run/final_model.pt \
    --out com --episodes "$EPS" --cal_episodes "$CAL" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
tar -czf commit.tar.gz com
echo "done $(date)  exit status $status"
exit $status
