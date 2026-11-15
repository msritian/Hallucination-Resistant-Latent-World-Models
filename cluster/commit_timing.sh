#!/bin/bash
# Wall-clock study on one robot (src/commit.py --study timing, docs/preregistration_wallclock.md).
# Args: <task> <grounded_result_name> <frontier_archive> <episodes> <seed_start>
# Reuses the robot's LBA fit (L) and IR thresholds from its frontier result; arms go to a fresh directory tim/.
TASK=$1; G=$2; FR=$3; EPS=$4; SEED=$5
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g
tar -xzf "/staging/s/smittal39/wm-results/$G.tar.gz" -C g run/final_model.pt
if [ ! -d tim ]; then
    mkdir -p tim fr && tar -xzf "$FR" -C fr && cp fr/com/lba.pkl fr/com/thresholds.json tim/ && rm -rf fr
fi
python -m src.commit --study timing --task "$TASK" --run grounded --model g/run/final_model.pt \
    --out tim --episodes "$EPS" --seed_start "$SEED" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
tar -czf commit.tar.gz tim
echo "done $(date)  exit status $status"
exit $status
