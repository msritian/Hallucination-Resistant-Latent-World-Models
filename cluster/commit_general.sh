#!/bin/bash
# Generalisation study on one robot (src/commit.py --study general, docs/preregistration_general.md).
# Args: <task> <result_name> <run: stock|grounded> <episodes> <seed_start>
# Collects 50 real episodes (seeds 2000-2049) to fit the LBA (gives L), calibrates on seeds 6900-6909, runs the arms.
TASK=$1; G=$2; RUN=$3; EPS=$4; SEED=$5
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g gen
tar -xzf "/staging/s/smittal39/wm-results/$G.tar.gz" -C g run/final_model.pt
python -m src.commit --study general --task "$TASK" --run "$RUN" --model g/run/final_model.pt \
    --out gen --episodes "$EPS" --seed_start "$SEED" --cal_episodes 10 --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
tar -czf commit.tar.gz gen
echo "done $(date)  exit status $status"
exit $status
