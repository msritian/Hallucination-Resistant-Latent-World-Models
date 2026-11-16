#!/bin/bash
# Live IR study on one robot (src/commit.py --study live, docs/preregistration_live_ir.md).
# Args: <task> <model_result_name> <run: stock|grounded> <archive> <dir> <episodes> <seed_start>
# <archive> holds an earlier study's directory <dir>/ (lba.pkl, thresholds, earlier arms on the same evaluation seeds);
# the live arms are added to it. Calibration seeds 6900-6909 as in every earlier study.
TASK=$1; G=$2; RUN=$3; AR=$4; D=$5; EPS=$6; SEED=$7
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g
if [ -f "$G.tar.gz" ]; then M="$G.tar.gz"; else M="/staging/s/smittal39/wm-results/$G.tar.gz"; fi
tar -xzf "$M" -C g run/final_model.pt
[ -d "$D" ] || tar -xzf "$AR"
python -m src.commit --study live --task "$TASK" --run "$RUN" --model g/run/final_model.pt \
    --out "$D" --episodes "$EPS" --seed_start "$SEED" --cal_episodes 10 --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
tar -czf commit.tar.gz "$D"
echo "done $(date)  exit status $status"
exit $status
