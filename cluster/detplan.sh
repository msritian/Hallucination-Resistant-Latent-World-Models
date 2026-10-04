#!/bin/bash
# Detector-in-the-planner experiment on one trained robot (src/detector_planning.py).
# Args: <task> <grounded_result_name> <episodes>
# Exit 85 = checkpoint (./plan is saved by HTCondor and the job restarts); any other exit returns plan as detplan.tar.gz.
TASK=$1; G=$2; EPS=$3
echo "host $(hostname)  start $(date)"
python -c "import torch, sklearn; print('GPU', torch.cuda.get_device_name(0), 'sklearn', sklearn.__version__)"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
mkdir -p g old
tar -xzf "$G.tar.gz" -C g
tar -xzf "real_$G.tar.gz" -C old
python -m src.detector_planning --task "$TASK" --run grounded --model g/run/final_model.pt \
    --episodes_file old/preflight/episodes_grounded.pt --out plan --episodes "$EPS" --max_hours 8
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
mkdir -p plan
tar -czf detplan.tar.gz plan
echo "done $(date)  exit status $status"
exit $status
