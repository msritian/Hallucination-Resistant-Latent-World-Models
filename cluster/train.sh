#!/bin/bash
# Runs inside the container on a GPU node. All arguments are passed to src.train; outputs go to ./run.
# Exit 85 = checkpoint saved: HTCondor saves ./run and restarts the job. Any other exit: ./run is bundled into
# run.tar.gz and returned (also on crashes, so logs and partial results come back instead of the job being held).
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD
# Use only the CPUs the job reserved (request_cpus); otherwise PyTorch and the CPU renderer may start one thread
# per core of the whole machine and slow each other down on shared nodes.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-2}
echo "threads: OMP=$OMP_NUM_THREADS LP=$LP_NUM_THREADS  cpus on node: $(nproc)"
python -m src.train --out_dir run "$@"
status=$?
if [ "$status" -eq 85 ]; then
    exit 85
fi
mkdir -p run
tar -czf run.tar.gz run
echo "done $(date)  exit status $status"
exit $status
