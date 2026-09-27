#!/bin/bash
# Update-speed benchmark around checkpoint saves. Args passed to src.tools.update_bench
echo "host $(hostname)  start $(date)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
tar -xzf code.tar.gz
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 LP_NUM_THREADS=2
python -m src.tools.update_bench "$@"
echo "done $(date)  exit status $?"
