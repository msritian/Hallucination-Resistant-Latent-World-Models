#!/bin/bash
# Non-interactive container build (CHTC build node): newer CUDA wheels (cu128, supports the newest GPUs) + dm_control.
set -e
# non-interactive jobs have no usable HOME/tmp: keep apptainer cache and temp files in the job sandbox
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=$PWD APPTAINER_CACHEDIR=$PWD/cache APPTAINER_TMPDIR=$PWD/tmp
mkdir -p cache tmp
ls -la; which apptainer
apptainer build tdmpc2_v3.sif tdmpc2_v3.def
apptainer exec tdmpc2_v3.sif python -c "import torch, mani_skill, dm_control; print('ok torch', torch.__version__, torch.version.cuda, 'arch list', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'n/a')"
cp tdmpc2_v3.sif /staging/s/smittal39/tdmpc2_v3.sif
ls -la /staging/s/smittal39/tdmpc2_v3.sif
