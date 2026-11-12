#!/bin/bash
# Commitment study on DeepMind Control with the released TD-MPC2 mt30-48M checkpoint (src/commit_dmc.py,
# docs/preregistration_dmc_commit.md). Args: <task> <episodes> <cal_episodes> [<fixed_ks> <seed0>] (headroom pilot)
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
W=/staging/s/smittal39/tdmpc2_pre
pip install --quiet --no-index --no-deps --target deps $W/wheels/{dm_control,dm_env,dm_tree,labmaze,mujoco,glfw,pyopengl,absl_py,etils,lxml}-*.whl
export PYTHONPATH=$PWD/deps:$PWD:$PWD/third_party/tdmpc2/tdmpc2
export MUJOCO_GL=egl
python -m src.commit_dmc task=mt30 model_size=48 checkpoint=$W/ckpt/mt30-48M.pt enable_wandb=false save_video=false \
    compile=false +out=$PWD/dcm +tasks_subset=$1 +episodes=$2 +cal_episodes=$3 ${4:++fixed_ks=$4} ${5:++seed0=$5}
status=$?
mkdir -p dcm && tar -czf dcm.tar.gz dcm
echo "done $(date)  exit status $status"
exit $status
