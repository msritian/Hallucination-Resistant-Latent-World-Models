#!/bin/bash
# Bellman audit of a released TD-MPC2 checkpoint on DeepMind Control (src/dmc_audit.py). Args: hydra overrides.
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
W=/staging/s/smittal39/tdmpc2_pre
pip install --quiet --no-index --no-deps --target deps $W/wheels/{dm_control,dm_env,dm_tree,labmaze,mujoco,glfw,pyopengl,absl_py,etils,lxml}-*.whl
export PYTHONPATH=$PWD/deps:$PWD:$PWD/third_party/tdmpc2/tdmpc2
export MUJOCO_GL=egl
python -m src.dmc_audit enable_wandb=false save_video=false compile=false +out=$PWD/dmc "$@"
status=$?
mkdir -p dmc && tar -czf dmc.tar.gz dmc
echo "done $(date)  exit status $status"
exit $status
