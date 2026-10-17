#!/bin/bash
# Smoke test: DeepMind Control in our container + a released TD-MPC2 checkpoint, using TD-MPC2's own evaluate.py.
echo "host $(hostname)  start $(date)"
tar -xzf code.tar.gz
W=/staging/s/smittal39/tdmpc2_pre
# only the DeepMind Control-specific wheels; keep the container's numpy/scipy/protobuf
pip install --quiet --no-index --no-deps --target deps $W/wheels/{dm_control,dm_env,dm_tree,labmaze,mujoco,glfw,pyopengl,absl_py,etils,lxml}-*.whl
export PYTHONPATH=$PWD/deps:$PWD:$PWD/third_party/tdmpc2/tdmpc2
export MUJOCO_GL=egl
python -c "import dm_control, mujoco, numpy; print('dm_control', dm_control.__version__, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
cd third_party/tdmpc2/tdmpc2
python evaluate.py task=mt30 model_size=48 checkpoint=$W/ckpt/mt30-48M.pt eval_episodes=1 enable_wandb=false save_video=false compile=false 2>&1 | tail -40
python evaluate.py task=walker-walk model_size=5 checkpoint=$W/ckpt/walker-walk-1.pt eval_episodes=2 enable_wandb=false save_video=false compile=false 2>&1 | tail -8
echo "done $(date)"
