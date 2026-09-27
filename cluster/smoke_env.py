"""GPU smoke test: CUDA, TD-MPC2 imports, and ManiSkill3 tasks. Prints facts the plan depends on."""
import sys
import time
import traceback

TASKS = ["PushCube-v1", "PickSingleYCB-v1", "PegInsertionSide-v1", "StackCube-v1"]


def section(title):
	print(f"\n===== {title} =====", flush=True)


def tdmpc2_discount(episode_length, denom=5, lo=0.95, hi=0.995):
	frac = episode_length / denom
	return min(max((frac - 1) / frac, lo), hi)


section("CUDA")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
	print("device", torch.cuda.get_device_name(0))
	x = torch.randn(1024, 1024, device="cuda")
	print("matmul ok", float((x @ x).sum()) is not None)

section("TD-MPC2 imports")
sys.path.insert(0, "third_party/tdmpc2/tdmpc2")
sys.path.insert(0, ".")
try:
	from common import math, layers  # noqa: F401
	from common.world_model import WorldModel  # noqa: F401
	from common.buffer import Buffer  # noqa: F401
	print("tdmpc2 imports ok")
except Exception:
	traceback.print_exc()

section("ManiSkill3")
import gymnasium as gym
import mani_skill
import mani_skill.envs  # noqa: F401  (registers tasks)
from mani_skill.utils.registration import REGISTERED_ENVS
from src.envs.lavapipe import register_lavapipe
RENDER = register_lavapipe()
print("mani_skill", getattr(mani_skill, "__version__", "?"), "gymnasium", gym.__version__)

for task in TASKS:
	try:
		env = gym.make(
			task, num_envs=1, obs_mode="state", control_mode="pd_ee_delta_pose",
			reward_mode="normalized_dense", sim_backend="cpu", render_backend=RENDER,
		)
		obs, info = env.reset(seed=0)
		T = REGISTERED_ENVS[task].max_episode_steps  # ManiSkill keeps this in its own registry, not env.spec
		t0, steps, total_r, success = time.time(), 0, 0.0, False
		for _ in range(T):
			obs, r, term, trunc, info = env.step(env.action_space.sample())
			steps += 1
			total_r += float(r)
			success = success or bool(info.get("success", False))
			if bool(term) or bool(trunc):
				break
		dt = time.time() - t0
		print(f"{task}: obs {tuple(obs.shape)} action {env.action_space.shape} "
		      f"max_episode_steps {T} ran {steps} steps in {dt:.2f}s ({steps/dt:.0f} steps/s) "
		      f"sum_reward {total_r:.3f} success {success} -> tdmpc2 discount {tdmpc2_discount(T):.3f}")
		print(f"  info keys: {sorted(info.keys())}")
		env.close()
	except Exception as e:
		print(f"{task}: FAILED -> {type(e).__name__}: {e}")

section("Rendering (for videos)")
for backend in [RENDER]:
	try:
		env = gym.make("PushCube-v1", num_envs=1, obs_mode="state", render_mode="rgb_array",
		               sim_backend="cpu", render_backend=backend)
		env.reset(seed=0)
		frame = env.render()
		print(f"render_backend={backend}: ok, frame {tuple(frame.shape)}")
		env.close()
	except Exception as e:
		print(f"render_backend={backend}: FAILED -> {type(e).__name__}: {e}")
