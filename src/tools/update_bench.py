"""Benchmark: does saving a checkpoint slow down or recompile TD-MPC2's compiled update?

For each save method, a fresh agent is trained on a random replay buffer. Update time and torch.compile counters are
measured before and after each save. Methods:
	none           no save (control)
	state_dict     agent.checkpoint_state() (current behavior: model/optimizer/scale state_dict)
	tensor_copy    copy parameters/buffers via named_parameters()/named_buffers() + optimizer state tensors
	state_dict_reset  state_dict, then torch._dynamo.reset() (forces one clean recompile)
"""
import argparse
import time
from types import SimpleNamespace

import torch
from tensordict.tensordict import TensorDict

import src.tdmpc2_path  # noqa: F401
from common.buffer import Buffer
from src.agents.grounded_tdmpc2 import GroundedTDMPC2
from src.envs.maniskill3 import ManiSkill3Env
from src.train import build_cfg


def dynamo_counts():
	from torch._dynamo.utils import counters
	return dict(frames_ok=sum(counters["frames"].values()) if "frames" in counters else 0,
	            unique_graphs=counters["stats"].get("unique_graphs", 0))


def tensor_copy_state(agent):
	m = agent.model
	params = {k: v.detach().to("cpu", copy=True) for k, v in m.named_parameters()}
	buffers = {k: v.detach().to("cpu", copy=True) for k, v in m.named_buffers()}
	return params, buffers


def fill_buffer(cfg, n_eps: int, T: int):
	buf = Buffer(cfg)
	obs_dim, A = cfg.obs_shape["state"][0], cfg.action_dim
	for _ in range(n_eps):
		td = TensorDict(obs=torch.randn(T + 1, obs_dim), action=torch.rand(T + 1, A) * 2 - 1,
		                reward=torch.rand(T + 1), terminated=torch.zeros(T + 1), batch_size=(T + 1,))
		buf.add(td)
	return buf


def time_updates(agent, buf, n):
	torch.cuda.synchronize()
	t = time.perf_counter()
	for _ in range(n):
		agent.update(buf)
	torch.cuda.synchronize()
	return 1000 * (time.perf_counter() - t) / n


def run_method(args, env, method: str, mode: str):
	torch._dynamo.reset()
	torch.manual_seed(0)
	a = SimpleNamespace(run=args.run, task=args.task, seed=0, steps=1_000_000, model_size=5, no_compile=False,
	                    compile_mode=mode, out_dir="bench_tmp")
	agent = GroundedTDMPC2(build_cfg(a, env))
	T = env.max_episode_steps
	buf = fill_buffer(agent.cfg, n_eps=100, T=T)
	warm = time_updates(agent, buf, args.updates)
	print(f"\n=== {args.run} mode={mode} method={method}: warm-up {warm:.1f} ms/update  {dynamo_counts()}", flush=True)
	for c in range(args.cycles):
		t0 = time.perf_counter()
		if method == "torch_save":
			torch.save({"agent": agent.checkpoint_state()}, "bench_ckpt.pt")
		elif method == "grow_buffer":
			obs_dim, A = agent.cfg.obs_shape["state"][0], agent.cfg.action_dim
			for _ in range(50):
				buf.add(TensorDict(obs=torch.randn(T + 1, obs_dim), action=torch.rand(T + 1, A) * 2 - 1,
				                   reward=torch.rand(T + 1), terminated=torch.zeros(T + 1), batch_size=(T + 1,)))
		elif method == "reset":
			torch._dynamo.reset()
		save_ms = 1000 * (time.perf_counter() - t0)
		first = time_updates(agent, buf, 1)
		steady = time_updates(agent, buf, args.updates)
		print(f"cycle {c}: {method} {save_ms:.0f} ms | first update after {first:.0f} ms | "
		      f"steady {steady:.1f} ms/update | {dynamo_counts()}", flush=True)
	del agent, buf
	torch.cuda.empty_cache()


def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", default="PegInsertionSide-v1")
	p.add_argument("--cycles", type=int, default=5)
	p.add_argument("--updates", type=int, default=300)
	p.add_argument("--methods", default="torch_save,grow_buffer,reset")
	p.add_argument("--modes", default="reduce-overhead,default")
	args = p.parse_args(argv)
	torch._logging.set_logs(recompiles=True)  # prints the reason for every recompilation
	env = ManiSkill3Env(args.task, seed=0)
	for mode in args.modes.split(","):
		for method in args.methods.split(","):
			run_method(args, env, method, mode)


if __name__ == "__main__":
	main()
