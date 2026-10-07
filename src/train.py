"""Train stock or grounded TD-MPC2 on a ManiSkill3 task, with checkpoint/resume for HTCondor.

Mirrors TD-MPC2's OnlineTrainer (third_party/tdmpc2 @ e9f5932). Data is always collected with the stock planner at the
training horizon (execution_final.md §7.1). Exits with code 85 after checkpointing when the time budget is used up,
so HTCondor (checkpoint_exit_code = 85) restarts the job and training resumes from the checkpoint.

Example:
	python -m src.train --run grounded --task PushCube-v1 --seed 10 --steps 200000 --out_dir runs/push_grounded_s10
"""
import os

os.environ.setdefault("LAZY_LEGACY_OP", "0")
os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

import src.tdmpc2_path  # noqa: F401
from src.tdmpc2_path import TDMPC2_DIR
from common import MODEL_SIZE
from common.buffer import Buffer
from common.parser import cfg_to_dataclass
from common.seed import set_seed
from src.agents.grounded_tdmpc2 import GroundedTDMPC2
from src.envs.maniskill3 import TASKS, ManiSkill3Env
from src.utils import mirror
from src.utils.csvlog import CSVLog
from src.utils.safety import episodes_that_fit, model_is_finite

CHECKPOINT_EXIT_CODE = 85
RUNS = {
	"stock": dict(grounded=False, num_aux_dynamics=0),
	"grounded": dict(grounded=True, num_aux_dynamics=4),
	# Ablations separating grounding from the auxiliary dynamics heads (which share the gradient-clipping norm):
	"grounded_noaux": dict(grounded=True, num_aux_dynamics=0),
	"stock_aux": dict(grounded=False, num_aux_dynamics=4),
}


def parse_args(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("--run", choices=list(RUNS), required=True)
	p.add_argument("--task", choices=TASKS, required=True)
	p.add_argument("--seed", type=int, required=True)
	p.add_argument("--steps", type=int, required=True)
	p.add_argument("--out_dir", type=str, required=True)
	p.add_argument("--model_size", type=int, default=5)
	p.add_argument("--eval_freq", type=int, default=25_000)
	p.add_argument("--eval_episodes", type=int, default=10)
	p.add_argument("--final_eval_episodes", type=int, default=None, help="episodes for the final evaluation (default: eval_episodes)")
	p.add_argument("--replay", choices=["uniform", "lba", "D"], default="uniform",
	               help="detector-guided replay (src/training/audit_replay.py): train more on windows flagged by the audit (lba) or by ensemble disagreement (D)")
	p.add_argument("--replay_refresh", type=int, default=50_000, help="re-score the replay buffer every N steps")
	p.add_argument("--replay_frac", type=float, default=0.5, help="fraction of each batch from the flagged windows")
	p.add_argument("--replay_top", type=float, default=0.3, help="fraction of windows kept as flagged")
	p.add_argument("--eval_seed_start", type=int, default=1000)
	p.add_argument("--video_freq", type=int, default=100_000, help="save an eval video at least this often (and at the end)")
	p.add_argument("--ckpt_freq", type=int, default=25_000)
	p.add_argument("--max_hours", type=float, default=10.0, help="checkpoint and exit 85 after this much wall time")
	p.add_argument("--stop_after_steps", type=int, default=None, help="debug: checkpoint and exit 85 at this step")
	p.add_argument("--no_compile", action="store_true")
	p.add_argument("--compile_mode", default="reduce-overhead", help='torch.compile mode ("reduce-overhead" = TD-MPC2 default, uses CUDA graphs; "default" = no CUDA graphs)')
	p.add_argument("--progress_every", type=int, default=5_000, help="print speed and time breakdown every N steps")
	p.add_argument("--reconfig_every", type=int, default=10,
	               help="training env rebuilds the scene (new object/shape) every N episodes; evaluation rebuilds every episode")
	p.add_argument("--mirror_dir", default="auto", help='"auto" = /staging/<u>/<user>/runs/<run_task_seed_steps> if writable; "none" to disable')
	return p.parse_args(argv)


def gpu_supports_compile() -> bool:
	"""torch.compile (Triton) needs CUDA capability >= 7.0; older GPUs train uncompiled (same weights/checkpoints)."""
	if not torch.cuda.is_available():
		return False
	ok = torch.cuda.get_device_capability(0) >= (7, 0)
	if not ok:
		print(f"GPU {torch.cuda.get_device_name(0)} is too old for torch.compile; training without compilation.")
	return ok


def mirror_dir(args):
	"""Persistent copy of the run on CHTC staging (visible inside the container), so evictions lose little progress."""
	if args.mirror_dir == "none":
		return None
	if args.mirror_dir != "auto":
		return Path(args.mirror_dir)
	base = mirror.staging_base()
	# Every setting that changes training must be in the name, or runs that differ only in it share (and resume
	# from) one mirror: a 2026-10 bug mixed the uniform / lba / D replay arms this way.
	variant = "" if getattr(args, "replay", "uniform") == "uniform" else f"_replay-{args.replay}-f{args.replay_frac}-t{args.replay_top}"
	return None if base is None else base / "runs" / f"{args.run}_{args.task}_s{args.seed}_{args.steps}{variant}"


def build_cfg(args, env: ManiSkill3Env):
	cfg = OmegaConf.load(TDMPC2_DIR / "config.yaml")
	cfg.pop("defaults", None)
	cfg.task = args.task
	cfg.obs = "state"
	cfg.seed = args.seed
	cfg.steps = args.steps
	cfg.model_size = args.model_size
	for k, v in MODEL_SIZE[args.model_size].items():
		cfg[k] = v
	cfg.compile = not args.no_compile and gpu_supports_compile()
	cfg.compile_mode = getattr(args, "compile_mode", "reduce-overhead")
	cfg.enable_wandb = False
	cfg.save_video = False
	cfg.save_agent = False
	cfg.exp_name = args.run
	cfg.work_dir = str(Path(args.out_dir).resolve())
	cfg.task_title = args.task
	cfg.multitask = False
	cfg.task_dim = 0
	cfg.tasks = [args.task]
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.obs_shape = {"state": list(env.observation_space.shape)}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5 * env.max_episode_steps)
	for k, v in RUNS[args.run].items():
		cfg[k] = v
	cfg.run = args.run
	return cfg_to_dataclass(cfg)


def to_td(env, obs, action=None, reward=None, terminated=None) -> TensorDict:
	obs = obs.unsqueeze(0).cpu()
	if action is None:
		action = torch.full_like(env.rand_act(), float("nan"))
	if reward is None:
		reward = torch.tensor(float("nan"))
	if terminated is None:
		terminated = torch.tensor(float("nan"))
	return TensorDict(obs=obs, action=action.unsqueeze(0), reward=reward.unsqueeze(0),
	                  terminated=terminated.unsqueeze(0), batch_size=(1,))


def save_video(frames, path: Path, fps: int = 20):
	import imageio

	path.parent.mkdir(parents=True, exist_ok=True)
	imageio.mimsave(str(path), frames, fps=fps)


class Trainer:
	def __init__(self, args):
		self.args = args
		self.out = Path(args.out_dir)
		self.out.mkdir(parents=True, exist_ok=True)
		self.ckpt_path = self.out / "checkpoint.pt"
		self.job_start = time.time()

		self.mirror = mirror_dir(args)
		if self.mirror is not None:
			self.mirror.mkdir(parents=True, exist_ok=True)
			print(f"Mirroring checkpoints to {self.mirror}")
			self._pull_mirror()
		resume = torch.load(self.ckpt_path, map_location="cpu", weights_only=False) if self.ckpt_path.exists() else None
		self.resume_count = resume["resume_count"] + 1 if resume else 0
		if resume and resume.get("finished"):
			print("Run already finished; nothing to do.")
			sys.exit(0)

		self.env = ManiSkill3Env(args.task, seed=args.seed + 7919 * self.resume_count, reconfiguration_freq=args.reconfig_every)
		self.eval_env = ManiSkill3Env(args.task, seed=args.eval_seed_start, render=True)
		self.cfg = build_cfg(args, self.env)
		set_seed(args.seed + 7919 * self.resume_count)
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision("high")

		self.agent = GroundedTDMPC2(self.cfg)
		self.buffer = Buffer(self.cfg)
		self.train_log = CSVLog(self.out / "train.csv")
		self.eval_log = CSVLog(self.out / "eval.csv")

		self.step, self.ep_idx, self.elapsed_before = 0, 0, 0.0
		self.next_eval, self.next_ckpt, self.next_video = 0, args.ckpt_freq, 0
		self.episodes = []
		self._tds = None
		self.sampler = None
		if args.replay != "uniform":
			from src.training.audit_replay import AuditReplay
			self.sampler = AuditReplay(self.buffer, args.replay, frac=args.replay_frac, top=args.replay_top, seed=args.seed)
		self.next_refresh = 0   # also refreshed right after a resume (scores are not checkpointed)
		if resume:
			self._restore(resume)

	# ---------------------------------------------------------------- checkpointing
	def _restore(self, ck):
		self.agent.load_checkpoint_state(ck["agent"])
		self.step, self.ep_idx, self.elapsed_before = ck["step"], ck["ep_idx"], ck["elapsed"]
		self.next_eval, self.next_ckpt, self.next_video = ck["next_eval"], ck["next_ckpt"], ck["next_video"]
		if ck["episodes"] is not None:
			eps = TensorDict(ck["episodes"], batch_size=ck["episodes"]["reward"].shape[:2])
			self.episodes = list(eps.unbind(0))
			# Reload only the newest episodes that fit: writing more entries than the buffer's capacity in a single
			# extend scrambles episodes (duplicate indices), which puts the NaN placeholder of an episode's first
			# entry into training batches and turns the weights to NaN.
			keep = episodes_that_fit(eps.shape[0], eps.shape[1], self.buffer.capacity)
			self.buffer.load(eps[eps.shape[0] - keep:].clone())
			print(f"Loaded the newest {keep:,} of {eps.shape[0]:,} episodes into the replay buffer")
		print(f"Resumed from step {self.step:,} ({len(self.episodes)} episodes), resume #{self.resume_count}")

	@staticmethod
	def _ckpt_step(path: Path) -> int:
		try:
			return int(torch.load(path, map_location="cpu", weights_only=False)["step"])
		except Exception:
			return -1

	def _pull_mirror(self):
		"""If the mirror has a newer checkpoint than the local sandbox (e.g. after an eviction), copy the run back."""
		if mirror.pull(self.out, self.mirror, self._ckpt_step):
			print(f"Restored run from mirror (step {self._ckpt_step(self.ckpt_path):,})")

	def _push_mirror(self):
		if self.mirror is None:
			return
		try:
			mirror.push(self.out, self.mirror)
		except OSError as e:
			print(f"Warning: mirroring to {self.mirror} failed: {e}")

	def save_checkpoint(self, finished=False):
		if not model_is_finite(self.agent.model):
			print("ERROR: model weights contain NaN/Inf; not overwriting the last good checkpoint.", flush=True)
			sys.exit(1)
		ck = {
			"agent": self.agent.checkpoint_state(),
			"step": self.step, "ep_idx": self.ep_idx, "elapsed": self.elapsed(),
			"next_eval": self.next_eval, "next_ckpt": self.next_ckpt, "next_video": self.next_video,
			# plain tensors (no tensordict dependency), shape [episodes, episode_length + 1, ...]
			"episodes": torch.stack(self.episodes).to_dict() if self.episodes else None,
			"resume_count": self.resume_count, "finished": finished, "args": vars(self.args),
		}
		tmp = self.ckpt_path.with_suffix(".tmp")
		torch.save(ck, tmp)
		os.replace(tmp, self.ckpt_path)
		self._push_mirror()
		print(f"Checkpoint saved at step {self.step:,}")

	def elapsed(self):
		return self.elapsed_before + time.time() - self.job_start

	# ---------------------------------------------------------------- evaluation
	def evaluate(self, record_video: bool, episodes=None, keep=None):
		"""``keep``: list that receives each episode's (obs, action, reward) for later analysis."""
		rewards, successes = [], []
		for i in range(episodes or self.args.eval_episodes):
			obs, done, ep_reward, t = self.eval_env.reset(seed=self.args.eval_seed_start + i), False, 0.0, 0
			traj = dict(obs=[obs], action=[torch.full_like(self.eval_env.rand_act(), float("nan"))], reward=[torch.tensor(float("nan"))])
			frames = [self.eval_env.render()] if (record_video and i == 0) else None
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t == 0, eval_mode=True)
				obs, reward, done, info = self.eval_env.step(action)
				traj["obs"].append(obs)
				traj["action"].append(action.cpu())
				traj["reward"].append(reward)
				ep_reward += float(reward)
				t += 1
				if frames is not None:
					frames.append(self.eval_env.render())
			rewards.append(ep_reward)
			successes.append(info["success"])
			if keep is not None:
				keep.append({k: torch.stack(v) for k, v in traj.items()})
			if frames is not None:
				save_video(frames, self.out / "videos" / f"eval_step{self.step:08d}.mp4")
		return dict(episode_reward=float(np.mean(rewards)), episode_success=float(np.mean(successes)),
		            successes=[float(x) for x in successes])

	# ---------------------------------------------------------------- main loop
	def train(self):
		args, env = self.args, self.env
		done, info, train_metrics = True, None, {}
		sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
		timers = dict(reset=0.0, act=0.0, env=0.0, update=0.0)   # seconds, per episode (logged to train.csv)
		window = dict(reset=0.0, act=0.0, env=0.0, update=0.0, start=time.time(), step=self.step)
		while self.step <= args.steps:
			if done:
				if self.step > 0 and self._tds is not None:
					ep = torch.cat(self._tds)
					self.episodes.append(ep.clone())
					self.ep_idx = self.buffer.add(ep)
					row = dict(step=self.step, episode=self.ep_idx, elapsed=round(self.elapsed(), 1),
					           episode_reward=float(ep["reward"][1:].sum()), episode_success=info["success"])
					row.update({k: float(v) for k, v in train_metrics.items()})
					row.update({f"time_{k}": round(v, 3) for k, v in timers.items()})
					self.train_log.write(row)
					for k in timers:
						window[k] += timers[k]
						timers[k] = 0.0
					if self.step - window["step"] >= args.progress_every:
						dt = time.time() - window["start"]
						n = self.step - window["step"]
						parts = "  ".join(f"{k} {100 * window[k] / dt:.0f}%" for k in ("reset", "act", "env", "update"))
						print(f"[progress] step {self.step:,}  {n / dt:.1f} steps/s  time: {parts}", flush=True)
						window = dict(reset=0.0, act=0.0, env=0.0, update=0.0, start=time.time(), step=self.step)

				if self.step >= args.steps:
					break  # the final evaluation and checkpoint happen after the loop

				if self.step >= self.next_eval:
					record = self.step >= self.next_video
					m = self.evaluate(record_video=record)
					m.pop("successes")
					self.eval_log.write(dict(step=self.step, elapsed=round(self.elapsed(), 1), **m))
					print(f"[eval] step {self.step:,}  success {m['episode_success']:.2f}  reward {m['episode_reward']:.2f}")
					self.next_eval += args.eval_freq
					if record:
						self.next_video += args.video_freq

				if self.step >= self.next_ckpt:
					self.save_checkpoint()
					self.next_ckpt += args.ckpt_freq

				out_of_time = time.time() - self.job_start > args.max_hours * 3600
				debug_stop = args.stop_after_steps is not None and self.step >= args.stop_after_steps and self.resume_count == 0
				if out_of_time or debug_stop:
					self.save_checkpoint()
					print(f"Stopping at step {self.step:,} to be resumed (exit {CHECKPOINT_EXIT_CODE}).")
					sys.exit(CHECKPOINT_EXIT_CODE)

				if (self.sampler is not None and self.step >= self.cfg.seed_steps and self.step >= self.next_refresh
				        and len(self.episodes) >= self.sampler.fit_episodes):
					t0 = time.time()
					self.sampler.refresh(self.agent, self.episodes)
					print(f"[replay] step {self.step:,}  refreshed in {time.time() - t0:.0f} s  {self.sampler.stats}", flush=True)
					self.next_refresh = self.step + args.replay_refresh

				t0 = time.time()
				obs = env.reset()
				timers["reset"] += time.time() - t0
				self._tds = [to_td(env, obs)]

			t0 = time.time()
			if self.step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds) == 1)
			else:
				action = env.rand_act()
			t1 = time.time()
			obs, reward, done, info = env.step(action)
			t2 = time.time()
			timers["act"] += t1 - t0
			timers["env"] += t2 - t1
			self._tds.append(to_td(env, obs, action, reward, info["terminated"]))

			if self.step >= self.cfg.seed_steps:
				num_updates = self.cfg.seed_steps if self.step == self.cfg.seed_steps else 1
				if num_updates > 1:
					print("Pretraining agent on seed data...")
				t0 = time.time()
				for _ in range(num_updates):
					train_metrics = self.agent.update(self.sampler if self.sampler is not None else self.buffer)
				sync()
				timers["update"] += time.time() - t0
			self.step += 1

		kept = []
		m = self.evaluate(record_video=True, episodes=args.final_eval_episodes, keep=kept)
		final = dict(step=self.step, seeds=list(range(args.eval_seed_start, args.eval_seed_start + len(m["successes"]))), **m)
		try:   # how wrong the final model's imagination is on its own evaluation episodes
			from src.training.audit_replay import imagination_error
			final.update(imagination_error(self.agent, kept))
		except Exception as exc:   # never lose a finished run over this diagnostic
			final["imagination_error_failed"] = repr(exc)
		(self.out / "final_eval.json").write_text(json.dumps(final, indent=1))
		m.pop("successes")
		self.eval_log.write(dict(step=self.step, elapsed=round(self.elapsed(), 1), **m))
		print(f"[final eval] step {self.step:,}  success {m['episode_success']:.2f}  return error {final.get('return_error_last')}")
		self.save_checkpoint(finished=True)
		torch.save(self.agent.checkpoint_state()["model"], self.out / "final_model.pt")
		self._push_mirror()


def main(argv=None):
	args = parse_args(argv)
	Trainer(args).train()


if __name__ == "__main__":
	main()
