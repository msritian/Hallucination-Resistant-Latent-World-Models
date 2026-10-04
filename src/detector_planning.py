"""Closed-loop test of planners that use the learned Bellman audit on every candidate
(pre-registered in docs/preregistration_detector_planning.md).

One job = one robot:
  0. Fit the audit on the robot's saved real episodes, once for 12-step and once for 24-step windows (same recipe as
     the detection results), move the trees to the GPU and check them against sklearn on real features.
  1. Run every arm on the same evaluation seeds (paired); one row per episode.
Resumable (exit 85 after an arm once --max_hours has passed; HTCondor restores --out).

Example:
	python -m src.detector_planning --model g/run/final_model.pt --run grounded --task PushCube-v1 \
		--episodes_file old/preflight/episodes_grounded.pt --out plan
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

from src.monitor import write_outputs
from src.planning.detector_planner import DetectorPlanner, DetectorRule
from src.planning.gpu_audit import GPUTrees
from src.preflight.run_preflight import fit_lba

R = DetectorRule
ARMS = [
	R("L08_H3", 3),                                                # short plan, same lambda-return
	R("L08_H12", 12),                                              # best earlier planner: constant lambda 0.8
	R("LBAlam_H12", 12, lam="lba"),                                # P1: lambda_t = 0.8 (1 - p_t)
	R("LBAlam_H12_shuffled", 12, lam="lba_shuffled"),              # P2 control
	R("L08_H12_drop25", 12, drop=0.25),                            # P3: drop the 25% most suspicious candidates
	R("L08_H12_dropRandom25", 12, drop=0.25, drop_random=True),    # P4 control
	R("LBAlam_pure_H12", 12, lam="lba_pure"),                      # exploratory: lambda_t = 1 - p_t
	R("L08_H24", 24),                                              # exploratory: long horizon
	R("LBAlam_H24", 24, lam="lba"),                                # exploratory: long horizon with the audit
]


def make_audits(agent, data: dict, device, horizons=(12, 24)) -> dict:
	out = {}
	for H in horizons:
		lba = fit_lba(agent, data, H, beta=1.0)
		trees = GPUTrees(lba["clf"], device)
		err = trees.check(lba["clf"], lba["X_sample"])
		print(f"audit H={H}: {lba['n_pos']} positive steps; GPU trees match sklearn (max error {err:.1e})", flush=True)
		out[H] = dict(trees=trees, bias=lba["bias"].to(device), clf=lba["clf"], X_sample=lba["X_sample"])
	return out


def run_episodes(env, planner, episodes: int, seed_start: int, name: str) -> list:
	rows = []
	for i in range(episodes):
		obs, done, t, ep_reward, start = env.reset(seed=seed_start + i), False, 0, 0.0, time.perf_counter()
		while not done:
			obs, reward, done, info = env.step(planner.act(obs, t0=t == 0, eval_mode=True))
			ep_reward, t = ep_reward + float(reward), t + 1
		rows.append(dict(arm=name, episode=i, seed=seed_start + i, success=float(info["success"]), reward=ep_reward,
		                 steps=t, seconds=time.perf_counter() - start))
	return rows


def main(argv=None):
	from src.envs.maniskill3 import ManiSkill3Env
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", choices=["stock", "grounded", "grounded_noaux", "stock_aux"], required=True)
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", required=True)
	p.add_argument("--out", required=True)
	p.add_argument("--episodes", type=int, default=100)
	p.add_argument("--seed_start", type=int, default=6000)
	p.add_argument("--arms", default=None, help="comma-separated subset of arm names")
	p.add_argument("--max_hours", type=float, default=None)
	args = p.parse_args(argv)
	out = Path(args.out)
	(out / "arms").mkdir(parents=True, exist_ok=True)
	job_start = time.time()

	agent = load_agent(args.model, args.run, args.task)
	device = next(agent.model.parameters()).device
	if (out / "audits.pkl").exists():   # sklearn trees from the first start; GPU copies are rebuilt and re-checked
		saved = pickle.loads((out / "audits.pkl").read_bytes())
		audits = {}
		for H, a in saved.items():
			trees = GPUTrees(a["clf"], device)
			trees.check(a["clf"], a["X_sample"])
			audits[H] = dict(trees=trees, bias=a["bias"].to(device))
	else:
		audits = make_audits(agent, torch.load(args.episodes_file), device)
		(out / "audits.pkl").write_bytes(pickle.dumps({H: dict(clf=a["clf"], bias=a["bias"].cpu(), X_sample=a["X_sample"])
		                                               for H, a in audits.items()}))

	arms = [a for a in ARMS if args.arms is None or a.name in set(args.arms.split(","))]
	env = ManiSkill3Env(args.task, seed=args.seed_start)
	for k, arm in enumerate(arms):
		f = out / "arms" / f"{arm.name}.pt"
		if f.exists():
			continue
		planner = DetectorPlanner(agent, arm, audits.get(arm.horizon), seed=k)
		rows = run_episodes(env, planner, args.episodes, args.seed_start, arm.name)
		stats = {f"mean_{s}": sum(v) / max(len(v), 1) for s, v in planner.stats.items()}
		torch.save(dict(rows=rows, stats=stats), f)
		succ = sum(r["success"] for r in rows) / len(rows)
		sec = sum(r["seconds"] for r in rows) / len(rows)
		print(f"{arm.name:24s} success {succ:.2f}  {sec:.1f} s/episode  " +
		      "  ".join(f"{s} {v:.3f}" for s, v in stats.items()), flush=True)
		if args.max_hours is not None and time.time() - job_start > args.max_hours * 3600 and k < len(arms) - 1:
			print("time limit reached: checkpointing (exit 85)", flush=True)
			env.close()
			sys.exit(85)
	env.close()
	write_outputs(out, arms)
	print("done", flush=True)


if __name__ == "__main__":
	main()
