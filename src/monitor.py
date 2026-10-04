"""Run-time warning light for the world model (pre-registered in docs/preregistration_monitor.md).

The robot plans with a long horizon (MPPI, H=12, standard scoring). At every step a monitor scores the plan it is about
to act on. If the score is above a threshold (a warning), the robot takes the action of a short-horizon planner (H=3)
instead. Monitors: the learned Bellman audit (LBA), the raw signals A, B, E (and D, M on robots with extra world
models), a random warning at the same rate (control), and an oracle that knows the plan's real outcome (ceiling).

One job = one robot:
  0. Fit the LBA on the robot's saved real episodes (same features, labels and trees as the detection results).
  1. Calibration: closed-loop H=12 episodes on separate seeds, every chosen plan also executed in the simulator.
     Threshold per monitor and warning rate = the quantile of its per-step score that gives that rate.
  2. Arms on the same evaluation seeds (paired). The plain H=12 arm also executes every chosen plan in the simulator,
     which measures how accurate each warning light is on the plans the robot actually acts on.

Resumable: the LBA, thresholds and every finished arm are saved in --out; with --max_hours the job exits with code 85
after an arm once the time is up (HTCondor saves --out and restarts it).

Example:
	python -m src.monitor --model g/run/final_model.pt --run grounded --task PushCube-v1 \
		--episodes_file old/preflight/episodes_grounded.pt --out mon
"""
import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.preflight.run_preflight import CRITIC_FEATURES, _features, fit_lba, plan_signals, return_error

RAW = ("A", "B", "E", "D", "M")           # raw signals used as monitors (D, M only with extra world models)
RATES = (0.10, 0.25, 0.50)               # target warning rates
MID = 0.25                               # the rate at which all monitors are compared
H_LONG, H_SHORT = 12, 3


@dataclass(frozen=True)
class Arm:
	name: str
	base: str                            # planner that acts: H3, H12, L08 (H=12 constant-lambda 0.8)
	monitor: Optional[str] = None        # LBA, A, B, E, D, M, random, oracle (switches H12 -> H3 on a warning)
	rate: float = 0.0
	replay: bool = False                 # execute every chosen H=12 plan in the simulator (labels / oracle)


def arms_for(has_ensemble: bool) -> list:
	arms = [Arm("H3", "H3"), Arm("H12", "H12", replay=True), Arm("L08_H12", "L08")]
	arms += [Arm(f"H12+LBA@{r}", "H12", "LBA", r) for r in RATES]
	arms += [Arm(f"H12+random@{r}", "H12", "random", r) for r in RATES]
	arms += [Arm(f"H12+{k}@{MID}", "H12", k, MID) for k in RAW if has_ensemble or k not in ("D", "M")]
	arms += [Arm(f"H12+oracle@{MID}", "H12", "oracle", MID, replay=True)]
	return arms


def make_planners(agent) -> dict:
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	return {"H12": AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard")),
	        "H3": AuditedPlanner(agent, PlannerConfig(horizon=H_SHORT, score="standard")),
	        "L08": AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="elvis", lambda_signal="const", lambda_max=0.8))}


# ------------------------------------------------------------------------------------------------ scoring one plan
@torch.no_grad()
def score_plan(agent, obs: torch.Tensor, plan: torch.Tensor, lba: dict, beta: float = 1.0):
	"""Per-step scores of every monitor for one plan [H, A] from the real observation obs [D].

	Returns ({monitor: per-step scores [L]}, rollout). The per-decision score of a monitor is the max over steps."""
	dev = next(agent.model.parameters()).device
	H = plan.shape[0]
	assert H == len(lba["bias"]), f"plan horizon {H} must match the audit's horizon {len(lba['bias'])}"
	z0 = agent.model.encode(obs.to(dev).view(1, -1), None)
	sig, roll = plan_signals(agent, z0, plan.to(dev).view(H, 1, -1), beta, bias=lba["bias"])
	sig = {k: v.cpu() for k, v in sig.items()}
	L = lba["L"]
	X = _features(sig, roll, CRITIC_FEATURES, L, True).numpy()
	steps = {"LBA": torch.as_tensor(lba["clf"].predict_proba(X)[:, 1], dtype=torch.float32)}
	for k in RAW:
		if k in sig:
			steps[k] = sig[k][:L, 0].float()
	return steps, roll


def plan_error(roll, r_true: torch.Tensor, discount: float) -> torch.Tensor:
	"""|imagined - real discounted return| per step [L] for one plan (the simulator answer key, as in detection)."""
	return return_error(roll, r_true.view(-1, 1), discount)[:, 0]


# ------------------------------------------------------------------------------------------------ episodes
@torch.no_grad()
def run_arm(agent, env, sim, arm: Arm, lba: dict, thresholds: dict, episodes: int, seed_start: int,
            beta: float = 1.0, seed: int = 0):
	"""Run one arm; returns (per-episode rows, per-decision record tensors)."""
	planners = make_planners(agent)
	gen = torch.Generator().manual_seed(seed)
	discount = float(agent.discount)
	rows, rec = [], dict(ep=[], t=[], alarm=[], score={}, steps={}, E=[], det_err=[])
	needs_long = arm.base == "H12"
	needs_short = arm.base == "H3" or arm.monitor is not None
	for i in range(episodes):
		obs, done, t, ep_reward, n_alarm, n_dec = env.reset(seed=seed_start + i), False, 0, 0.0, 0, 0
		start = time.perf_counter()
		while not done:
			t0 = t == 0
			if arm.base == "L08":
				action = planners["L08"].act(obs, t0=t0)
			else:
				a_short = planners["H3"].act(obs, t0=t0) if needs_short else None
				action = a_short
				if needs_long:
					a_long = planners["H12"].act(obs, t0=t0)
					plan = planners["H12"].last_plan.detach().cpu()
					steps, roll = score_plan(agent, obs, plan, lba, beta)
					E = torch.full((lba["L"],), float("nan"))
					snap = None
					if arm.replay:
						snap = sim.save()
						r_true, _ = sim.rollout(snap, plan.view(plan.shape[0], 1, -1))
						E = plan_error(roll, r_true[:, 0], discount)
					alarm = False
					if arm.monitor == "random":
						alarm = bool(torch.rand(1, generator=gen) < arm.rate)
					elif arm.monitor is not None:
						score = float(E.max()) if arm.monitor == "oracle" else float(steps[arm.monitor].max())
						alarm = warns(score, thresholds[arm.monitor][str(arm.rate)], gen)
					action = a_short if alarm else a_long
					n_alarm += int(alarm)
					n_dec += 1
					rec["ep"].append(i)
					rec["t"].append(t)
					rec["alarm"].append(alarm)
					rec["E"].append(E)
					for k, v in steps.items():
						rec["steps"].setdefault(k, []).append(v)
			check = None
			if needs_long and arm.replay:  # the snapshot must reproduce the real step (validity of the labels);
				_, check = sim.rollout(snap, action.view(1, 1, -1))  # run BEFORE the real step: rollout restores snap
			obs_next, reward, done, info = env.step(action)
			if check is not None:
				rec["det_err"].append(float((check[0, 0] - obs_next).abs().max()))
			obs, ep_reward, t = obs_next, ep_reward + float(reward), t + 1
		rows.append(dict(arm=arm.name, episode=i, seed=seed_start + i, success=float(info["success"]), reward=ep_reward,
		                 warning_rate=n_alarm / max(n_dec, 1), steps=t, seconds=time.perf_counter() - start))
	out = dict(ep=torch.tensor(rec["ep"]), t=torch.tensor(rec["t"]), alarm=torch.tensor(rec["alarm"], dtype=torch.bool),
	           E=torch.stack(rec["E"]) if rec["E"] else torch.empty(0),
	           steps={k: torch.stack(v) for k, v in rec["steps"].items()},
	           det_err=torch.tensor(rec["det_err"]))
	return rows, out


def warns(score: float, th: dict, gen: torch.Generator) -> bool:
	"""Warning if the score is above the threshold; at exactly the threshold, with probability th['tie']."""
	if score > th["thr"]:
		return True
	return score == th["thr"] and bool(torch.rand(1, generator=gen) < th["tie"])


def calibrate_thresholds(rec: dict) -> dict:
	"""{monitor: {rate: {thr, tie}}} so that each monitor warns on `rate` of the calibration decisions.

	``tie`` handles scores that repeat exactly (e.g. a saturated probability of 1.0): decisions scoring exactly ``thr``
	warn with that probability, so the expected warning rate is ``rate`` even when many scores are equal."""
	scores = {k: v.max(1).values for k, v in rec["steps"].items()}
	scores["oracle"] = rec["E"].max(1).values
	out = {}
	for k, s in scores.items():
		s = s.float()
		out[k] = {}
		for r in RATES:
			thr = float(torch.quantile(s, 1 - r))
			above, equal = float((s > thr).float().mean()), float((s == thr).float().mean())
			tie = min(max((r - above) / equal, 0.0), 1.0) if equal > 0 else 0.0
			out[k][str(r)] = dict(thr=thr, tie=tie)
	return out


def write_outputs(out: Path, arms: list):
	"""episodes.csv (one row per arm and episode, the format pool_lambda reads) and summary.json, from the arm files."""
	import csv
	summary, all_rows = {}, []
	for arm in arms:
		rows = torch.load(out / "arms" / f"{arm.name}.pt", weights_only=False)["rows"]
		all_rows += rows
		summary[arm.name] = dict(success=sum(r["success"] for r in rows) / len(rows),
		                         warning_rate=sum(r.get("warning_rate", 0.0) for r in rows) / len(rows), episodes=len(rows))
	with open(out / "episodes.csv", "w", newline="") as fh:
		w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
		w.writeheader()
		w.writerows(all_rows)
	(out / "summary.json").write_text(json.dumps(summary, indent=1))


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
	from src.envs.maniskill3 import ManiSkill3Env
	from src.preflight.run_preflight import load_agent
	from src.tools.plan_diag import SimSnapshot

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", choices=["stock", "grounded", "grounded_noaux", "stock_aux"], required=True)
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", required=True, help="episodes_<model>.pt saved by run_preflight (fits the LBA)")
	p.add_argument("--out", required=True)
	p.add_argument("--episodes", type=int, default=100)
	p.add_argument("--seed_start", type=int, default=6000)
	p.add_argument("--cal_episodes", type=int, default=10)
	p.add_argument("--cal_seed_start", type=int, default=6900)
	p.add_argument("--arms", default=None, help="comma-separated subset of arm names (default: all)")
	p.add_argument("--max_hours", type=float, default=None, help="exit 85 after an arm once this much time has passed")
	args = p.parse_args(argv)
	out = Path(args.out)
	(out / "arms").mkdir(parents=True, exist_ok=True)
	job_start = time.time()

	agent = load_agent(args.model, args.run, args.task)
	has_ensemble = agent.num_aux_dynamics > 0
	# 0. the learned Bellman audit, fitted on the robot's saved real episodes
	if (out / "lba.pkl").exists():
		lba = pickle.loads((out / "lba.pkl").read_bytes())
	else:
		data = torch.load(args.episodes_file)
		lba = fit_lba(agent, data, H_LONG, beta=1.0)
		(out / "lba.pkl").write_bytes(pickle.dumps(lba))
		(out / "lba.json").write_text(json.dumps({k: lba[k] for k in ("eps_clean", "eps_hall", "L", "n_pos")}, indent=1))
		print(f"LBA fitted on {data['obs'].shape[0]} episodes ({lba['n_pos']} positive steps)", flush=True)

	env = ManiSkill3Env(args.task, seed=args.seed_start)
	sim = SimSnapshot(env)
	# 1. thresholds from closed-loop H=12 calibration episodes (separate seeds)
	if (out / "thresholds.json").exists():
		thresholds = json.loads((out / "thresholds.json").read_text())
	else:
		_, cal = run_arm(agent, env, sim, Arm("calibration", "H12", replay=True), lba, {}, args.cal_episodes,
		                 args.cal_seed_start)
		thresholds = calibrate_thresholds(cal)
		torch.save(cal, out / "calibration.pt")
		(out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
		print(f"thresholds: {json.dumps(thresholds)}", flush=True)

	# 2. arms, same evaluation seeds
	arms = arms_for(has_ensemble)
	if args.arms:
		keep = set(args.arms.split(","))
		arms = [a for a in arms if a.name in keep]
	for k, arm in enumerate(arms):
		f = out / "arms" / f"{arm.name}.pt"
		if f.exists():
			continue
		rows, rec = run_arm(agent, env, sim, arm, lba, thresholds, args.episodes, args.seed_start, seed=k)
		torch.save(dict(rows=rows, rec=rec), f)
		succ = sum(r["success"] for r in rows) / len(rows)
		warn = sum(r["warning_rate"] for r in rows) / len(rows)
		sec = sum(r["seconds"] for r in rows) / len(rows)
		print(f"{arm.name:22s} success {succ:.2f}  warning rate {warn:.2f}  {sec:.1f} s/episode", flush=True)
		if args.max_hours is not None and time.time() - job_start > args.max_hours * 3600 and k < len(arms) - 1:
			print("time limit reached: checkpointing (exit 85)", flush=True)
			env.close()
			sys.exit(85)
	env.close()

	write_outputs(out, arms)
	print("done", flush=True)


if __name__ == "__main__":
	main()
