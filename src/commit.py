"""Commitment study: is detection worth more when the robot executes its imagination open-loop?
(pre-registered in docs/preregistration_commitment.md)

The robot plans with MPPI (H=12, standard scoring) and then executes the first k actions of the chosen plan without
replanning. Replanning every step (k = 1) lets reality correct a hallucination after one action; committing to k > 1
actions makes acting on a hallucinated plan costly. A gate decides k:

  fixed      k = 1, 2, 4, 8 for every plan;
  before acting (from the plan alone): commit to the longest prefix (at most K_MAX actions) whose per-step score stays
             at or below a threshold. Scores: the learned Bellman audit (LBA), the label-free anchored residual (Aa),
             the dynamics ensemble (D), MOBILE-style target spread (M), and an oracle (the plan's true return error,
             from executing the plan in a simulator copy);
  while acting: PE, the prediction-error check, executes until the real next state disagrees with the imagined one
             (latent distance above a threshold), then replans; it sees reality, the other gates do not;
  random     continue each next step with a fixed probability (control).

Thresholds are set from closed-loop k = 1 calibration episodes (separate seeds) so that every gate commits to the
same mean number of actions per plan (BUDGETS), i.e. the same planning compute as fixed k = budget.

One job = one robot. Resumable: the LBA, thresholds and every finished arm are saved in --out; with --max_hours the job
exits with code 85 after an arm once the time is up (HTCondor saves --out and restarts it).

Example:
	python -m src.commit --model g/run/final_model.pt --run grounded --task PushCube-v1 \
		--episodes_file old/preflight/episodes_grounded.pt --out com
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

from src.monitor import H_LONG, plan_error
from src.preflight.run_preflight import CRITIC_FEATURES, _features, fit_lba, plan_signals

K_MAX = 8
BUDGETS = (2, 4)                                          # target mean actions per plan for the gates
SCORES = ("LBA", "Aa", "D", "M")                          # before-acting gates from the plan's per-step scores
GATES = ("oracle", *SCORES, "PE", "random")


@dataclass(frozen=True)
class Arm:
	name: str
	k: int = 1                     # fixed commitment (when gate is None)
	gate: Optional[str] = None
	budget: int = 0                # target mean commitment of the gate
	abort: bool = False            # also stop early on the prediction-error check (combination arm)


def arms_for(has_ensemble: bool) -> list:
	arms = [Arm(f"k{k}", k=k) for k in (1, 2, 4, 8)]
	for b in BUDGETS:
		arms += [Arm(f"{g}@{b}", gate=g, budget=b) for g in GATES if has_ensemble or g not in ("D", "M")]
	arms.append(Arm("LBA+PE@4", gate="LBA", budget=4, abort=True))
	return arms


# ------------------------------------------------------------------------------------------------ scoring one plan
@torch.no_grad()
def score_plan(agent, obs: torch.Tensor, plan: torch.Tensor, lba: dict):
	"""Per-step scores [L] of the before-acting gates for one plan [H, A] from the real observation obs [D].

	Step j judges the imagined transition into z_{j+1}, i.e. whether action j+1 is still chosen from a trustworthy
	imagined state. Returns ({score: [L]}, rollout)."""
	dev = next(agent.model.parameters()).device
	H = plan.shape[0]
	z0 = agent.model.encode(obs.to(dev).view(1, -1), None)
	sig, roll = plan_signals(agent, z0, plan.to(dev).view(H, 1, -1), 1.0, bias=lba["bias"])
	sig = {k: v.cpu() for k, v in sig.items()}
	L = lba["L"]
	X = _features(sig, roll, CRITIC_FEATURES, L, True).numpy()
	steps = {"LBA": torch.as_tensor(lba["clf"].predict_proba(X)[:, 1], dtype=torch.float32)}
	for k in SCORES[1:]:
		if k in sig:
			steps[k] = sig[k][:L, 0].float()
	return steps, roll


@torch.no_grad()
def latent_error(agent, obs: torch.Tensor, z_hat: torch.Tensor) -> float:
	"""Distance between the encoded real observation [D] and the imagined latent [d] (the PE check)."""
	dev = next(agent.model.parameters()).device
	return float((agent.model.encode(obs.to(dev).view(1, -1), None)[0] - z_hat.to(dev)).norm())


def passes(score: float, th: dict, gen: torch.Generator) -> bool:
	"""Trust a step if its score is below the threshold; at exactly the threshold, with probability th['tie']."""
	if score < th["thr"]:
		return True
	return score == th["thr"] and bool(torch.rand(1, generator=gen) < th["tie"])


def commit_len(steps: torch.Tensor, th: dict, gen: torch.Generator) -> int:
	"""1 + the number of leading steps (at most K_MAX - 1) whose running-max score passes the threshold."""
	c = torch.cummax(steps[:K_MAX - 1].float(), 0).values
	k = 1
	for v in c.tolist():
		if not passes(v, th, gen):
			break
		k += 1
	return k


def random_q(budget: int) -> float:
	"""Continue probability q with E[k] = 1 + q + ... + q^(K_MAX-1) = budget (bisection)."""
	lo, hi = 0.0, 1.0
	for _ in range(60):
		q = (lo + hi) / 2
		if 1 + sum(q ** i for i in range(1, K_MAX)) < budget:
			lo = q
		else:
			hi = q
	return (lo + hi) / 2


def shift_warm_start(planner, j: int):
	"""After executing j actions of a plan, align MPPI's warm start (the planner itself shifts by one step)."""
	if j > 1:
		pm = planner.prev_mean
		s = j - 1
		pm[:-s] = pm[s:].clone()
		pm[-s:] = 0


# ------------------------------------------------------------------------------------------------ calibration
@torch.no_grad()
def calibrate(agent, env, sim, lba: dict, episodes: int, seed_start: int) -> dict:
	"""Closed-loop k = 1 episodes; every chosen plan is also executed in the simulator copy.

	Per decision: the gate scores [L], the true return error E [L] (oracle and labels), and the latent prediction
	error PE [L] (distance of the encoded real state after j+1 plan actions from the imagined z_{j+1})."""
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	dev = next(agent.model.parameters()).device
	discount = float(agent.discount)
	L = lba["L"]
	rec = dict(steps={}, E=[], PE=[], ep=[], t=[])
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			plan = planner.last_plan.detach().cpu()
			steps, roll = score_plan(agent, obs, plan, lba)
			snap = sim.save()
			r_true, o_true = sim.rollout(snap, plan.view(plan.shape[0], 1, -1))
			rec["E"].append(plan_error(roll, r_true[:, 0], discount))
			z_true = agent.model.encode(o_true[:L, 0].to(dev), None)
			rec["PE"].append((z_true - roll.z[1:L + 1, 0]).norm(dim=-1).cpu())
			for k, v in steps.items():
				rec["steps"].setdefault(k, []).append(v)
			rec["ep"].append(i)
			rec["t"].append(t)
			obs, _, done, _ = env.step(action)
			t += 1
	return dict(steps={k: torch.stack(v) for k, v in rec["steps"].items()}, E=torch.stack(rec["E"]),
	            PE=torch.stack(rec["PE"]), ep=torch.tensor(rec["ep"]), t=torch.tensor(rec["t"]))


def expected_k(c: torch.Tensor, thr: float, tie: float) -> float:
	"""Mean commitment of commit_len over plans with running-max scores c [N, K_MAX-1] (ties pass w.p. tie)."""
	p = torch.where(c < thr, 1.0, torch.where(c == thr, tie, 0.0))
	return float(1 + torch.cumprod(p, 1).sum(1).mean())


def thresholds_from(cal: dict) -> dict:
	"""{gate: {budget: {thr, tie}}} so that each gate's mean commitment on the calibration plans equals the budget.

	The threshold is the smallest observed score at which trusting ties reaches the budget; the tie probability is
	then set by bisection, so the match is exact even when scores repeat (e.g. probabilities saturated at 1.0)."""
	scores = dict(cal["steps"], oracle=cal["E"], PE=cal["PE"])
	out = {}
	for g, s in scores.items():
		c = torch.cummax(s[:, :K_MAX - 1].float(), 1).values
		vals = torch.unique(c)
		out[g] = {}
		for b in BUDGETS:
			lo, hi = 0, len(vals) - 1          # smallest index with expected_k(vals[i], tie=1) >= b
			while lo < hi:
				mid = (lo + hi) // 2
				if expected_k(c, float(vals[mid]), 1.0) >= b:
					hi = mid
				else:
					lo = mid + 1
			thr = float(vals[lo])
			t_lo, t_hi = 0.0, 1.0
			for _ in range(40):
				t = (t_lo + t_hi) / 2
				t_lo, t_hi = (t, t_hi) if expected_k(c, thr, t) < b else (t_lo, t)
			out[g][str(b)] = dict(thr=thr, tie=(t_lo + t_hi) / 2)
	return out


# ------------------------------------------------------------------------------------------------ episodes
@torch.no_grad()
def run_arm(agent, env, sim, arm: Arm, lba: dict, th: dict, episodes: int, seed_start: int, seed: int = 0):
	"""Run one arm on the evaluation seeds; returns per-episode rows and the per-plan commitments."""
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	gen = torch.Generator().manual_seed(seed)
	discount = float(agent.discount)
	q = random_q(arm.budget) if arm.gate == "random" else None
	pe_th = th["PE"][str(arm.budget)] if arm.gate == "PE" or arm.abort else None
	rows, rec = [], dict(ep=[], t=[], k=[])
	for i in range(episodes):
		obs, done, t, ep_reward, calls, j = env.reset(seed=seed_start + i), False, 0, 0.0, 0, 0
		start = time.perf_counter()
		while not done:
			shift_warm_start(planner, j)
			planner.act(obs, t0=calls == 0)
			plan = planner.last_plan.detach().cpu().clamp(-1, 1)
			calls += 1
			roll = None
			if arm.gate is None:
				k = arm.k
			elif arm.gate == "random":
				k = 1
				while k < K_MAX and bool(torch.rand(1, generator=gen) < q):
					k += 1
			elif arm.gate == "PE":
				k = K_MAX
				_, roll = score_plan(agent, obs, plan, lba)
			elif arm.gate == "oracle":
				steps, roll = score_plan(agent, obs, plan, lba)
				r_true, _ = sim.rollout(sim.save(), plan.view(plan.shape[0], 1, -1))
				k = commit_len(plan_error(roll, r_true[:, 0], discount), th["oracle"][str(arm.budget)], gen)
			else:
				steps, roll = score_plan(agent, obs, plan, lba)
				k = commit_len(steps[arm.gate], th[arm.gate][str(arm.budget)], gen)
			rec["ep"].append(i)
			rec["t"].append(t)
			j = 0
			while True:
				obs, reward, done, info = env.step(plan[j])
				ep_reward, t, j = ep_reward + float(reward), t + 1, j + 1
				if done or j >= k:
					break
				if pe_th is not None and not passes(latent_error(agent, obs, roll.z[j, 0]), pe_th, gen):
					break
			rec["k"].append(j)
		rows.append(dict(arm=arm.name, episode=i, seed=seed_start + i, success=float(info["success"]), reward=ep_reward,
		                 steps=t, plan_calls=calls, mean_k=t / calls, seconds=time.perf_counter() - start))
	return rows, {k: torch.tensor(v) for k, v in rec.items()}


def write_outputs(out: Path, arms: list):
	"""episodes.csv (one row per arm and episode) and summary.json, from the arm files."""
	import csv
	summary, all_rows = {}, []
	for arm in arms:
		rows = torch.load(out / "arms" / f"{arm.name}.pt", weights_only=False)["rows"]
		all_rows += rows
		summary[arm.name] = dict(success=sum(r["success"] for r in rows) / len(rows),
		                         mean_k=sum(r["steps"] for r in rows) / sum(r["plan_calls"] for r in rows),
		                         episodes=len(rows))
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
	# 0. the learned Bellman audit, fitted on the robot's saved real episodes (as in the monitor study)
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
	# 1. thresholds from closed-loop k = 1 calibration episodes (separate seeds)
	if (out / "thresholds.json").exists():
		th = json.loads((out / "thresholds.json").read_text())
	else:
		cal = calibrate(agent, env, sim, lba, args.cal_episodes, args.cal_seed_start)
		th = thresholds_from(cal)
		torch.save(cal, out / "calibration.pt")
		(out / "thresholds.json").write_text(json.dumps(th, indent=1))
		print(f"thresholds: {json.dumps(th)}", flush=True)

	# 2. arms, same evaluation seeds
	arms = arms_for(has_ensemble)
	if args.arms:
		keep = set(args.arms.split(","))
		arms = [a for a in arms if a.name in keep]
	for n, arm in enumerate(arms):
		f = out / "arms" / f"{arm.name}.pt"
		if f.exists():
			continue
		rows, rec = run_arm(agent, env, sim, arm, lba, th, args.episodes, args.seed_start, seed=n)
		torch.save(dict(rows=rows, rec=rec), f)
		succ = sum(r["success"] for r in rows) / len(rows)
		mk = sum(r["steps"] for r in rows) / sum(r["plan_calls"] for r in rows)
		sec = sum(r["seconds"] for r in rows) / len(rows)
		print(f"{arm.name:12s} success {succ:.2f}  actions per plan {mk:.2f}  {sec:.1f} s/episode", flush=True)
		if args.max_hours is not None and time.time() - job_start > args.max_hours * 3600 and n < len(arms) - 1:
			print("time limit reached: checkpointing (exit 85)", flush=True)
			env.close()
			sys.exit(85)
	env.close()

	write_outputs(out, arms)
	print("done", flush=True)


if __name__ == "__main__":
	main()
