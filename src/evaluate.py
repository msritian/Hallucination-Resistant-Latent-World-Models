"""Evaluate planner arms on a trained model (execution_final.md §7.2–7.4).

1. Calibrate tau(t) for every signal and percentile on fresh episodes collected with the stock H=3 planner
   (calibration seeds, disjoint from evaluation seeds), using windows up to the longest horizon evaluated.
2. Run each planner arm for the same evaluation episodes (same seeds for every arm -> paired comparisons).
3. Write one CSV row per (arm, episode) and a summary per arm.

Example:
	python -m src.evaluate --model run/final_model.pt --run grounded --task PushCube-v1 --preset pilot --out eval_push
"""
import argparse
import json
from pathlib import Path

import torch

from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from src.preflight.analysis import calibrate_all
from src.preflight.run_preflight import CAL, rollout_signals, windows
from src.utils.csvlog import CSVLog

TAU_PCTS = (90, 95, 99)


def preset(name: str, grounded: bool) -> list:
	"""Named sets of planner arms. 'stock' = TD-MPC2's own planner (Arm 0)."""
	P = PlannerConfig
	if name == "pilot":
		arms = ["stock", P(horizon=3, score="standard")]
		for H in (3, 6, 12, 24):
			arms += [P(horizon=H, score="standard"), P(horizon=H, score="trust", signal="A"), P(horizon=H, score="elvis")]
		arms += [P(horizon=12, score="trust", signal=s) for s in ("B", "C", "E")]
		if grounded:
			arms += [P(horizon=12, score="trust", signal=s) for s in ("D", "M")]
		return _dedupe(arms)
	if name == "final":
		# Final task performance of a trained agent: TD-MPC2's own planner (H=3), for comparing training variants.
		return ["stock"]
	if name == "tuning":
		arms = [P(horizon=12, score="trust", signal="A", kappa=k, tau_pct=p)
		        for k in (0.5, 1.0, 2.0, float("inf")) for p in TAU_PCTS]
		arms += [P(horizon=12, score="elvis", lambda_min=lmin, lambda_max=lmax, beta=b)
		         for lmin in (0.0, 0.5) for lmax in (0.95, 1.0) for b in (0.5, 1.0, 2.0)]
		return arms
	if name == "fixes":
		# Post-hoc planner variants (2026-09-28): safer fallbacks, one-sided / training-target residuals, random control.
		arms = ["stock", P(horizon=3, score="standard")]
		for H in (12, 24):
			arms += [P(horizon=H, score="standard")]
			arms += [P(horizon=H, score="trust", signal="A", fallback=fb) for fb in ("q", "qmin", "v")]
			arms += [P(horizon=H, score="trust", signal=s, fallback="v") for s in ("Ao", "At")]
		arms += [P(horizon=12, score="trust", signal="A", shuffle_trust=True)]
		arms += [P(horizon=12, score="trust", signal="A", fallback="v", kappa=k) for k in (0.25, float("inf"))]
		return _dedupe(arms)
	if name == "confirm":
		# Pre-registered 2026-09-28 (execution_final.md, Phase 3): one rule chosen from the stock-YCB and grounded-Push
		# planning diagnostics, tested closed-loop on fresh seeds. Controls: standard scoring, the original rule, and
		# the same rule with trust randomly reassigned across candidates.
		arms = ["stock", P(horizon=3, score="standard")]
		for H in (12, 24):
			arms += [P(horizon=H, score="standard"),
			         P(horizon=H, score="trust", signal="Ao", fallback="qmin"),
			         P(horizon=H, score="trust", signal="A")]
		arms += [P(horizon=12, score="trust", signal="Ao", fallback="qmin", shuffle_trust=True)]
		return _dedupe(arms)
	if name == "structure":
		# Why does the ELVIS-style lambda-return work? Same structure with a constant lambda (no signal) and with our
		# residual in place of the UCB signal; plus standard scoring and our pre-registered trust rule.
		arms = ["stock", P(horizon=3, score="standard")]
		for H in (12, 24):
			arms += [P(horizon=H, score="standard"),
			         P(horizon=H, score="elvis"),
			         P(horizon=H, score="elvis", lambda_signal="A"),
			         P(horizon=H, score="trust", signal="Ao", fallback="qmin")]
			arms += [P(horizon=H, score="elvis", lambda_signal="const", lambda_max=lm) for lm in (0.5, 0.8)]
		return _dedupe(arms)
	if name == "lambda":
		# Pre-registered 2026-09-28 (execution_final.md, Phase 3): C1 constant lambda vs ELVIS, C2 long-horizon
		# constant lambda vs H3, C3 optimism-residual penalty on top of constant lambda.
		arms = ["stock", P(horizon=3, score="standard")]
		for H in (12, 24):
			arms += [P(horizon=H, score="standard"), P(horizon=H, score="elvis")]
			arms += [P(horizon=H, score="elvis", lambda_signal="const", lambda_max=lm) for lm in (0.5, 0.8, 0.9)]
			arms += [P(horizon=H, score="elvis", lambda_signal="const", lambda_max=0.8, penalty=b) for b in (0.5, 1.0)]
		return _dedupe(arms)
	if name == "pessimism":
		# Pessimistic value (min over critic heads; Chang et al., ICML 2026) with standard and constant-lambda scoring,
		# at short and long horizons.
		arms = ["stock"]
		for H in (3, 12, 24):
			arms += [P(horizon=H, score="standard"), P(horizon=H, score="standard", value_agg="min")]
		for H in (12, 24):
			arms += [P(horizon=H, score="elvis", lambda_signal="const", lambda_max=0.8),
			         P(horizon=H, score="elvis", lambda_signal="const", lambda_max=0.8, value_agg="min")]
		return _dedupe(arms)
	raise ValueError(f"unknown preset {name}")


def _dedupe(arms):
	seen, out = set(), []
	for a in arms:
		key = a if isinstance(a, str) else a.name()
		if key not in seen:
			seen.add(key)
			out.append(a)
	return out


def calibrate_taus(agent, data: dict, max_h: int, beta: float) -> dict:
	"""{tau_pct: {signal: [tau(0..max_h-1)]}} from ground-truth-clean open-loop replays."""
	dev = next(agent.model.parameters()).device
	o, a, _ = windows(data, max_h, range(data["obs"].shape[0]))
	with torch.no_grad():
		sig, err, _ = rollout_signals(agent, agent.model.encode(o.to(dev), None), a.to(dev), beta)
	sig = {k: v.cpu() for k, v in sig.items()}
	out = {}
	for pct in TAU_PCTS:
		taus, eps = calibrate_all(sig, err.cpu(), dict(CAL, tau_pct=pct))
		out[pct] = {k: t.tau.tolist() for k, t in taus.items()}
		out["error_thresholds"] = eps
	return out


def run_episodes(env, actor, episodes: int, seed_start: int) -> list:
	rows = []
	for i in range(episodes):
		obs, done, t, ep_reward = env.reset(seed=seed_start + i), False, 0, 0.0
		while not done:
			action = actor(obs, t == 0)
			obs, reward, done, info = env.step(action)
			ep_reward += float(reward)
			t += 1
		rows.append(dict(episode=i, seed=seed_start + i, success=info["success"], reward=ep_reward))
	return rows


def evaluate_arms(agent, env, arms, taus_by_pct, episodes, seed_start, out: Path):
	log = CSVLog(out / "episodes.csv")
	summary = {}
	for arm in arms:
		if arm == "stock":
			name, planner = "stock_tdmpc2_H3", None
			actor = lambda obs, t0: agent.act(obs, t0=t0, eval_mode=True)
		else:
			name = arm.name()
			planner = AuditedPlanner(agent, arm, taus_by_pct.get(arm.tau_pct))
			actor = lambda obs, t0, p=planner: p.act(obs, t0=t0, eval_mode=True)
		rows = run_episodes(env, actor, episodes, seed_start)
		for r in rows:
			log.write(dict(arm=name, **r))
		s = dict(success=sum(r["success"] for r in rows) / len(rows), reward=sum(r["reward"] for r in rows) / len(rows))
		if planner is not None:
			for k, v in planner.stats.items():
				s[f"mean_{k}"] = sum(v) / max(len(v), 1)
		summary[name] = s
		print(f"{name:45s} success {s['success']:.2f}  reward {s['reward']:.1f}" +
		      (f"  plan_ms {s.get('mean_plan_ms', 0):.1f}  H_eff {s.get('mean_h_eff', float('nan')):.1f}" if planner else ""))
		(out / "summary.json").write_text(json.dumps(summary, indent=1))
	return summary


def main(argv=None):
	from src.envs.maniskill3 import TASKS, ManiSkill3Env
	from src.preflight.run_preflight import collect, load_agent
	from src.train import RUNS

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", choices=list(RUNS), required=True)
	p.add_argument("--task", choices=TASKS, required=True)
	p.add_argument("--preset", default="pilot")
	p.add_argument("--out", required=True)
	p.add_argument("--episodes", type=int, default=50)
	p.add_argument("--seed_start", type=int, default=1000, help="evaluation seeds (tuning uses 3000)")
	p.add_argument("--cal_episodes", type=int, default=50)
	p.add_argument("--cal_seed_start", type=int, default=2000)
	p.add_argument("--beta", type=float, default=1.0)
	p.add_argument("--taus_file", default=None,
	               help="thresholds from a file (e.g. plan_diag's taus_candidates.json) instead of on-policy calibration")
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	agent = load_agent(args.model, args.run, args.task)
	arms = preset(args.preset, grounded=RUNS[args.run]["num_aux_dynamics"] > 0)  # D/M arms need the extra heads
	planners = [a for a in arms if not isinstance(a, str)]
	taus = {}
	if planners and args.taus_file:
		taus = {int(k) if k.isdigit() else k: v for k, v in json.loads(Path(args.taus_file).read_text()).items()}
		max_h = max(a.horizon for a in planners)
		for pct in TAU_PCTS:  # carry the last threshold forward, as the calibrator does when data runs out
			if pct in taus:
				taus[pct] = {k: t + [t[-1]] * max(0, max_h - len(t)) for k, t in taus[pct].items()}
		print(f"Thresholds from {args.taus_file}")
	elif planners:  # thresholds are only needed for audited/ELVIS planner arms
		max_h = max(a.horizon for a in planners)
		print(f"Calibrating thresholds on {args.cal_episodes} episodes (windows up to H={max_h})...")
		cal = collect(agent, args.task, args.cal_episodes, args.cal_seed_start)
		taus = calibrate_taus(agent, cal, max_h, args.beta)
		(out / "taus.json").write_text(json.dumps(taus, indent=1))
	env = ManiSkill3Env(args.task, seed=args.seed_start)
	evaluate_arms(agent, env, arms, taus, args.episodes, args.seed_start, out)


if __name__ == "__main__":
	main()
