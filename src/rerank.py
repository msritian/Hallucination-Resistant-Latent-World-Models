"""Idea 1 pilot: correct the over-optimism of candidate plans with the Bellman audit (docs/preregistration_rerank.md).

At every real step MPPI (standard scoring) runs as usual; then one of its top-K candidates is executed, chosen by:
  default    TD-MPC2-style sample from the elite softmax (the planner's own choice)
  imagined   argmax imagined score
  corrected  argmax imagined score - predicted over-estimation (trees on audit features; signed, learned from reality)
  penD/penM  argmax imagined score - lambda * ensemble uncertainty (lambda tuned on calibration data)
  oracle     argmax TRUE score (all K candidates executed in the simulator): the ceiling
  random     a random candidate (control)
TRUE score of a candidate = sum_k gamma^k r_true_k + gamma^H V(encode(real obs after H steps)), the same quantity the
imagined score estimates.

One job = one robot: (1) calibration episodes with every top-K candidate executed in the simulator -> fit the correction
(first half) and measure offline selection quality on the second half; (2) closed-loop arms on shared seeds.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

from src.preflight.run_preflight import CRITIC_FEATURES, _features, calibration_bias, plan_signals, value_fn, windows

RULES = ("H3", "default", "imagined", "corrected", "relvalue", "valuepred", "penD", "penM", "oracle", "random")
LEARNED = ("corrected", "relvalue", "valuepred")


# ------------------------------------------------------------------------------------------------ candidates
@torch.no_grad()
def candidates(planner, K: int):
	"""Top-K candidates of the planner's last MPPI iteration: actions [H, K, A], imagined scores [K]."""
	pool = planner.last_pool
	v = pool["value"].squeeze(-1)
	idx = v.topk(K).indices
	return pool["actions"][:, idx], v[idx]


@torch.no_grad()
def plan_features(agent, obs, acts, bias, beta=1.0):
	"""Plan-level features [K, 3F+2]: for each audit feature its last, max and mean over steps, plus the imagined
	score's ingredients (sum of imagined rewards, terminal value); and ensemble uncertainties (D, M) [K]."""
	dev = next(agent.model.parameters()).device
	H, K = acts.shape[:2]
	z0 = agent.model.encode(obs.to(dev).view(1, -1), None).repeat(K, 1)
	sig, roll = plan_signals(agent, z0, acts.to(dev), beta, bias=bias)
	L = H - 1
	X = _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, L, True).view(L, K, -1)
	disc = agent.discount ** torch.arange(H, dtype=torch.float32).view(-1, 1)
	extra = torch.stack([(disc * roll.r_hat.squeeze(-1).cpu()).sum(0), roll.v_mean[-1].squeeze(-1).cpu()], -1)
	Xp = torch.cat([X[-1], X.max(0).values, X.mean(0), extra], -1)
	unc = {k: sig[k][:L].max(0).values.cpu().float() for k in ("D", "M") if k in sig}
	return Xp, unc


@torch.no_grad()
def true_scores(agent, sim, snap, acts):
	"""Execute each candidate in the simulator from the snapshot: TRUE score [K]."""
	dev = next(agent.model.parameters()).device
	rew, obs = sim.rollout(snap, acts.cpu())                                  # [H, K], [H, K, D]
	H = acts.shape[0]
	disc = agent.discount ** torch.arange(H, dtype=torch.float32).view(-1, 1)
	v_end = value_fn(agent, agent.model.encode(obs[-1].to(dev), None)).cpu().view(-1)
	return (disc * rew).sum(0) + agent.discount ** H * v_end


def _inputs(X, imagined, centered: bool):
	Z = torch.cat([X, imagined.view(-1, 1)], -1)
	return (Z - Z.mean(0, keepdim=True)) if centered else Z


def choose(rule, imagined, X, unc, model, lam, gen, true=None):
	K = len(imagined)
	if rule == "imagined":
		return int(imagined.argmax())
	if rule == "corrected":
		return int((imagined - torch.as_tensor(model["corrected"].predict(X.numpy()), dtype=torch.float32)).argmax())
	if rule == "relvalue":   # true value relative to the other candidates of the same decision
		return int(model["relvalue"].predict(_inputs(X, imagined, True).numpy()).argmax())
	if rule == "valuepred":  # true value directly
		return int(model["valuepred"].predict(_inputs(X, imagined, False).numpy()).argmax())
	if rule in ("penD", "penM"):
		return int((imagined - lam[rule] * unc[rule[-1]]).argmax())
	if rule == "oracle":
		return int(true.argmax())
	if rule == "random":
		return int(torch.randint(K, (1,), generator=gen))
	raise ValueError(rule)


# ------------------------------------------------------------------------------------------------ calibration
@torch.no_grad()
def collect_calibration(agent, env, sim, planner_fn, bias, K, episodes, seed_start):
	recs = []
	for i in range(episodes):
		planner = planner_fn()
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			acts, imagined = candidates(planner, K)
			X, unc = plan_features(agent, obs, acts, bias)
			snap = sim.save()
			true = true_scores(agent, sim, snap, acts)
			recs.append(dict(ep=i, X=X, unc=unc, imagined=imagined.cpu(), true=true))
			obs, _, done, _ = env.step(action)
			t += 1
	return recs


def fit_correction(recs):
	"""Three learned rules from the calibration decisions (each with its K candidates and their TRUE scores)."""
	from sklearn.ensemble import HistGradientBoostingRegressor
	reg = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
	                                            random_state=0)
	X = torch.cat([r["X"] for r in recs]).numpy()
	over = torch.cat([r["imagined"] - r["true"] for r in recs]).numpy()                  # signed over-estimation
	Zc = torch.cat([_inputs(r["X"], r["imagined"], True) for r in recs]).numpy()
	rel = torch.cat([r["true"] - r["true"].mean() for r in recs]).numpy()                # true value vs the others
	Z = torch.cat([_inputs(r["X"], r["imagined"], False) for r in recs]).numpy()
	tru = torch.cat([r["true"] for r in recs]).numpy()
	return dict(corrected=reg().fit(X, over), relvalue=reg().fit(Zc, rel), valuepred=reg().fit(Z, tru))


def tune_lambda(recs, key):
	"""Penalty weight maximising the true score of the chosen candidate on calibration data (generous to the baseline)."""
	if key not in recs[0]["unc"]:
		return None
	best, best_val = 0.0, -float("inf")
	for lam in [0.0, 0.1, 0.3, 1, 3, 10, 30, 100]:
		val = sum(float(r["true"][(r["imagined"] - lam * r["unc"][key]).argmax()]) for r in recs) / len(recs)
		if val > best_val:
			best, best_val = lam, val
	return best


def offline_quality(recs, model, lam):
	"""Mean regret (best true score - chosen true score) of each rule on held-out calibration decisions."""
	gen = torch.Generator().manual_seed(0)
	out = {}
	for rule in ("imagined", "corrected", "relvalue", "valuepred", "penD", "penM", "random"):
		if rule.startswith("pen") and lam.get(rule) is None:
			continue
		reg = [float(r["true"].max() - r["true"][choose(rule, r["imagined"], r["X"], r["unc"], model, lam, gen)]) for r in recs]
		out[rule] = sum(reg) / len(reg)
	return out


# ------------------------------------------------------------------------------------------------ closed loop
@torch.no_grad()
def run_rule(agent, env, sim, planner_fn, rule, bias, K, model, lam, episodes, seed_start, seed=0):
	gen = torch.Generator().manual_seed(seed)
	rows = []
	for i in range(episodes):
		planner = planner_fn()
		obs, done, t, start = env.reset(seed=seed_start + i), False, 0, time.perf_counter()
		while not done:
			action = planner.act(obs, t0=t == 0)
			if rule != "default":
				acts, imagined = candidates(planner, K)
				X, unc = plan_features(agent, obs, acts, bias) if rule in LEARNED + ("penD", "penM") else (None, None)
				true = true_scores(agent, sim, sim.save(), acts) if rule == "oracle" else None
				action = acts[0, choose(rule, imagined.cpu(), X, unc, model, lam, gen, true)].cpu()
			obs, _, done, info = env.step(action)
			t += 1
		rows.append(dict(arm=rule, episode=i, seed=seed_start + i, success=float(info["success"]),
		                 seconds=time.perf_counter() - start))
	return rows


def main(argv=None):
	from src.envs.maniskill3 import ManiSkill3Env
	from src.monitor import write_outputs
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	from src.preflight.run_preflight import load_agent
	from src.tools.plan_diag import SimSnapshot

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", required=True, help="saved real episodes (for the audit's bias term)")
	p.add_argument("--out", required=True)
	p.add_argument("--horizon", type=int, default=12)
	p.add_argument("--K", type=int, default=16)
	p.add_argument("--cal_episodes", type=int, default=20)
	p.add_argument("--episodes", type=int, default=50)
	p.add_argument("--seed_start", type=int, default=7000)
	p.add_argument("--cal_seed_start", type=int, default=7900)
	p.add_argument("--max_hours", type=float, default=None)
	a = p.parse_args(argv)
	out = Path(a.out)
	(out / "arms").mkdir(parents=True, exist_ok=True)
	start = time.time()
	agent = load_agent(a.model, a.run, a.task)
	dev = next(agent.model.parameters()).device
	planner_fn = lambda: AuditedPlanner(agent, PlannerConfig(horizon=a.horizon, score="standard"))
	env = ManiSkill3Env(a.task, seed=a.seed_start)
	sim = SimSnapshot(env)

	if (out / "correction.pkl").exists():
		bias, model, lam = pickle.loads((out / "correction.pkl").read_bytes())
	else:
		data = torch.load(a.episodes_file)
		o, act, _ = windows(data, a.horizon, range(data["obs"].shape[0]))
		with torch.no_grad():
			bias = calibration_bias(agent, agent.model.encode(o.to(dev), None), act.to(dev), 1.0, a.horizon).to(dev)
		recs = collect_calibration(agent, env, sim, planner_fn, bias, a.K, a.cal_episodes, a.cal_seed_start)
		torch.save(recs, out / "calibration.pt")   # every candidate's features, imagined and true score
		half = a.cal_episodes // 2
		fit, held = [r for r in recs if r["ep"] < half], [r for r in recs if r["ep"] >= half]
		model_h = fit_correction(fit)
		lam_h = {r: tune_lambda(fit, r[-1]) for r in ("penD", "penM")}
		quality = offline_quality(held, model_h, lam_h)
		over = torch.cat([r["imagined"] - r["true"] for r in recs])
		info = dict(offline_regret_heldout=quality, lambda_fit_half=lam_h, decisions=len(recs),
		            mean_overestimation=float(over.mean()), mean_overestimation_of_imagined_best=
		            float(sum(float(r["imagined"].max() - r["true"][r["imagined"].argmax()]) for r in recs) / len(recs)))
		(out / "offline.json").write_text(json.dumps(info, indent=1))
		print("offline (held-out calibration):", json.dumps(info), flush=True)
		model, lam = fit_correction(recs), {r: tune_lambda(recs, r[-1]) for r in ("penD", "penM")}
		(out / "correction.pkl").write_bytes(pickle.dumps((bias, model, lam)))

	rules = [r for r in RULES if not (r.startswith("pen") and lam.get(r) is None)]
	for k, rule in enumerate(rules):
		f = out / "arms" / f"{rule}.pt"
		if f.exists():
			continue
		if rule == "H3":   # reference: standard TD-MPC2-length planning (3 steps), the planner's own choice
			h3 = lambda: AuditedPlanner(agent, PlannerConfig(horizon=3, score="standard"))
			rows = run_rule(agent, env, sim, h3, "default", bias, a.K, model, lam, a.episodes, a.seed_start, seed=k)
			for r in rows:
				r["arm"] = "H3"
		else:
			rows = run_rule(agent, env, sim, planner_fn, rule, bias, a.K, model, lam, a.episodes, a.seed_start, seed=k)
		torch.save(dict(rows=rows), f)
		print(f"{rule:10s} success {sum(r['success'] for r in rows) / len(rows):.2f}  "
		      f"{sum(r['seconds'] for r in rows) / len(rows):.1f} s/episode", flush=True)
		if a.max_hours is not None and time.time() - start > a.max_hours * 3600 and k < len(rules) - 1:
			env.close()
			sys.exit(85)
	env.close()
	write_outputs(out, [type("A", (), {"name": r})() for r in rules])
	print("done", flush=True)


if __name__ == "__main__":
	main()
