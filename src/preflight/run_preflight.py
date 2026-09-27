"""Phase 2 go/no-go preflight on trained checkpoints (execution_final.md §3.4 and §6, Phase 2).

For each model (grounded, and stock for comparison):
  1. collect fresh episodes with the stock H=3 planner (calibration seeds);
  2. replay the world model open-loop along the real actions (all windows of length H) and compute every signal
     plus the true latent error;
  3. calibrate tau(t) on the first half of the episodes, evaluate detection on the second half;
  4. plant hallucinations at imagined step 3 (Types 1-3) and label natural mistakes (Type 4);
  5. report AUROCs with episode-level bootstrap CIs and apply the go/no-go rule to the grounded model.

Example:
	python -m src.preflight.run_preflight --task PushCube-v1 \
		--grounded_model grounded/final_model.pt --stock_model stock/final_model.pt --out preflight_push
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.auditor.injection import goal_interpolation, injector_at, structured_noise, value_neutral_match
from src.auditor.signals import audit_rollout, bellman_target_spread, dynamics_disagreement, pi_mean, q_heads
from src.preflight.analysis import calibrate_all, natural_labels, normalize, replay_scores
from src.preflight.metrics import bootstrap_auroc, go_no_go

TASKS = ["PushCube-v1", "PickSingleYCB-v1", "PegInsertionSide-v1", "StackCube-v1", "PickCube-v1"]

CAL = dict(eps_clean_pct=90, eps_hall_pct=99, tau_pct=95, min_clean_samples=200)
INJECT_STEP = 3
SIGMAS = [0.5, 1.0, 2.0]
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_agent(model_path: str, run: str, task: str):
	from src.agents.grounded_tdmpc2 import GroundedTDMPC2
	from src.envs.maniskill3 import ManiSkill3Env
	from src.train import build_cfg

	env = ManiSkill3Env(task, seed=0)
	args = SimpleNamespace(run=run, task=task, seed=0, steps=1, model_size=5, no_compile=True, out_dir="preflight_tmp")
	agent = GroundedTDMPC2(build_cfg(args, env))
	env.close()
	agent.model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
	agent.model.eval()
	return agent


@torch.no_grad()
def collect(agent, task: str, episodes: int, seed_start: int) -> dict:
	from src.envs.maniskill3 import ManiSkill3Env

	env = ManiSkill3Env(task, seed=seed_start)
	obs_all, act_all, succ_all = [], [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		obs_ep, act_ep, succ_ep = [obs], [], []
		while not done:
			action = agent.act(obs, t0=t == 0, eval_mode=True)
			obs, _, done, info = env.step(action)
			obs_ep.append(obs)
			act_ep.append(action)
			succ_ep.append(info["success"])
			t += 1
		obs_all.append(torch.stack(obs_ep))
		act_all.append(torch.stack(act_ep))
		succ_all.append(torch.tensor(succ_ep))
	env.close()
	return dict(obs=torch.stack(obs_all), action=torch.stack(act_all), success=torch.stack(succ_all))


def windows(data: dict, H: int, episode_ids) -> tuple:
	"""All length-H windows of the given episodes: obs [H+1, M, D], actions [H, M, A], episode id [M]."""
	obs, act = data["obs"], data["action"]
	T = act.shape[1]
	o, a, e = [], [], []
	for ep in episode_ids:
		for s in range(T - H + 1):
			o.append(obs[ep, s:s + H + 1])
			a.append(act[ep, s:s + H])
			e.append(ep)
	return torch.stack(o, 1), torch.stack(a, 1), torch.tensor(e)


@torch.no_grad()
def rollout_signals(agent, z_real, actions, beta: float, inject=None):
	"""All signals [H, M], the true latent error e_{t+1} [H, M], and the rollout, for one batch of replays."""
	cfg, model = agent.cfg, agent.model
	roll = audit_rollout(model, cfg, z_real[0], actions, float(agent.discount), inject=inject)
	sig = {
		"A": roll.delta.squeeze(-1),
		"B": roll.critic_spread().squeeze(-1),
		"E": roll.ucb(beta)[1:].squeeze(-1),
	}
	if agent.num_aux_dynamics > 0:
		heads = agent.aux_dynamics_heads()
		sig["D"] = dynamics_disagreement(heads, roll.z, actions).squeeze(-1)
		sig["M"] = bellman_target_spread(heads, model, cfg, roll, actions).squeeze(-1)
	err = (roll.z[1:] - z_real[1:]).norm(dim=-1)
	return sig, err, roll


def value_fn(agent, z):
	"""V(z) = mean-head Q(z, mu_pi(z)); shape [...]."""
	return q_heads(agent.model, z, pi_mean(agent.model, z), agent.cfg).mean(0).squeeze(-1)


def value_error(agent, roll, z_real):
	"""|V(z_hat_{t+1}) - V(z_{t+1})| for t = 0..H-1: [H, M]."""
	return (roll.v_mean[1:].squeeze(-1) - value_fn(agent, z_real[1:])).abs()


def auroc_entry(scores, labels, groups, strata=None):
	point, lo, hi = bootstrap_auroc(scores.cpu(), labels.cpu(), n_boot=1000, groups=groups.cpu(),
	                                strata=None if strata is None else strata.cpu())
	return dict(auroc=point, ci95=[lo, hi], n_pos=int(labels.sum()), n_neg=int((~labels.bool()).sum()))


def evaluate_model(agent, data, H: int, beta: float) -> dict:
	E = data["obs"].shape[0]
	cal_eps, ev_eps = list(range(E // 2)), list(range(E // 2, E))
	dev = next(agent.model.parameters()).device
	enc = lambda o: agent.model.encode(o.to(dev), None)

	# Calibration on the first half of the episodes (natural replays).
	o, a, _ = windows(data, H, cal_eps)
	with torch.no_grad():
		z_cal = enc(o)
		sig_cal, err_cal, roll_cal = rollout_signals(agent, z_cal, a.to(dev), beta)
		verr_cal = value_error(agent, roll_cal, z_cal).cpu()
	taus, eps = calibrate_all({k: v.cpu() for k, v in sig_cal.items()}, err_cal.cpu(), CAL)
	v1 = verr_cal[0].flatten().float()
	eps.update(eps_v_clean=torch.quantile(v1, CAL["eps_clean_pct"] / 100).item(),
	           eps_v_hall=torch.quantile(v1, CAL["eps_hall_pct"] / 100).item())

	# Evaluation half: natural replays.
	o, a, ep_id = windows(data, H, ev_eps)
	with torch.no_grad():
		z_real = enc(o)
		sig_clean, err, roll_clean = rollout_signals(agent, z_real, a.to(dev), beta)
	s_clean = normalize({k: v.cpu() for k, v in sig_clean.items()}, taus)
	neg_scores = replay_scores(s_clean)
	names = list(s_clean)
	results = {n: {} for n in names}

	# Type 4: natural hallucinations, per step. The gate uses the step-stratified AUROC (positives and negatives compared
	# only within the same imagined step), because errors grow with the step index; the pooled AUROC is kept for reference.
	labels, keep = natural_labels(err.cpu(), eps["eps_clean"], eps["eps_hall"])
	groups = ep_id.view(1, -1).expand_as(labels)
	steps = torch.arange(labels.shape[0]).view(-1, 1).expand_as(labels)
	for n in names:
		results[n]["type4"] = auroc_entry(s_clean[n][keep], labels[keep], groups[keep], strata=steps[keep])
		results[n]["type4_pooled"] = auroc_entry(s_clean[n][keep], labels[keep], groups[keep])

	# Type 4v: natural hallucinations labeled by VALUE error |V(z_hat) - V(z_real)| (decision-relevant errors).
	# Primary gate from 2026-09-27 on (execution_final.md §6, Phase 2); step-stratified like Type 4.
	with torch.no_grad():
		v_err = value_error(agent, roll_clean, z_real).cpu()
		v_real = value_fn(agent, z_real[1:]).cpu()
		delta_real = (roll_clean.q_sa.squeeze(-1).cpu() - (roll_clean.r_hat.squeeze(-1).cpu() + roll_clean.discount * v_real)).abs()
	v_labels, v_keep = natural_labels(v_err, eps["eps_v_clean"], eps["eps_v_hall"])
	for n in names:
		results[n]["type4_value"] = auroc_entry(s_clean[n][v_keep], v_labels[v_keep], groups[v_keep], strata=steps[v_keep])
	lat_pos, val_pos = labels & keep, v_labels & v_keep
	raw_delta = sig_clean["A"].cpu()
	value_diag = dict(
		frac_latent_errors_that_change_value=float((lat_pos & (v_err > eps["eps_v_hall"])).sum() / max(int(lat_pos.sum()), 1)),
		frac_value_errors_that_are_latent_errors=float((val_pos & (err.cpu() > eps["eps_hall"])).sum() / max(int(val_pos.sum()), 1)),
		critic_noise_median=float(delta_real.median()),
		residual_median_on_value_errors=float(raw_delta[val_pos].median()) if val_pos.any() else float("nan"),
		residual_median_on_value_clean=float(raw_delta[v_labels.logical_not() & v_keep].median()),
	)

	def injected(fn, subset=None):
		idx = torch.arange(z_real.shape[1]) if subset is None else subset
		zr, ac = z_real[:, idx.to(dev)], a.to(dev)[:, idx]
		with torch.no_grad():
			sig, _, _ = rollout_signals(agent, zr, ac, beta, inject=injector_at(INJECT_STEP, lambda z: fn(z, idx)))
		return replay_scores(normalize({k: v.cpu() for k, v in sig.items()}, taus)), idx

	def score_vs_clean(pos_scores, idx):
		out = {}
		for n in names:
			scores = torch.cat([pos_scores[n], neg_scores[n][idx]])
			lab = torch.cat([torch.ones(len(idx), dtype=torch.bool), torch.zeros(len(idx), dtype=torch.bool)])
			out[n] = auroc_entry(scores, lab, torch.cat([ep_id[idx], ep_id[idx]]))
		return out

	# Type 1: SimNorm-preserving noise.
	for sigma in SIGMAS:
		gen = torch.Generator(device=dev).manual_seed(0)
		pos, idx = injected(lambda z, idx, s=sigma: structured_noise(z, s, agent.cfg.simnorm_dim, gen))
		for n, v in score_vs_clean(pos, idx).items():
			results[n][f"type1_sigma{sigma}"] = v

	# Type 2: interpolation toward a goal latent (first success state; fallback: highest-value real state).
	with torch.no_grad():
		pool = enc(data["obs"].reshape(-1, data["obs"].shape[-1]))
		pool_v = q_heads(agent.model, pool, pi_mean(agent.model, pool), agent.cfg).mean(0).squeeze(-1)
	succ = data["success"]
	if succ.any():
		ep, t = torch.nonzero(succ > 0)[0].tolist()
		z_goal = enc(data["obs"][ep, t + 1:t + 2])[0]
		goal_source = f"first success state (episode {ep}, step {t + 1})"
	else:
		z_goal = pool[pool_v.argmax()]
		goal_source = "highest-value real state (no successful episode)"
	pos_all, idx_all = {n: [] for n in names}, []
	for alpha in ALPHAS:
		pos, idx = injected(lambda z, idx, al=alpha: goal_interpolation(z, z_goal.expand_as(z), al))
		for n, v in score_vs_clean(pos, idx).items():
			results[n][f"type2_alpha{alpha}"] = v
		if alpha >= 0.5:
			for n in names:
				pos_all[n].append(pos[n])
			idx_all.append(idx)
	idx_cat = torch.cat(idx_all)
	for n in names:
		scores = torch.cat([torch.cat(pos_all[n]), neg_scores[n]])
		lab = torch.cat([torch.ones(len(idx_cat), dtype=torch.bool), torch.zeros(len(ep_id), dtype=torch.bool)])
		results[n]["type2"] = auroc_entry(scores, lab, torch.cat([ep_id[idx_cat], ep_id]))

	# Type 3: value-neutral teleport (equal value, far away).
	z3, v3 = roll_clean.z[INJECT_STEP], roll_clean.v_mean[INJECT_STEP].squeeze(-1)
	spread = (torch.quantile(pool_v.float(), 0.95) - torch.quantile(pool_v.float(), 0.05)).item()
	match = value_neutral_match(v3, z3, pool_v, pool, q_tol=0.02 * spread, min_dist=eps["eps_hall"])
	has = torch.nonzero(match.cpu() >= 0).flatten()
	if len(has) > 0:
		repl = pool[match[has.to(dev)]]
		pos, idx = injected(lambda z, idx: repl, subset=has)
		for n, v in score_vs_clean(pos, idx).items():
			results[n]["type3"] = v
	type3_matched = int(len(has))

	summary = {n: {k: v["auroc"] for k, v in r.items()} for n, r in results.items()}
	return dict(
		results=results, summary=summary,
		calibration=dict(**eps, tau={k: t.tau.tolist() for k, t in taus.items()},
		                 clean_counts=taus["A"].clean_counts.tolist(), carried=taus["A"].carried.tolist()),
		episodes=dict(calibration=len(cal_eps), evaluation=len(ev_eps),
		              success_rate=float(data["success"][:, -1].float().mean())),
		type2_goal=goal_source, type3_matched_windows=type3_matched,
		type4_positive_fraction=float(labels[keep].float().mean()),
		value_diagnostics=value_diag,
	)


def decisions(summary: dict) -> dict:
	"""Primary gate (value-error labels, pre-registered 2026-09-27) and the original gate (latent-error labels)."""
	pick = lambda key: {n: {"type2": v.get("type2", float("nan")), "type4": v.get(key, float("nan"))} for n, v in summary.items()}
	return dict(primary_value=go_no_go(pick("type4_value")), original_state=go_no_go(pick("type4")))


def write_report(out: Path, report: dict):
	lines = ["# Phase 2 preflight report", ""]
	for key, title in [("primary_value", "Primary gate (Type 2 + value-error Type 4v, pre-registered 2026-09-27)"),
	                   ("original_state", "Original gate (Type 2 + latent-error Type 4)")]:
		d = report["decision"][key]
		lines += [f"**{title}: {d['decision']}**" + (f", signal {d['signal']}" if d["signal"] else "") +
		          f" — best critic-ensemble AUROC on the natural test: {d['critic_ensemble_best_type4']:.3f}"]
	lines += ["", "type4 = latent-error labels, type4_value = value-error labels (both step-stratified); "
	          "type4_pooled = latent labels, all steps pooled (reference).", ""]
	for model in ["grounded", "stock"]:
		if model not in report:
			continue
		r = report[model]
		lines += [f"## {model} model", "",
		          f"Evaluation episodes success rate: {r['episodes']['success_rate']:.2f}; "
		          f"eps_clean {r['calibration']['eps_clean']:.4f}, eps_hall {r['calibration']['eps_hall']:.4f}; "
		          f"Type 4 positive fraction {r['type4_positive_fraction']:.3f}; Type 3 matched {r['type3_matched_windows']}", ""]
		vd = r["value_diagnostics"]
		lines += [f"Value diagnostics: {vd['frac_latent_errors_that_change_value']:.2f} of latent errors change value; "
		          f"critic noise (median residual with real next state) {vd['critic_noise_median']:.4f}; "
		          f"median residual on value errors {vd['residual_median_on_value_errors']:.4f} vs clean {vd['residual_median_on_value_clean']:.4f}", ""]
		tests = ["type2", "type4_value", "type4", "type4_pooled", "type3", "type1_sigma0.5", "type1_sigma1.0", "type1_sigma2.0"]
		lines.append("| Signal | " + " | ".join(tests) + " |")
		lines.append("| :-: | " + " | ".join([":-:"] * len(tests)) + " |")
		for sig, res in r["results"].items():
			cells = []
			for t in tests:
				v = res.get(t)
				cells.append("—" if v is None else f"{v['auroc']:.3f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
			lines.append(f"| {sig} | " + " | ".join(cells) + " |")
		lines.append("")
	(out / "report.md").write_text("\n".join(lines))


def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("--task", choices=TASKS, required=True)
	p.add_argument("--grounded_model", required=True)
	p.add_argument("--stock_model", default=None)
	p.add_argument("--out", required=True)
	p.add_argument("--episodes", type=int, default=50)
	p.add_argument("--seed_start", type=int, default=2000)
	p.add_argument("--horizon", type=int, default=12)
	p.add_argument("--beta", type=float, default=1.0)
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	report = {}
	for name, path, run in [("grounded", args.grounded_model, "grounded"), ("stock", args.stock_model, "stock")]:
		if path is None:
			continue
		print(f"=== {name}: loading {path}")
		agent = load_agent(path, run, args.task)
		data = collect(agent, args.task, args.episodes, args.seed_start)
		torch.save(data, out / f"episodes_{name}.pt")
		print(f"=== {name}: collected {args.episodes} episodes, success {float(data['success'][:, -1].float().mean()):.2f}")
		report[name] = evaluate_model(agent, data, args.horizon, args.beta)
		print(json.dumps(report[name]["summary"], indent=1))
		del agent
		torch.cuda.empty_cache()

	report["decision"] = decisions(report["grounded"]["summary"])
	(out / "report.json").write_text(json.dumps(report, indent=1))
	write_report(out, report)
	print(f"\nDECISION: {report['decision']}")


if __name__ == "__main__":
	main()
