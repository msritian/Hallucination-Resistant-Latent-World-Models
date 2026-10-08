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
from src.auditor.signals import (audit_rollout, bellman_target_spread, dynamics_disagreement, pi_mean, q_heads,
                                 training_target_residual)
from src.auditor.calibrator import clean_prefix_mask, error_thresholds
from src.preflight.analysis import calibrate_all, natural_labels, normalize, replay_scores
from src.preflight.metrics import bootstrap_auroc, go_no_go

TASKS = ["PushCube-v1", "PickSingleYCB-v1", "PegInsertionSide-v1", "StackCube-v1", "PickCube-v1", "PullCubeTool-v1", "PokeCube-v1", "LiftPegUpright-v1"]

CAL = dict(eps_clean_pct=90, eps_hall_pct=99, tau_pct=95, min_clean_samples=200)
INJECT_STEP = 3
K_STEPS = (2, 3, 5, 8)
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
	obs_all, act_all, succ_all, rew_all = [], [], [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		obs_ep, act_ep, succ_ep, rew_ep = [obs], [], [], []
		while not done:
			action = agent.act(obs, t0=t == 0, eval_mode=True)
			obs, reward, done, info = env.step(action)
			obs_ep.append(obs)
			act_ep.append(action)
			succ_ep.append(info["success"])
			rew_ep.append(float(reward))
			t += 1
		obs_all.append(torch.stack(obs_ep))
		act_all.append(torch.stack(act_ep))
		succ_all.append(torch.tensor(succ_ep))
		rew_all.append(torch.tensor(rew_ep))
	env.close()
	return dict(obs=torch.stack(obs_all), action=torch.stack(act_all), success=torch.stack(succ_all),
	            reward=torch.stack(rew_all))


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


def reward_windows(data: dict, H: int, episode_ids) -> torch.Tensor:
	"""True environment rewards for the same windows as ``windows``: [H, M]."""
	rew, T = data["reward"], data["action"].shape[1]
	return torch.stack([rew[ep, s:s + H] for ep in episode_ids for s in range(T - H + 1)], 1)


def return_error(roll, r_true: torch.Tensor, discount: float) -> torch.Tensor:
	"""Critic-free hallucination measure: |imagined minus real discounted return| from step 1 to t+1, for t = 0..H-2.

	Step t of a detector judges the transition into z_hat_{t+1}; the reward predicted AT z_hat_{t+1} (r_hat_{t+1}) is
	compared with the environment's reward at the real state s_{t+1} along the same actions. [H-1, M].
	"""
	d = roll.r_hat.squeeze(-1).cpu()[1:] - r_true[1:]
	disc = discount ** torch.arange(d.shape[0], dtype=d.dtype).view(-1, 1)
	return torch.cumsum(disc * d, 0).abs()


@torch.no_grad()
def rollout_signals(agent, z_real, actions, beta: float, inject=None, bias=None, ensemble: bool = True, task=None):
	"""All signals [H, M], the true latent error e_{t+1} [H, M], and the rollout, for one batch of replays.

	``bias`` [H]: the critic's typical signed residual on correct steps (from calibration). When given, adds the
	bias-corrected residuals Ab (one step) and Acb (cumulative)."""
	sig, roll = plan_signals(agent, z_real[0], actions, beta, inject=inject, bias=bias, ensemble=ensemble, task=task)
	err = (roll.z[1:] - z_real[1:]).norm(dim=-1)
	return sig, err, roll


@torch.no_grad()
def plan_signals(agent, z0, actions, beta: float, inject=None, bias=None, ensemble: bool = True, task=None):
	"""All signals [H, M] and the rollout for imagined rollouts from z0 [M, d] along actions [H, M, A].

	Needs only the start latent, so it also scores plans the robot has not executed (run-time monitoring).
	``ensemble=False`` skips the dynamics-ensemble signals D and M (faster; not inputs of the learned audit)."""
	cfg, model = agent.cfg, agent.model
	roll = audit_rollout(model, cfg, z0, actions, float(agent.discount), task=task, inject=inject)   # task: multi-task models
	sig = {
		"A": roll.delta.squeeze(-1),
		"B": roll.critic_spread().squeeze(-1),
		"E": roll.ucb(beta)[1:].squeeze(-1),
		# Exploratory variants (post-hoc, 2026-09-28; not part of the gate):
		"Ao": roll.optimism().squeeze(-1),                                  # one-sided (optimistic) residual
		"P": roll.head_residual_spread().squeeze(-1),                       # per-head residual spread
		"At": training_target_residual(model, cfg, roll, task).squeeze(-1),  # residual vs the critic's training target
		# Gradual-drift variants (2026-09-28; designed on the s10 development models):
		"Aa": roll.anchored_residual().squeeze(-1),                         # multi-step residual anchored at the real start
		"Ac": roll.cumulative_signed_residual().squeeze(-1),                # cumulative signed residual (CUSUM-style)
		**{f"A{k}": roll.k_step_residual(k).squeeze(-1) for k in K_STEPS},    # k-step residuals (k = 1 is A)
	}
	if bias is not None:
		excess = roll.signed_residual().squeeze(-1) - bias.to(roll.r_hat.device).view(-1, 1)   # [H, M]
		disc = float(agent.discount) ** torch.arange(excess.shape[0], device=excess.device, dtype=excess.dtype).view(-1, 1)
		sig["Ab"] = excess.abs()
		sig["Acb"] = torch.cumsum(disc * excess, 0).abs()
	if ensemble and getattr(agent, "num_aux_dynamics", 0) > 0:
		heads = agent.aux_dynamics_heads()
		sig["D"] = dynamics_disagreement(heads, roll.z, actions).squeeze(-1)
		sig["M"] = bellman_target_spread(heads, model, cfg, roll, actions).squeeze(-1)
	return sig, roll


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


def evaluate_model(agent, data, H: int, beta: float, dump_path=None) -> dict:
	E = data["obs"].shape[0]
	cal_eps, ev_eps = list(range(E // 2)), list(range(E // 2, E))
	dev = next(agent.model.parameters()).device
	enc = lambda o: agent.model.encode(o.to(dev), None)

	# Calibration on the first half of the episodes (natural replays).
	o, a, _ = windows(data, H, cal_eps)
	with torch.no_grad():
		z_cal = enc(o)
		sig_cal, err_cal, roll_cal = rollout_signals(agent, z_cal, a.to(dev), beta)
		# Systematic critic bias: median signed residual per step on ground-truth-clean calibration replays.
		clean = clean_prefix_mask(err_cal.cpu(), error_thresholds(err_cal.cpu(), CAL["eps_clean_pct"], CAL["eps_hall_pct"]).eps_clean)
		signed_cal = roll_cal.signed_residual().squeeze(-1).cpu()
		bias = torch.stack([signed_cal[t][clean[t]].median() if clean[t].any() else signed_cal[t].median() for t in range(H)])
		sig_cal, _, _ = rollout_signals(agent, z_cal, a.to(dev), beta, bias=bias)
		verr_cal = value_error(agent, roll_cal, z_cal).cpu()
	taus, eps = calibrate_all({k: v.cpu() for k, v in sig_cal.items()}, err_cal.cpu(), CAL)
	v1 = verr_cal[0].flatten().float()
	eps.update(eps_v_clean=torch.quantile(v1, CAL["eps_clean_pct"] / 100).item(),
	           eps_v_hall=torch.quantile(v1, CAL["eps_hall_pct"] / 100).item())

	# Evaluation half: natural replays.
	o, a, ep_id = windows(data, H, ev_eps)
	with torch.no_grad():
		z_real = enc(o)
		sig_clean, err, roll_clean = rollout_signals(agent, z_real, a.to(dev), beta, bias=bias)
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
	# Exploratory: labels mark ACCUMULATED error at step t, while a per-step signal flags the step where an error
	# STARTS. The planner uses cumulative trust (Omega), so the matching detector is the running max of s up to t.
	for n in names:
		s_cum = torch.cummax(s_clean[n], dim=0).values
		results[n]["type4_cum"] = auroc_entry(s_cum[keep], labels[keep], groups[keep], strata=steps[keep])
		results[n]["type4_value_cum"] = auroc_entry(s_cum[v_keep], v_labels[v_keep], groups[v_keep], strata=steps[v_keep])
	lat_pos, val_pos = labels & keep, v_labels & v_keep
	raw_delta = sig_clean["A"].cpu()
	value_diag = dict(
		frac_latent_errors_that_change_value=float((lat_pos & (v_err > eps["eps_v_hall"])).sum() / max(int(lat_pos.sum()), 1)),
		frac_value_errors_that_are_latent_errors=float((val_pos & (err.cpu() > eps["eps_hall"])).sum() / max(int(val_pos.sum()), 1)),
		critic_noise_median=float(delta_real.median()),
		# Step 0 only: Q at the REAL start latent vs the REAL next latent (later steps mix imagined and real states).
		critic_noise_median_step0=float(delta_real[0].median()),
		residual_median_on_value_errors=float(raw_delta[val_pos].median()) if val_pos.any() else float("nan"),
		residual_median_on_value_clean=float(raw_delta[v_labels.logical_not() & v_keep].median()),
	)

	def injected(fn, subset=None):
		idx = torch.arange(z_real.shape[1]) if subset is None else subset
		zr, ac = z_real[:, idx.to(dev)], a.to(dev)[:, idx]
		with torch.no_grad():
			sig, _, _ = rollout_signals(agent, zr, ac, beta, inject=injector_at(INJECT_STEP, lambda z: fn(z, idx)), bias=bias)
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
	pos_all, idx_all, eff_all = {n: [] for n in names}, [], []
	z3_clean, v3_clean = roll_clean.z[INJECT_STEP], roll_clean.v_mean[INJECT_STEP].squeeze(-1)
	for alpha in ALPHAS:
		pos, idx = injected(lambda z, idx, al=alpha: goal_interpolation(z, z_goal.expand_as(z), al))
		for n, v in score_vs_clean(pos, idx).items():
			results[n][f"type2_alpha{alpha}"] = v
		if alpha >= 0.5:
			for n in names:
				pos_all[n].append(pos[n])
			idx_all.append(idx)
			# Does this teleport change what the planner would believe? (No-op when the state is already near the goal.)
			with torch.no_grad():
				v_inj = value_fn(agent, goal_interpolation(z3_clean, z_goal.expand_as(z3_clean), alpha))
			eff_all.append(((v_inj - v3_clean).abs() > eps["eps_v_hall"]).cpu())
	idx_cat = torch.cat(idx_all)
	for n in names:
		scores = torch.cat([torch.cat(pos_all[n]), neg_scores[n]])
		lab = torch.cat([torch.ones(len(idx_cat), dtype=torch.bool), torch.zeros(len(ep_id), dtype=torch.bool)])
		results[n]["type2"] = auroc_entry(scores, lab, torch.cat([ep_id[idx_cat], ep_id]))
	# Exploratory: only teleports that change the value by more than eps_v_hall count as positives.
	eff = torch.cat(eff_all)
	type2_effective_fraction = float(eff.float().mean())
	if eff.any():
		for n in names:
			scores = torch.cat([torch.cat(pos_all[n])[eff], neg_scores[n]])
			lab = torch.cat([torch.ones(int(eff.sum()), dtype=torch.bool), torch.zeros(len(ep_id), dtype=torch.bool)])
			results[n]["type2_effective"] = auroc_entry(scores, lab, torch.cat([ep_id[idx_cat][eff], ep_id]))

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

	gradual = gradual_analysis(agent, roll_clean, z_real, verr_cal, v_err, eps, s_clean, groups, steps, sig_clean)
	gradual_auroc = gradual.pop("auroc")
	for n in names:
		results[n].update(gradual_auroc[n])

	# Critic-free labels: imagined vs real return from the simulator (needs episodes collected with rewards).
	real = None
	if "reward" in data:
		g = float(agent.discount)
		E_cal = return_error(roll_cal, reward_windows(data, H, cal_eps), g)
		E = return_error(roll_clean, reward_windows(data, H, ev_eps), g)
		if dump_path is not None:  # raw per-step signals and simulator labels, for offline analyses
			T = data["action"].shape[1]
			starts = lambda eps: torch.tensor([s_ for _ in eps for s_ in range(T - H + 1)])
			eps_of = lambda eps: torch.tensor([e for e in eps for _ in range(T - H + 1)])
			pack = lambda sig, roll: dict(signals={k: v.detach().cpu() for k, v in sig.items()},
			                              v_hat=roll.v_mean[1:].squeeze(-1).cpu(), r_hat=roll.r_hat.squeeze(-1).cpu())
			torch.save(dict(cal=dict(**pack(sig_cal, roll_cal), E=E_cal, episode=eps_of(cal_eps), start=starts(cal_eps)),
			                ev=dict(**pack(sig_clean, roll_clean), E=E, episode=eps_of(ev_eps), start=starts(ev_eps)),
			                success=data["success"].cpu(), H=H, discount=g), dump_path)
		learned = learned_audits({k: v.cpu() for k, v in sig_cal.items()}, roll_cal, E_cal,
		                         {k: v.cpu() for k, v in sig_clean.items()}, roll_clean, tuple(E.shape))
		s_real = dict(s_clean)
		for k, v in learned.items():  # learned scores are defined only for steps 0..H-2
			s_real[k] = torch.cat([v, v[-1:]], 0)
			results.setdefault(k, {})
		real = real_label_analysis(E_cal, E, s_real, groups, steps)
		real_auroc = real.pop("auroc")
		# Answer-key checks (2026-09-29, development analysis): threshold sensitivity and a scale-free relative error.
		rw_cal, rw = reward_windows(data, H, cal_eps), reward_windows(data, H, ev_eps)

		def scale(r):  # discounted real |reward| mass up to step t+1, same alignment as return_error
			d = r[1:].abs()
			return torch.cumsum(g ** torch.arange(d.shape[0], dtype=d.dtype).view(-1, 1) * d, 0)
		S_cal, S = scale(rw_cal), scale(rw)
		floor = 0.05 * float(S_cal.median()) + 1e-6
		variants = {
			"simP95": (E_cal, E, 0.95), "simP999": (E_cal, E, 0.999),
			"simrel": (E_cal / (S_cal + floor), E / (S + floor), 0.99),
		}
		real["answer_key_checks"] = {}
		for prefix, (ec, ee, hq) in variants.items():
			v = real_label_analysis(ec, ee, s_real, groups, steps, prefix=prefix, hall_q=hq, with_cum=False)
			for n in s_real:
				real_auroc[n].update(v["auroc"][n])
			real["answer_key_checks"][prefix] = {k: val for k, val in v.items() if k != "auroc"}
		for n in s_real:
			results[n].update(real_auroc[n])

	summary = {n: {k: v["auroc"] for k, v in r.items()} for n, r in results.items()}
	return dict(
		results=results, summary=summary,
		calibration=dict(**eps, tau={k: t.tau.tolist() for k, t in taus.items()},
		                 clean_counts=taus["A"].clean_counts.tolist(), carried=taus["A"].carried.tolist()),
		episodes=dict(calibration=len(cal_eps), evaluation=len(ev_eps),
		              success_rate=float(data["success"][:, -1].float().mean())),
		type2_goal=goal_source, type3_matched_windows=type3_matched, type2_effective_fraction=type2_effective_fraction,
		type4_positive_fraction=float(labels[keep].float().mean()),
		value_diagnostics=value_diag,
		gradual_diagnostics=gradual,
		critic_bias_by_step=bias.tolist(),
		real_label_diagnostics=real,
	)


CRITIC_FEATURES = ("A", "Aa", "Ac", "Ab", "Acb", "P", "B", "E", "At", "Ao", "A2", "A3", "A5", "A8")
ENSEMBLE_FEATURES = ("D", "M")


def _features(sig: dict, roll, keys, L: int, critic_extras: bool) -> torch.Tensor:
	"""Per-step features [L*M, F] for the learned audits (raw signals, their running max, step index)."""
	cols = []
	for k in keys:
		if k in sig:
			x = sig[k][:L].float().cpu()
			cols += [x, torch.cummax(x, 0).values]
	if critic_extras:
		cols += [roll.v_mean[1:L + 1].squeeze(-1).float().cpu(), roll.r_hat[:L].squeeze(-1).float().cpu()]
	M = cols[0].shape[1]
	cols.append(torch.arange(L, dtype=torch.float32).view(-1, 1).expand(L, M))
	return torch.stack(cols, -1).reshape(L * M, -1)


def make_gbt():
	"""The tree reader of the learned Bellman audit (hyperparameters fixed 2026-09-30)."""
	from sklearn.ensemble import HistGradientBoostingClassifier
	return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
	                                      l2_regularization=1.0, class_weight="balanced", random_state=0)


def calibration_bias(agent, z, actions, beta: float, H: int, task=None) -> torch.Tensor:
	"""The critic's typical signed residual per step [H] on ground-truth-clean replays (as in evaluate_model)."""
	with torch.no_grad():
		_, err, roll = rollout_signals(agent, z, actions, beta, task=task)
	err = err.cpu()
	clean = clean_prefix_mask(err, error_thresholds(err, CAL["eps_clean_pct"], CAL["eps_hall_pct"]).eps_clean)
	signed = roll.signed_residual().squeeze(-1).cpu()
	return torch.stack([signed[t][clean[t]].median() if clean[t].any() else signed[t].median() for t in range(H)])


def fit_lba(agent, data: dict, H: int, beta: float, episode_ids=None, task=None) -> dict:
	"""Fit the learned Bellman audit on real episodes (same features, labels and trees as ``learned_audits``).

	Returns bias [H] (for Ab, Acb), the fitted trees, and the label thresholds on the return error."""
	episode_ids = list(range(data["obs"].shape[0])) if episode_ids is None else list(episode_ids)
	dev = next(agent.model.parameters()).device
	o, a, _ = windows(data, H, episode_ids)
	with torch.no_grad():
		tk = None if task is None else torch.full((o.shape[1],), int(task), device=dev, dtype=torch.long)
		z = agent.model.encode(o.to(dev), tk)
		bias = calibration_bias(agent, z, a.to(dev), beta, H, task=tk)
		sig, _, roll = rollout_signals(agent, z, a.to(dev), beta, bias=bias, task=tk)
	E = return_error(roll, reward_windows(data, H, episode_ids), float(agent.discount))
	L = E.shape[0]
	e1 = E[0].flatten().float()
	eps_clean, eps_hall = torch.quantile(e1, 0.9).item(), torch.quantile(e1, 0.99).item()
	y = (E > eps_hall).reshape(-1)
	keep = ((E > eps_hall) | (E <= eps_clean)).reshape(-1)
	X_all = _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, L, True)
	X = X_all[keep].numpy()
	clf = make_gbt().fit(X, y[keep].numpy())
	idx = torch.randperm(X_all.shape[0], generator=torch.Generator().manual_seed(0))[:2000]
	return dict(bias=bias, clf=clf, eps_clean=eps_clean, eps_hall=eps_hall, L=L, n_pos=int(y[keep].sum()),
	            X_sample=X_all[idx],  # real features, to check a re-implementation of the trees against sklearn
	            X_cal=X_all, y_cal=y, keep_cal=keep)   # for control fits on exactly the same calibration data


def learned_audits(sig_cal: dict, roll_cal, E_cal, sig_ev: dict, roll_ev, E_shape) -> dict:
	"""Learned detectors of REAL errors, fitted on the calibration half only (hyperparameters fixed 2026-09-30).

	Labels on the calibration windows: E_cal > P99 of E_cal[0] (positive) vs E_cal <= P90 (negative), the same
	definition as the simulator answer key. Returns {name: scores [L, M_eval]}:
	  LBA      gradient-boosted trees on critic-only features (no extra models)
	  LBA_lin  logistic regression on the same features
	  LENS     gradient-boosted trees on dynamics-ensemble features (D, M)   [models with extra heads only]
	  LALL     gradient-boosted trees on both
	"""
	try:
		from sklearn.ensemble import HistGradientBoostingClassifier
		from sklearn.linear_model import LogisticRegression
		from sklearn.pipeline import make_pipeline
		from sklearn.preprocessing import StandardScaler
	except ImportError:
		print("scikit-learn unavailable: skipping learned audits")
		return {}
	L, M_ev = E_shape
	e1 = E_cal[0].flatten().float()
	eps_clean, eps_hall = torch.quantile(e1, 0.9).item(), torch.quantile(e1, 0.99).item()
	y = (E_cal > eps_hall).reshape(-1)
	keep = ((E_cal > eps_hall) | (E_cal <= eps_clean)).reshape(-1)
	if y[keep].sum() < 5:
		return {}
	gbt = make_gbt
	lin = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
	specs = {"LBA": (CRITIC_FEATURES, True, gbt), "LBA_lin": (CRITIC_FEATURES, True, lin)}
	if all(k in sig_cal for k in ENSEMBLE_FEATURES):
		specs["LENS"] = (ENSEMBLE_FEATURES, False, gbt)
		specs["LALL"] = (CRITIC_FEATURES + ENSEMBLE_FEATURES, True, gbt)
	out = {}
	for name, (keys, extras, make) in specs.items():
		X_tr = _features(sig_cal, roll_cal, keys, L, extras)[keep].numpy()
		X_ev = _features(sig_ev, roll_ev, keys, L, extras).numpy()
		clf = make().fit(X_tr, y[keep].numpy())
		out[name] = torch.as_tensor(clf.predict_proba(X_ev)[:, 1], dtype=torch.float32).reshape(L, M_ev)
	return out


def real_label_analysis(E_cal: torch.Tensor, E: torch.Tensor, s_norm: dict, groups, steps, prefix: str = "real",
                        hall_q: float = 0.99, clean_q: float = 0.9, with_cum: bool = True) -> dict:
	"""Detection judged by the SIMULATOR, with no critic involved in the labels.

	E[t] = |imagined - real discounted return| up to z_hat_{t+1} (``return_error``), [H-1, M]. Positives: E > P99 of
	calibration E[0]; negatives: E <= P90. SUDDEN if some step's increase of E exceeded the P99 of calibration
	increments, otherwise GRADUAL. Returns {"auroc": {signal: {test: entry}}, counts...}.
	"""
	def increments(v):
		return torch.cat([v[:1], v[1:] - v[:-1]], 0)

	e1 = E_cal[0].flatten().float()
	eps_clean, eps_hall = torch.quantile(e1, clean_q).item(), torch.quantile(e1, hall_q).item()
	big = torch.quantile(increments(E_cal).flatten().float(), 0.99).item()
	any_big = torch.cummax((increments(E) > big).int(), 0).values.bool()
	pos, neg = E > eps_hall, E <= eps_clean
	sudden, gradual = pos & any_big, pos & ~any_big
	L = E.shape[0]
	g, st = groups[:L], steps[:L]

	def entry(x, p):
		keep = p | neg
		if p[keep].sum() == 0 or (~p[keep]).sum() == 0:
			return dict(auroc=float("nan"), ci95=[float("nan")] * 2, n_pos=int(p.sum()), n_neg=int(neg.sum()))
		return auroc_entry(x[keep], p[keep], g[keep], strata=st[keep])

	auroc = {}
	for n, x in s_norm.items():
		x = x[:L]
		auroc[n] = {f"{prefix}_all": entry(x, pos), f"{prefix}_gradual": entry(x, gradual), f"{prefix}_sudden": entry(x, sudden)}
		if with_cum:
			x_cum = torch.cummax(x, 0).values
			auroc[n].update({f"{prefix}_all_cum": entry(x_cum, pos), f"{prefix}_gradual_cum": entry(x_cum, gradual),
			                 f"{prefix}_sudden_cum": entry(x_cum, sudden)})
	med = lambda m: float(E[m].median()) if m.any() else float("nan")
	return dict(auroc=auroc, n_sudden=int(sudden.sum()), n_gradual=int(gradual.sum()), n_neg=int(neg.sum()),
	            eps_clean=eps_clean, eps_hall=eps_hall, big_increment_threshold=big,
	            median_error_positives=med(pos), median_error_negatives=med(neg),
	            positive_error_quartiles=[float(torch.quantile(E[pos].float(), q)) for q in (0.25, 0.5, 0.75)] if pos.any() else None)


def gradual_analysis(agent, roll, z_real, verr_cal, v_err, eps, s_norm, groups, steps, sig_raw) -> dict:
	"""Split natural value errors into SUDDEN and GRADUAL and explain detector behavior on each.

	v_err[t] = |V(z_hat_{t+1}) - V(z_{t+1})|; its per-step increment inc[t] = v_err[t] - v_err[t-1] (inc[0] = v_err[0]).
	A positive (v_err > eps_v_hall) is SUDDEN if some step j <= t had inc[j] > the P99 of calibration increments,
	otherwise GRADUAL (it built up through small steps only). Negatives: v_err <= eps_v_clean.
	"""
	def increments(v):
		return torch.cat([v[:1], v[1:] - v[:-1]], 0)

	inc_cal = increments(verr_cal)
	big = torch.quantile(inc_cal.flatten().float(), 0.99).item()
	inc = increments(v_err)
	any_big = torch.cummax((inc > big).int(), 0).values.bool()
	pos = v_err > eps["eps_v_hall"]
	neg = v_err <= eps["eps_v_clean"]
	sudden, gradual = pos & any_big, pos & ~any_big

	def entry(x, p):
		keep = p | neg
		if p[keep].sum() == 0 or (~p[keep]).sum() == 0:
			return dict(auroc=float("nan"), ci95=[float("nan")] * 2, n_pos=int(p.sum()), n_neg=int(neg.sum()))
		return auroc_entry(x[keep], p[keep], groups[keep], strata=steps[keep])

	auroc = {}
	for n, x in s_norm.items():
		x_cum = torch.cummax(x, 0).values
		auroc[n] = dict(gradual=entry(x, gradual), gradual_cum=entry(x_cum, gradual),
		                sudden_natural=entry(x, sudden), sudden_natural_cum=entry(x_cum, sudden),
		                increment=entry(x, inc > big))

	# Why does the one-step residual struggle on gradual errors?
	signed = roll.signed_residual().squeeze(-1).cpu()                              # [H, M]
	noise = sig_raw["A"].cpu()[0][neg[0]].median().item()                          # one-step residual on clean steps
	inc_gradual = inc[gradual].abs().median().item() if gradual.any() else float("nan")
	# sign consistency along gradual errors: fraction of steps j <= t whose signed residual matches the sum's sign
	cons = []
	idx = torch.nonzero(gradual)
	for t, m in idx[:5000].tolist():
		sgn = torch.sign(signed[:t + 1, m])
		total = torch.sign(signed[:t + 1, m].sum())
		if t > 0 and total != 0:
			cons.append(float((sgn == total).float().mean()))
	clean_cons = []
	for t, m in torch.nonzero(neg)[:5000].tolist():
		if t > 0:
			sgn = torch.sign(signed[:t + 1, m])
			total = torch.sign(signed[:t + 1, m].sum())
			if total != 0:
				clean_cons.append(float((sgn == total).float().mean()))
	a_raw = sig_raw["A"].cpu()
	return dict(
		auroc=auroc,
		n_sudden=int(sudden.sum()), n_gradual=int(gradual.sum()), n_neg=int(neg.sum()),
		big_increment_threshold=big,
		median_step_increment_on_gradual=inc_gradual,
		median_one_step_residual_on_clean=noise,
		snr_gradual=inc_gradual / max(noise, 1e-8),
		median_A_on_gradual=float(a_raw[gradual].median()) if gradual.any() else float("nan"),
		median_A_on_sudden=float(a_raw[sudden].median()) if sudden.any() else float("nan"),
		median_A_on_clean=float(a_raw[neg].median()),
		sign_consistency_gradual=sum(cons) / max(len(cons), 1),
		sign_consistency_clean=sum(clean_cons) / max(len(clean_cons), 1),
	)


def decisions(summary: dict) -> dict:
	"""Primary gate (value-error labels, pre-registered 2026-09-27) and the original gate (latent-error labels)."""
	pick = lambda key: {n: {"type2": v.get("type2", float("nan")), "type4": v.get(key, float("nan"))} for n, v in summary.items()}
	return dict(primary_value=go_no_go(pick("type4_value")), original_state=go_no_go(pick("type4")))


EXPLORATORY = ("Ao", "P", "At", "Aa", "Ac", "A+Aa", "A+Ac", "Ab", "Acb", "Ab+Acb", "A+Aa+P", "A+A5+P") + tuple(f"A{k}" for k in K_STEPS) + tuple(f"A+A{k}" for k in K_STEPS)


def _table(results: dict, tests: list, signals: list) -> list:
	lines = ["| Signal | " + " | ".join(tests) + " |", "| :-: | " + " | ".join([":-:"] * len(tests)) + " |"]
	for sig in signals:
		cells = []
		for t in tests:
			v = results[sig].get(t)
			cells.append("—" if v is None else f"{v['auroc']:.3f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
		lines.append(f"| {sig} | " + " | ".join(cells) + " |")
	return lines


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
		lines += _table(r["results"], tests, [s for s in r["results"] if s not in EXPLORATORY])
		lines += ["", f"**Exploratory (post-hoc, not part of the gate).** Effective teleports: "
		          f"{r.get('type2_effective_fraction', float('nan')):.2f} of Type 2 injections change value by > eps_v_hall; "
		          f"critic noise at step 0: {vd.get('critic_noise_median_step0', float('nan')):.4f}. "
		          "Ao = optimistic residual, P = per-head residual spread, At = residual vs training target, "
		          "Aa = residual anchored at the real start, Ac = cumulative signed residual, Ak = k-step residual, "
		          "Ab / Acb = one-step / cumulative residual after removing the critic's calibrated per-step bias, A+X = max of the two; "
		          "*_cum = running max of the signal up to step t.", ""]
		lines += _table(r["results"], ["type2_effective", "type4_value_cum", "type4_cum", "type2", "type4_value"], list(r["results"]))
		gd = r.get("gradual_diagnostics")
		if gd:
			lines += ["", f"**Sudden vs gradual natural value errors.** {gd['n_sudden']} sudden / {gd['n_gradual']} gradual positives, "
			          f"{gd['n_neg']} negatives. Per-step error growth on gradual errors (median) {gd['median_step_increment_on_gradual']:.4f} "
			          f"vs one-step residual on clean steps {gd['median_one_step_residual_on_clean']:.4f} (ratio {gd['snr_gradual']:.2f}). "
			          f"Median A: sudden {gd['median_A_on_sudden']:.4f}, gradual {gd['median_A_on_gradual']:.4f}, clean {gd['median_A_on_clean']:.4f}. "
			          f"Sign consistency of the residual along gradual errors {gd['sign_consistency_gradual']:.2f} vs clean {gd['sign_consistency_clean']:.2f}.", ""]
			lines += _table(r["results"], ["gradual", "gradual_cum", "sudden_natural", "sudden_natural_cum", "increment"], list(r["results"]))
		rl = r.get("real_label_diagnostics")
		if rl:
			lines += ["", f"**Critic-free labels (simulator): imagined vs real return.** {rl['n_sudden']} sudden / "
			          f"{rl['n_gradual']} gradual positives, {rl['n_neg']} negatives; median error positives "
			          f"{rl['median_error_positives']:.3f} vs negatives {rl['median_error_negatives']:.3f} "
			          f"(threshold {rl['eps_hall']:.4f}).", ""]
			lines += _table(r["results"], ["real_all", "real_all_cum", "real_gradual", "real_gradual_cum", "real_sudden", "real_sudden_cum"], list(r["results"]))
			for k, c in rl.get("answer_key_checks", {}).items():
				lines += ["", f"*Answer-key check {k}:* {c['n_sudden']} sudden / {c['n_gradual']} gradual positives; median error "
				          f"positives {c['median_error_positives']:.3f} vs negatives {c['median_error_negatives']:.3f}; "
				          f"threshold {c['eps_hall']:.4f}."]
			if rl.get("answer_key_checks"):
				lines += [""] + _table(r["results"], [f"{k}_{t}" for k in rl["answer_key_checks"] for t in ("all", "gradual", "sudden")], list(r["results"]))
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
	p.add_argument("--reuse_episodes", default=None, help="directory with episodes_<model>.pt from an earlier run")
	p.add_argument("--dump_features", action="store_true", help="also save raw per-step signals and labels (features_<model>.pt)")
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	report = {}
	for name, path, run in [("grounded", args.grounded_model, "grounded"), ("stock", args.stock_model, "stock")]:
		if path is None:
			continue
		print(f"=== {name}: loading {path}")
		agent = load_agent(path, run, args.task)
		saved = None if args.reuse_episodes is None else Path(args.reuse_episodes) / f"episodes_{name}.pt"
		if saved is not None and saved.exists():
			data = torch.load(saved)
			print(f"=== {name}: reusing {saved}")
		else:
			data = collect(agent, args.task, args.episodes, args.seed_start)
		torch.save(data, out / f"episodes_{name}.pt")
		print(f"=== {name}: collected {args.episodes} episodes, success {float(data['success'][:, -1].float().mean()):.2f}")
		report[name] = evaluate_model(agent, data, args.horizon, args.beta,
		                              dump_path=(out / f"features_{name}.pt") if args.dump_features else None)
		print(json.dumps(report[name]["summary"], indent=1))
		del agent
		torch.cuda.empty_cache()

	report["decision"] = decisions(report["grounded"]["summary"])
	(out / "report.json").write_text(json.dumps(report, indent=1))
	write_report(out, report)
	print(f"\nDECISION: {report['decision']}")


if __name__ == "__main__":
	main()
