"""Ground-truth planning diagnostics: do imagined plans hallucinate, and does trust scoring rank them better?

The preflight replays the world model along the actions the agent actually took. The planner, however, scores MPPI
candidates (partly random actions). This tool measures the planner's own candidates against the real simulator:

1. Drive episodes with TD-MPC2's own planner. Alongside, run a standard H-step planner and a trust planner (Signal A)
   on the same observations so their warm starts evolve naturally.
2. At every `every`-th step, snapshot the simulator, take candidates from each planner's final MPPI iteration
   (top by score and random), and execute every candidate for H steps in the real simulator (restoring the snapshot).
3. Offline, compare the model's imagination with reality per candidate and step, and score every candidate with
   several scoring rules against the same target:
       target = sum_t gamma^t r_true_t + gamma^H V(encode(real obs_H))
   (what the plan is worth under the true dynamics, judged by the same critic).

Outputs (out/): records.pt (raw data, re-analysable), diag.json, report.md, taus_candidates.json (thresholds
calibrated on candidates, same format as evaluate.py's taus.json, usable via `evaluate.py --taus_file`).

Example:
	python -m src.tools.plan_diag --model run/final_model.pt --run grounded --task PickSingleYCB-v1 --out diag_ycb
"""
import argparse
import json
from pathlib import Path

import torch

from src.auditor.elvis import RunningNorm, elvis_lambdas, elvis_return
from src.auditor.trust import trust_weighted_return, trust_weights
from src.preflight.analysis import calibrate_all
from src.preflight.metrics import bootstrap_auroc
from src.preflight.run_preflight import CAL, rollout_signals, value_fn

SOURCES = ("pi", "top", "rand")
TAU_PCTS = (90, 95, 99)


# ---------------------------------------------------------------------------------------------- simulator access
class SimSnapshot:
	"""Save / restore a single-env ManiSkill3 (CPU) simulator, and execute action sequences from a snapshot."""

	def __init__(self, env):
		self.u = env.env.unwrapped

	def save(self):
		elapsed = getattr(self.u, "_elapsed_steps", None)
		state = self.u.get_state_dict() if hasattr(self.u, "get_state_dict") else self.u.get_state()
		return state, None if elapsed is None else elapsed.clone()

	def restore(self, snap):
		state, elapsed = snap
		if hasattr(self.u, "set_state_dict"):
			self.u.set_state_dict(state)
		else:
			self.u.set_state(state)
		if elapsed is not None:
			self.u._elapsed_steps = elapsed.clone()

	def rollout(self, snap, actions: torch.Tensor):
		"""actions [H, K, A] -> true rewards [H, K], observations after each step [H, K, D]; restores the snapshot."""
		H, K, _ = actions.shape
		rew, obs = torch.empty(H, K), []
		for k in range(K):
			self.restore(snap)
			ob_k = []
			for t in range(H):
				o, r, *_ = self.u.step(actions[t, k].cpu().numpy().reshape(1, -1))
				rew[t, k] = float(torch.as_tensor(r).reshape(-1)[0])
				ob_k.append(torch.as_tensor(o).reshape(-1).float().cpu())
			obs.append(torch.stack(ob_k))
		self.restore(snap)
		return rew, torch.stack(obs, 1)


def pick_candidates(pool: dict, num_pi: int, n_top: int, n_rand: int, gen: torch.Generator):
	"""Indices into the final MPPI pool: all policy trajectories, top-n by score, n random others; with source codes."""
	value = pool["value"].squeeze(-1).cpu()
	N = value.shape[0]
	pi_idx = torch.arange(num_pi)
	rest = torch.arange(num_pi, N)
	top = rest[value[rest].argsort(descending=True)[:n_top]]
	others = rest[~torch.isin(rest, top)]
	rand = others[torch.randperm(len(others), generator=gen)[:n_rand]]
	idx = torch.cat([pi_idx, top, rand])
	src = torch.cat([torch.full((len(pi_idx),), 0), torch.full((len(top),), 1), torch.full((len(rand),), 2)])
	return idx, src


@torch.no_grad()
def collect_records(agent, planners: dict, task: str, episodes: int, seed_start: int, every: int,
                    n_top: int, n_rand: int) -> list:
	from src.envs.maniskill3 import ManiSkill3Env

	env = ManiSkill3Env(task, seed=seed_start)
	sim = SimSnapshot(env)
	gen = torch.Generator().manual_seed(0)
	records, det_err = [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			diag_step = t % every == 0
			snap = sim.save() if diag_step else None
			for p in planners.values():
				p.act(obs, t0=t == 0, eval_mode=True)  # keeps each planner's warm start realistic
			action = agent.act(obs, t0=t == 0, eval_mode=True)
			if diag_step:
				acts, srcs, owner = [], [], []
				for j, p in enumerate(planners.values()):
					idx, src = pick_candidates(p.last_pool, agent.cfg.num_pi_trajs, n_top, n_rand, gen)
					acts.append(p.last_pool["actions"][:, idx.to(p.last_pool["actions"].device)].cpu())
					srcs.append(src)
					owner.append(torch.full_like(src, j))
				acts = torch.cat(acts, 1)
				rew, obs_true = sim.rollout(snap, acts)
				# determinism check: the executed action from the snapshot must reproduce the real next observation
				_, check = sim.rollout(snap, action.view(1, 1, -1))
			obs_next, _, done, info = env.step(action)
			if diag_step:
				det_err.append(float((check[0, 0] - obs_next).abs().max()))
				records.append(dict(episode=i, t=t, obs0=obs.cpu(), actions=acts, r_true=rew, obs_true=obs_true,
				                    source=torch.cat(srcs), owner=torch.cat(owner), success_before=info["success"]))
			obs = obs_next
			t += 1
		print(f"episode {i}: {len(records)} diagnostic states so far, success {info['success']:.0f}, "
		      f"max determinism error {max(det_err):.2e}", flush=True)
	env.close()
	return records, det_err


# ---------------------------------------------------------------------------------------------- analysis
def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
	rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
	rx, ry = rx - rx.mean(), ry - ry.mean()
	d = (rx.norm() * ry.norm()).item()
	return float((rx * ry).sum() / d) if d > 0 else float("nan")


def selection_quality(score: torch.Tensor, target: torch.Tensor, state: torch.Tensor, n_elite: int = 8) -> dict:
	"""Per diagnostic state: regret of the argmax, mean target of the top-n (elite quality), rank correlation."""
	regret, elite, rho = [], [], []
	for s in state.unique():
		m = state == s
		sc, tg = score[m], target[m]
		regret.append(float(tg.max() - tg[sc.argmax()]))
		elite.append(float(tg[sc.argsort(descending=True)[:n_elite]].mean() - tg.mean()))
		rho.append(spearman(sc, tg))
	mean = lambda v: float(torch.tensor(v).nanmean())
	return dict(regret=mean(regret), elite_gain=mean(elite), spearman=mean(rho), per_state_regret=regret)


def paired_diff_ci(a: list, b: list, n_boot: int = 2000) -> tuple:
	"""Mean of a - b over states with a 95% bootstrap CI (states resampled)."""
	d = torch.tensor(a) - torch.tensor(b)
	g = torch.Generator().manual_seed(0)
	boots = d[torch.randint(len(d), (n_boot, len(d)), generator=g)].mean(1)
	return float(d.mean()), float(torch.quantile(boots, 0.025)), float(torch.quantile(boots, 0.975))


def _norm_signals(sig: dict, tau_lists: dict, H: int) -> dict:
	s = {k: v / torch.as_tensor(tau_lists[k][:H], dtype=v.dtype).view(-1, 1) for k, v in sig.items() if k in tau_lists}
	if "A" in s and "B" in s:
		s["C"] = torch.maximum(s["A"], s["B"])
	return s


@torch.no_grad()
def analyze(agent, records: list, taus_onpolicy: dict, beta: float = 1.0, kappa: float = 1.0) -> dict:
	dev = next(agent.model.parameters()).device
	g = float(agent.discount)
	K_per = [r["actions"].shape[1] for r in records]
	actions = torch.cat([r["actions"] for r in records], 1).to(dev)                       # [H, M, A]
	H, M = actions.shape[:2]
	obs0 = torch.cat([r["obs0"].view(1, -1).expand(k, -1) for r, k in zip(records, K_per)])
	obs_true = torch.cat([r["obs_true"] for r in records], 1)                             # [H, M, D]
	r_true = torch.cat([r["r_true"] for r in records], 1)                                 # [H, M]
	source = torch.cat([r["source"] for r in records])
	state = torch.cat([torch.full((k,), i) for i, k in enumerate(K_per)])
	half = state >= len(records) // 2                                                    # evaluation half

	enc = lambda o: agent.model.encode(o.to(dev), None)
	z_real = torch.cat([enc(obs0)[None], enc(obs_true)])                                  # [H+1, M, d]
	sig, lat_err, roll = rollout_signals(agent, z_real, actions, beta)
	sig, lat_err = {k: v.cpu() for k, v in sig.items()}, lat_err.cpu()
	v_real = value_fn(agent, z_real).cpu()                                                # [H+1, M]
	v_hat = roll.v_mean.squeeze(-1).cpu()                                                 # [H+1, M]
	r_hat = roll.r_hat.squeeze(-1).cpu()                                                  # [H, M]
	val_err = (v_hat[1:] - v_real[1:]).abs()
	disc = g ** torch.arange(H, dtype=torch.float32).view(H, 1)
	target = (disc * r_true).sum(0) + g ** H * v_real[H]

	out = dict(num_states=len(records), num_candidates=M, horizon=H)

	# 1. Does imagination drift from reality? Per step and candidate source.
	per_step = {}
	for si, name in enumerate(SOURCES):
		m = source == si
		if m.any():
			per_step[name] = dict(
				reward_bias=(r_hat - r_true)[:, m].mean(1).tolist(),
				reward_abs_err=(r_hat - r_true)[:, m].abs().mean(1).tolist(),
				latent_err=lat_err[:, m].mean(1).tolist(),
				value_err=val_err[:, m].mean(1).tolist(),
				value_bias=(v_hat[1:] - v_real[1:])[:, m].mean(1).tolist())
	out["per_step"] = per_step

	# 2. Is the critic biased on candidate actions? Q(z0, a0) vs its own one-step backup on the REAL next state.
	q0 = roll.q_sa[0].squeeze(-1).cpu()
	backup = r_true[0] + g * v_real[1]
	out["critic_bias_step0"] = {name: dict(mean=float((q0 - backup)[source == si].mean()),
	                                      mean_abs=float((q0 - backup)[source == si].abs().mean()))
	                            for si, name in enumerate(SOURCES) if (source == si).any()}

	# 3. Ranking quality by horizon (standard score; imagined vs "perfect model" with the same critic).
	by_h = {}
	for h in sorted({1, 2, 3, 6, 9, H} & set(range(1, H + 1))):
		std_h = (disc[:h] * r_hat[:h]).sum(0) + g ** h * v_hat[h]
		orc_h = (disc[:h] * r_true[:h]).sum(0) + g ** h * v_real[h]
		by_h[h] = dict(standard=selection_quality(std_h, target, state), perfect_model=selection_quality(orc_h, target, state))
	out["ranking_by_horizon"] = by_h

	# 4. Candidate-calibrated thresholds (first half of states) and detection on candidates (second half).
	cal = ~half
	taus_c, eps = calibrate_all({k: v[:, cal] for k, v in sig.items()}, lat_err[:, cal], CAL)
	taus_cand = {}
	for pct in TAU_PCTS:
		t_pct, _ = calibrate_all({k: v[:, cal] for k, v in sig.items()}, lat_err[:, cal], dict(CAL, tau_pct=pct))
		taus_cand[pct] = {k: t.tau.tolist() for k, t in t_pct.items()}
	out["taus_candidates"] = taus_cand
	out["taus_onpolicy_vs_candidates_A"] = dict(onpolicy=taus_onpolicy[95]["A"][:H], candidates=taus_cand[95]["A"])
	v1 = val_err[0, cal]
	eps_v_clean, eps_v_hall = torch.quantile(v1, 0.9).item(), torch.quantile(v1, 0.99).item()
	pos, neg = val_err > eps_v_hall, val_err <= eps_v_clean
	keep = (pos | neg) & half.view(1, -1)
	steps = torch.arange(H).view(-1, 1).expand(H, M)
	grp = state.view(1, -1).expand(H, M)
	det = {}
	s_cand = _norm_signals(sig, taus_cand[95], H)
	for name, x in s_cand.items():
		a, lo, hi = bootstrap_auroc(x[keep], pos[keep], n_boot=300, groups=grp[keep], strata=steps[keep])
		x_cum = torch.cummax(x, 0).values
		ac, _, _ = bootstrap_auroc(x_cum[keep], pos[keep], n_boot=0, groups=grp[keep], strata=steps[keep])
		det[name] = dict(auroc=a, ci95=[lo, hi], auroc_cum=ac)
	out["detection_on_candidates"] = dict(signals=det, n_pos=int(pos[keep].sum()), n_neg=int(neg[keep].sum()))

	# 5. Scoring rules on the evaluation half: standard vs trust (signal x thresholds x fallback) vs controls.
	ev = half
	sel = lambda score: selection_quality(score[ev], target[ev], state[ev])
	std_H = (disc * r_hat).sum(0) + g ** H * v_hat[H]
	q_sa, q_min, v_t = roll.q_sa.squeeze(-1).cpu(), roll.q_sa_heads.min(0).values.squeeze(-1).cpu(), v_hat[:-1]
	fallbacks = dict(q=q_sa, qmin=q_min, v=v_t)
	unsq = lambda x: x.unsqueeze(-1)

	def trust_score(w, fb):
		score, omega = trust_weighted_return(unsq(r_hat), unsq(fallbacks[fb]), unsq(v_hat[H]), unsq(w), g)
		return score.squeeze(-1), float(omega.sum(0)[ev].mean())

	rules = {"standard_H": dict(**sel(std_H), h_eff=float(H))}
	h3 = min(3, H)
	rules["standard_H3"] = dict(**sel((disc[:h3] * r_hat[:h3]).sum(0) + g ** h3 * v_hat[h3]), h_eff=float(h3))
	# Upper bound: trust from the TRUE value error (perfect detector). If this does not beat standard, no detector can.
	w_oracle = (val_err <= eps_v_hall).float()
	for fb in fallbacks:
		sc, he = trust_score(w_oracle, fb)
		rules[f"oracle_trust_fb{fb}"] = dict(**sel(sc), h_eff=he)
	s_on = _norm_signals(sig, taus_onpolicy[95], H)
	gen = torch.Generator().manual_seed(0)
	for tau_name, s_all in [("onpolicy", s_on), ("candidates", s_cand)]:
		for name, x in s_all.items():
			w = trust_weights(x, kappa)
			for fb in fallbacks:
				sc, he = trust_score(w, fb)
				rules[f"trust_{name}_{tau_name}_fb{fb}"] = dict(**sel(sc), h_eff=he)
			if name == "A":
				sc, he = trust_score(w[:, torch.randperm(M, generator=gen)], "q")
				rules[f"trust_A_{tau_name}_fbq_shuffled"] = dict(**sel(sc), h_eff=he)
	# ELVIS-style lambda-returns (lambda from UCB, from our residual, or constant); normalizer fitted on all candidates.
	v_mean3, r3 = roll.v_mean.cpu(), roll.r_hat.cpu()
	for lab, x in [("ucb", roll.ucb(beta).cpu()), ("A", roll.delta.cpu())]:
		norm = RunningNorm()
		norm.update(x)
		lam = elvis_lambdas(x, norm, 0.0, 1.0)
		rules[f"elvis_{lab}"] = dict(**sel(elvis_return(r3, v_mean3, lam, g).squeeze(-1)), h_eff=float("nan"))
	for c in (0.5, 0.8):
		lam = torch.full_like(v_mean3, c)
		rules[f"elvis_const{c}"] = dict(**sel(elvis_return(r3, v_mean3, lam, g).squeeze(-1)), h_eff=float("nan"))
	base = rules["standard_H"]["per_state_regret"]
	for v in rules.values():
		v["regret_vs_standard"] = paired_diff_ci(v["per_state_regret"], base)
	out["scoring_rules"] = rules
	return out


def write_report(out: Path, d: dict):
	L = ["# Planning diagnostics (ground truth from the simulator)", "",
	     f"{d['num_states']} states, {d['num_candidates']} candidates, horizon {d['horizon']}. "
	     f"Determinism check max error: {d.get('determinism_max_err', float('nan')):.2e}", "",
	     "## 1. Imagination vs reality per step (mean over candidates)", ""]
	for src, v in d["per_step"].items():
		L += [f"**{src}** candidates", "", "| step | reward bias | reward |err| | latent err | value err | value bias |",
		      "| :-: | :-: | :-: | :-: | :-: | :-: |"]
		for t in range(len(v["reward_bias"])):
			L.append(f"| {t} | {v['reward_bias'][t]:+.3f} | {v['reward_abs_err'][t]:.3f} | {v['latent_err'][t]:.3f} | "
			         f"{v['value_err'][t]:.3f} | {v['value_bias'][t]:+.3f} |")
		L.append("")
	L += ["## 2. Critic bias on candidate actions: Q(z0, a0) - (r_true + gamma V(real z1))", ""]
	L += [f"- {k}: mean {v['mean']:+.3f}, mean |.| {v['mean_abs']:.3f}" for k, v in d["critic_bias_step0"].items()]
	L += ["", "## 3. Ranking quality by horizon (standard score)", "",
	      "regret = target(best) - target(chosen); elite gain = mean target of top-8 minus mean target.", "",
	      "| h | regret (model) | regret (perfect model) | elite gain (model) | spearman (model) |", "| :-: | :-: | :-: | :-: | :-: |"]
	for h, v in d["ranking_by_horizon"].items():
		L.append(f"| {h} | {v['standard']['regret']:.3f} | {v['perfect_model']['regret']:.3f} | "
		         f"{v['standard']['elite_gain']:.3f} | {v['standard']['spearman']:.3f} |")
	det = d["detection_on_candidates"]
	L += ["", f"## 4. Detection on candidates (value-error labels, step-stratified; {det['n_pos']} pos / {det['n_neg']} neg)", "",
	      "| signal | AUROC | 95% CI | AUROC (running max) |", "| :-: | :-: | :-: | :-: |"]
	for k, v in det["signals"].items():
		L.append(f"| {k} | {v['auroc']:.3f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] | {v['auroc_cum']:.3f} |")
	tv = d["taus_onpolicy_vs_candidates_A"]
	L += ["", "tau_A(t), on-policy vs candidates: " + ", ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(tv["onpolicy"], tv["candidates"])),
	      "", "## 5. Scoring rules (evaluation half; lower regret is better)", "",
	      "Δ regret = rule minus standard_H, paired over states, with a 95% bootstrap CI (negative = better).", "",
	      "| rule | regret | Δ regret vs standard [95% CI] | elite gain | spearman | H_eff |", "| :- | :-: | :-: | :-: | :-: | :-: |"]
	for k, v in sorted(d["scoring_rules"].items(), key=lambda kv: kv[1]["regret"]):
		m, lo, hi = v.get("regret_vs_standard", (float("nan"),) * 3)
		L.append(f"| {k} | {v['regret']:.3f} | {m:+.3f} [{lo:+.2f}, {hi:+.2f}] | {v['elite_gain']:.3f} | {v['spearman']:.3f} | {v['h_eff']:.1f} |")
	(out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None):
	from src.evaluate import calibrate_taus
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	from src.preflight.run_preflight import collect, load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", choices=["stock", "grounded", "grounded_noaux", "stock_aux"], required=True)
	p.add_argument("--task", required=True)
	p.add_argument("--out", required=True)
	p.add_argument("--horizon", type=int, default=12)
	p.add_argument("--episodes", type=int, default=10)
	p.add_argument("--seed_start", type=int, default=4000)
	p.add_argument("--every", type=int, default=5)
	p.add_argument("--n_top", type=int, default=24)
	p.add_argument("--n_rand", type=int, default=24)
	p.add_argument("--cal_episodes", type=int, default=20)
	p.add_argument("--cal_seed_start", type=int, default=2000)
	p.add_argument("--taus_file", default=None, help="on-policy taus.json from evaluate.py (skips calibration)")
	p.add_argument("--records", default=None, help="re-analyse an existing records.pt instead of collecting")
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	agent = load_agent(args.model, args.run, args.task)
	if args.taus_file:
		taus = {int(k) if k.isdigit() else k: v for k, v in json.loads(Path(args.taus_file).read_text()).items()}
	else:
		cal = collect(agent, args.task, args.cal_episodes, args.cal_seed_start)
		taus = calibrate_taus(agent, cal, max(args.horizon, 24), beta=1.0)
		(out / "taus_onpolicy.json").write_text(json.dumps(taus, indent=1))

	if args.records:
		saved = torch.load(args.records)
		records, det_err = saved["records"], saved["determinism_err"]
	else:
		planners = {
			"standard": AuditedPlanner(agent, PlannerConfig(horizon=args.horizon, score="standard")),
			"trustA": AuditedPlanner(agent, PlannerConfig(horizon=args.horizon, score="trust", signal="A"), taus[95]),
		}
		records, det_err = collect_records(agent, planners, args.task, args.episodes, args.seed_start, args.every,
		                                   args.n_top, args.n_rand)
		torch.save(dict(records=records, determinism_err=det_err, args=vars(args)), out / "records.pt")

	d = analyze(agent, records, taus)
	d["determinism_max_err"] = max(det_err) if det_err else float("nan")
	(out / "taus_candidates.json").write_text(json.dumps(d["taus_candidates"], indent=1))
	(out / "diag.json").write_text(json.dumps(d, indent=1))
	write_report(out, d)
	print((out / "report.md").read_text())


if __name__ == "__main__":
	main()
