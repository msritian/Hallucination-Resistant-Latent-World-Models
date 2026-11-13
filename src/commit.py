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
from src.auditor.signals import decode, member_rollouts, pi_mean, q_heads
from src.preflight.run_preflight import CRITIC_FEATURES, _features, fit_lba, make_gbt, plan_signals

K_MAX = 8
BUDGETS = (2, 4)                                          # target mean actions per plan for the gates
SCORES = ("LBA", "Aa", "D", "M")                          # before-acting gates from the plan's per-step scores
FLIP = ("PD", "IR")                                       # decision-flip signals (docs/preregistration_decision_flip.md)
EXTRA = ("IR1", "FI", "RS")                               # frontier study (docs/preregistration_frontier.md)
FRONTIER_BUDGETS = (2, 6)
RS_ELITES = 16
GATES = ("oracle", *SCORES, "PE", "random")
DECISION = ("act", "reg")                                 # decision oracles (confirmation study, see decision_scores)
STEP = 0.05
LEVELS = [1 + STEP * i for i in range(int((K_MAX - 1) / STEP) + 1)]   # commitment levels 1.00 ... 8.00
ETA = 0.05                                               # gain of the budget controller


@dataclass(frozen=True)
class Arm:
	name: str
	k: int = 1                     # fixed commitment (when gate is None)
	gate: Optional[str] = None
	budget: int = 0                # target mean commitment of the gate
	abort: bool = False            # also stop early on the prediction-error check (combination arm)
	fast: bool = False             # timing study: compute only this gate's score (lean scorer, see ir_only)
	fast2: bool = False            # timing study 2: the batched scorer ir_fast


def arms_for(has_ensemble: bool, study: str = "main") -> list:
	if study == "frontier":  # docs/preregistration_frontier.md: budgets 2 and 6, plus baselines/ablation at 4
		arms = [Arm(f"k{k}", k=k) for k in FRONTIER_BUDGETS]
		for b in FRONTIER_BUDGETS:
			arms += [Arm(f"{g}@{b}", gate=g, budget=b) for g in ("random", "oracle", *DECISION, "IR", "LBA", "PE")]
		return arms + [Arm(f"{g}@4", gate=g, budget=4) for g in EXTRA]
	if study == "general":   # docs/preregistration_general.md: new robots (stock TD-MPC2, new tasks)
		return ([Arm(f"k{k}", k=k) for k in (1, 2, 4, 6)] + [Arm("random@4", gate="random", budget=4)]
		        + [Arm(f"IR@{b}", gate="IR", budget=b, fast=True, fast2=True) for b in (2, 4, 6)]
		        + [Arm("PE@4", gate="PE", budget=4)])
	if study == "timing2":   # docs/preregistration_wallclock2.md: the batched IR scorer
		return [Arm(f"k{k}", k=k) for k in (1, 2, 4, 6)] + [Arm(f"IR@{b}", gate="IR", budget=b, fast=True, fast2=True)
		                                                     for b in (2, 4, 6)]
	if study == "timing":    # docs/preregistration_wallclock.md: lean IR gate vs fixed chunking, wall-clock
		return [Arm(f"k{k}", k=k) for k in (1, 2, 4, 6)] + [Arm(f"IR@{b}", gate="IR", budget=b, fast=True)
		                                                     for b in (2, 4, 6)]
	if study == "confirm":   # docs/preregistration_commitment_confirm.md: budget 4 only
		gates = ("random", "oracle", *DECISION, "DBA", *SCORES, "PE", *FLIP)
		return [Arm("k1", k=1), Arm("k4", k=4)] + [Arm(f"{g}@4", gate=g, budget=4) for g in gates
		                                           if has_ensemble or g not in ("D", "M")]
	arms = [Arm(f"k{k}", k=k) for k in (1, 2, 4, 8)]
	for b in BUDGETS:
		arms += [Arm(f"{g}@{b}", gate=g, budget=b) for g in GATES if has_ensemble or g not in ("D", "M")]
	arms.append(Arm("LBA+PE@4", gate="LBA", budget=4, abort=True))
	return arms


# ------------------------------------------------------------------------------------------------ scoring one plan
@torch.no_grad()
def score_plan(agent, obs: torch.Tensor, plan: torch.Tensor, lba: dict, pool: Optional[dict] = None):
	"""Per-step scores [L] of the before-acting gates for one plan [H, A] from the real observation obs [D].

	Step j judges the imagined transition into z_{j+1}, i.e. whether action j+1 is still chosen from a trustworthy
	imagined state. Returns ({score: [L]}, rollout, features [L, F]). With a decision-aware audit in lba["dba"], also
	its score "DBA"."""
	dev = next(agent.model.parameters()).device
	H = plan.shape[0]
	z0 = agent.model.encode(obs.to(dev).view(1, -1), None)
	sig, roll = plan_signals(agent, z0, plan.to(dev).view(H, 1, -1), 1.0, bias=lba["bias"])
	sig = {k: v.cpu() for k, v in sig.items()}
	L = lba["L"]
	X = _features(sig, roll, CRITIC_FEATURES, L, True).numpy()
	steps = {"LBA": torch.as_tensor(lba["clf"].predict_proba(X)[:, 1], dtype=torch.float32)}
	if "dba" in lba:
		steps["DBA"] = torch.as_tensor(lba["dba"].predict_proba(X)[:, 1], dtype=torch.float32)
	for k in SCORES[1:]:
		if k in sig:
			steps[k] = sig[k][:L, 0].float()
	steps.update(flip_scores(agent, z0, plan.to(dev), L))
	if pool is not None:
		steps["RS"] = rank_instability(agent, z0, plan.to(dev), pool, L)
	return steps, roll, torch.as_tensor(X, dtype=torch.float32)


@torch.no_grad()
def flip_scores(agent, z0: torch.Tensor, plan: torch.Tensor, L: int) -> dict:
	"""Training-free decision-flip signals for steps j = 0..L-1, each judging plan action j+1 (taken at z_{j+1}).

	Every dynamics head imagines the plan on its own (main head first), giving plausible states z^m_{j+1}.
	  PD[j] = mean_m |pi(z^m_{j+1}) - pi(z^0_{j+1})|: would the policy act differently in the plausible states?
	          (disagreement in ACTION space; D measures it in state space)
	  IR[j] = mean over heads m and critic heads h of max(0, Q_h(z^m_{j+1}, pi(z^m_{j+1})) - Q_h(z^m_{j+1}, a_{j+1})):
	          imagined regret of the stale plan action, a label-free estimate of the reg oracle."""
	model, cfg = agent.model, agent.cfg
	H, A = plan.shape
	full = member_rollouts(agent.aux_dynamics_heads(), z0, plan.view(H, 1, A))[:, :, 0]         # [M, H+1, d]
	trajs = full[:, 1:L + 1]                                                                    # [M, L, d]
	M, _, d = trajs.shape
	zf = trajs.reshape(M * L, d)
	pa = pi_mean(model, zf).view(M, L, A)
	pd = (pa - pa[:1]).norm(dim=-1).mean(0) * M / max(M - 1, 1)                                 # mean over m >= 1
	a_plan = plan[1:L + 1].unsqueeze(0).expand(M, L, A).reshape(M * L, A)
	q_pi = q_heads(model, zf, pa.view(M * L, A), cfg)                                           # [K, M*L, 1]
	q_plan = q_heads(model, zf, a_plan, cfg)
	ir_all = (q_pi - q_plan).clamp(min=0).view(-1, M, L)
	ir, ir1 = ir_all.mean((0, 1)), ir_all[:, 0].mean(0)
	# FI (frontier study): temporal instability of the main head's imagined latent flow around z_{j+1}, the
	# transferable analogue of the flow-instability signal of Hansen & Wang (2606.27326); TD-MPC2 has no decoder.
	z = full[0]
	fi = torch.stack([(z[min(j + 2, H)] - 2 * z[j + 1] + z[j]).norm() for j in range(L)])
	return {"PD": pd.float().cpu(), "IR": ir.float().cpu(), "IR1": ir1.float().cpu(), "FI": fi.float().cpu()}


@torch.no_grad()
def ir_only(agent, obs: torch.Tensor, plan: torch.Tensor, L: int) -> torch.Tensor:
	"""IR [L] alone (timing study): the same computation as flip_scores' IR, without the other gates' scores."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	plan = plan.to(dev)
	H, A = plan.shape
	z0 = model.encode(obs.to(dev).view(1, -1), None)
	trajs = member_rollouts(agent.aux_dynamics_heads(), z0, plan.view(H, 1, A))[:, 1:L + 1, 0]  # [M, L, d]
	M, _, d = trajs.shape
	zf = trajs.reshape(M * L, d)
	pa = pi_mean(model, zf)
	a_plan = plan[1:L + 1].unsqueeze(0).expand(M, L, A).reshape(M * L, A)
	ir = (q_heads(model, zf, pa, cfg) - q_heads(model, zf, a_plan, cfg)).clamp(min=0).view(-1, M, L).mean((0, 1))
	return ir.float().cpu()


def _stack_heads(agent) -> list:
	"""Per layer of the dynamics MLPs: (W [M, in, out], b [M, 1, out], ln_w [M, 1, out], ln_b [M, 1, out], eps, act),
	stacking the main head (first) and the auxiliary heads, which share one architecture (layers.mlp). Cached."""
	if getattr(agent, "_stacked_heads", None) is None:
		mods = [agent.model._dynamics] + (list(agent.model._dyn_aux) if agent.num_aux_dynamics > 0 else [])
		out = []
		for i in range(len(mods[0])):
			ls = [m[i] for m in mods]
			out.append((torch.stack([l.weight.t() for l in ls]), torch.stack([l.bias.view(1, -1) for l in ls]),
			            torch.stack([l.ln.weight.view(1, -1) for l in ls]), torch.stack([l.ln.bias.view(1, -1) for l in ls]),
			            ls[0].ln.eps, ls[0].act))
		agent._stacked_heads = out
	return agent._stacked_heads


def heads_next(stacked: list, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
	"""All dynamics heads at once: z [M, N, d], a [M, N, A] -> [M, N, d] (each head on its own latent)."""
	x = torch.cat([z, a], -1)
	for W, b, g, beta, eps, act in stacked:
		x = torch.baddbmm(b, x, W)
		x = act(torch.nn.functional.layer_norm(x, x.shape[-1:], None, None, eps) * g + beta)
	return x


@torch.no_grad()
def ir_fast(agent, obs: torch.Tensor, plan: torch.Tensor, L: int) -> torch.Tensor:
	"""IR [L] as ir_only, faster (timing study 2): all heads in one batched call per step, only the L steps used
	(z_1 .. z_L), and one critic call for the policy and plan actions together."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	plan = plan.to(dev)
	A = plan.shape[1]
	st = _stack_heads(agent)
	M = st[0][0].shape[0]
	z = model.encode(obs.to(dev).view(1, -1), None).expand(M, 1, -1)
	zs = []
	for t in range(L):
		z = heads_next(st, z, plan[t].view(1, 1, A).expand(M, 1, A))
		zs.append(z[:, 0])
	zf = torch.stack(zs, 1).reshape(M * L, -1)                                                  # [M*L, d], m-major
	pa = pi_mean(model, zf)
	a_plan = plan[1:L + 1].unsqueeze(0).expand(M, L, A).reshape(M * L, A)
	q = q_heads(model, torch.cat([zf, zf]), torch.cat([pa, a_plan]), cfg)                       # [K, 2*M*L, 1]
	ir = (q[:, :M * L] - q[:, M * L:]).clamp(min=0).view(-1, M, L).mean((0, 1))
	return ir.float().cpu()


@torch.no_grad()
def rank_instability(agent, z0: torch.Tensor, plan: torch.Tensor, pool: dict, L: int) -> torch.Tensor:
	"""RS[j] (frontier study): would the stale plan still win from the plausible states z^m_{j+1}?

	Rank-stability baseline in the spirit of "Planners Exploit Deterministic Error" (2611.11780): the chosen plan
	plus the RS_ELITES - 1 best other candidates of the planner's final MPPI pool; from each dynamics head's
	imagined state z^m_{j+1} (of the chosen plan) every candidate's remaining actions j+1..H-1 are scored with the
	planner's own value estimate (main model). RS[j] = mean over heads of the share of candidates that beat the plan."""
	model, cfg, disc0 = agent.model, agent.cfg, float(agent.discount)
	H, A = plan.shape
	acts, val = pool["actions"], pool["value"].view(-1)
	top = torch.topk(val, min(RS_ELITES - 1, val.numel())).indices
	cand = torch.cat([plan.view(H, 1, A), acts[:, top].to(plan.device).clamp(-1, 1)], 1)      # [H, E, A]
	E = cand.shape[1]
	full = member_rollouts(agent.aux_dynamics_heads(), z0, plan.view(H, 1, A))[:, :, 0]       # [M, H+1, d]
	M = full.shape[0]
	out = []
	for j in range(L):
		z = full[:, j + 1].repeat_interleave(E, 0)                                             # [M*E, d]
		a = cand[j + 1:].repeat(1, M, 1)                                                       # [T, M*E, A]
		G, disc = 0, 1.0
		for t in range(a.shape[0]):
			G = G + disc * decode(model.reward(z, a[t], None), cfg)
			z = model.next(z, a[t], None)
			disc *= disc0
		v = (G + disc * q_heads(model, z, pi_mean(model, z), cfg).mean(0)).view(M, E)
		out.append((v[:, 1:] > v[:, :1]).float().mean())
	return torch.stack(out).cpu()


@torch.no_grad()
def decision_scores(agent, sim, planner2, prev_mean: torch.Tensor, plan: torch.Tensor) -> dict:
	"""Decision oracles: is it still right to execute plan action j blind, for j = 1 .. K_MAX-1?

	Executes plan[:j] in a simulator copy, replans from the real state reached (a second planner, warm-started exactly
	as the robot would be), and compares the fresh decision with the stale plan action:
	  act[j-1] = |a*_j - plan[j]|                    (would a fresh plan act differently?)
	  reg[j-1] = Q(z_j, a*_j) - Q(z_j, plan[j])       (critic's regret of the stale action, at the REAL state z_j)
	Unlike the return-error oracle, these ask whether an error changes the action, not whether the imagined value
	is off."""
	dev = next(agent.model.parameters()).device
	n = K_MAX - 1
	_, o = sim.rollout(sim.save(), plan[:n].view(n, 1, -1))              # obs after 1..n plan actions [n, 1, D]
	act, reg = torch.zeros(n), torch.zeros(n)
	for j in range(1, n + 1):
		pm = prev_mean.clone()
		if j > 1:                                                         # the planner itself shifts by one more
			pm[:-(j - 1)] = prev_mean[j - 1:]
			pm[-(j - 1):] = 0
		planner2.prev_mean.copy_(pm)
		a_star = planner2.act(o[j - 1, 0], t0=False)
		a_plan = plan[j].cpu()
		act[j - 1] = (a_star - a_plan).norm()
		z = agent.model.encode(o[j - 1, 0].to(dev).view(1, -1), None).repeat(2, 1)
		q = q_heads(agent.model, z, torch.stack([a_star, a_plan]).to(dev), agent.cfg).mean(0).view(-1)
		reg[j - 1] = float(q[0] - q[1])
	return dict(act=act, reg=reg)


def fit_dba(rec: dict):
	"""Decision-aware Bellman audit: the LBA's trees and features, but labels from the decision oracle (act).

	Step j (judging action j+1) is positive if act is in the top quarter, negative if in the bottom half."""
	n = K_MAX - 1
	X = rec["X"][:, :n].reshape(-1, rec["X"].shape[-1])
	y = rec["act"].reshape(-1)
	hi, lo = float(torch.quantile(y, 0.75)), float(torch.quantile(y, 0.5))
	keep = (y > hi) | (y <= lo)
	return make_gbt().fit(X[keep].numpy(), (y[keep] > hi).numpy())


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
def calibrate(agent, env, sim, lba: dict, episodes: int, seed_start: int, decision: bool = False,
              pool: bool = False) -> dict:
	"""Closed-loop k = 1 episodes; every chosen plan is also executed in the simulator copy.

	Per decision: the gate scores [L], the true return error E [L] (oracle and labels), and the latent prediction
	error PE [L] (distance of the encoded real state after j+1 plan actions from the imagined z_{j+1}).
	With decision=True also the decision oracles act, reg [K_MAX-1] and the features X [L, F] (to fit the DBA)."""
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	planner2 = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	dev = next(agent.model.parameters()).device
	discount = float(agent.discount)
	L = lba["L"]
	rec = dict(steps={}, E=[], PE=[], ep=[], t=[], act=[], reg=[], X=[])
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			plan = planner.last_plan.detach().cpu()
			steps, roll, X = score_plan(agent, obs, plan, lba, pool=planner.last_pool if pool else None)
			if decision:
				d = decision_scores(agent, sim, planner2, planner.prev_mean, plan)
				rec["act"].append(d["act"])
				rec["reg"].append(d["reg"])
				rec["X"].append(X)
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
	out = dict(steps={k: torch.stack(v) for k, v in rec["steps"].items()}, E=torch.stack(rec["E"]),
	           PE=torch.stack(rec["PE"]), ep=torch.tensor(rec["ep"]), t=torch.tensor(rec["t"]))
	if decision:
		out.update(act=torch.stack(rec["act"]), reg=torch.stack(rec["reg"]), X=torch.stack(rec["X"]))
	return out


def expected_k(c: torch.Tensor, thr: float, tie: float) -> float:
	"""Mean commitment of commit_len over plans with running-max scores c [N, K_MAX-1] (ties pass w.p. tie)."""
	p = torch.where(c < thr, 1.0, torch.where(c == thr, tie, 0.0))
	return float(1 + torch.cumprod(p, 1).sum(1).mean())


def match(c: torch.Tensor, b: float) -> dict:
	"""{thr, tie} with expected_k(c, thr, tie) = b: the smallest observed score at which trusting ties reaches b, then
	the tie probability by bisection (exact even when scores repeat, e.g. probabilities saturated at 1.0)."""
	vals = torch.unique(c)
	lo, hi = 0, len(vals) - 1
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
	return dict(thr=thr, tie=(t_lo + t_hi) / 2)


def thresholds_from(cal: dict) -> dict:
	"""{gate: {level: {thr, tie}}}: for every level on LEVELS, the threshold whose mean commitment on the calibration
	plans equals that level. Run time starts each gate at its budget and moves along these levels (BudgetControl)."""
	scores = dict(cal["steps"], oracle=cal["E"], PE=cal["PE"])
	scores.update({g: cal[g] for g in DECISION if g in cal})
	return {g: {key(b): match(torch.cummax(s[:, :K_MAX - 1].float(), 1).values, b) for b in LEVELS}
	        for g, s in scores.items()}


def key(level: float) -> str:
	return f"{min(max(round(level / STEP) * STEP, 1.0), float(K_MAX)):.2f}"


class BudgetControl:
	"""Integral controller that keeps a gate's mean commitment at its budget at run time (amendment 1).

	Calibration thresholds do not transfer exactly: score levels shift between episodes, so a fixed threshold can
	commit to far fewer or more actions than the budget, and gates would no longer use the same planning compute.
	Each gate therefore uses the calibration threshold of level = budget + ETA * sum(budget - k) over its decisions so
	far (across episodes of the arm), which drives the running mean of k to the budget. Same rule for every gate."""

	def __init__(self, budget: float):
		self.budget, self.err = float(budget), 0.0

	def level(self) -> str:
		return key(self.budget + ETA * self.err)

	def update(self, k: int):
		self.err += self.budget - k


# ------------------------------------------------------------------------------------------------ episodes
@torch.no_grad()
def run_arm(agent, env, sim, arm: Arm, lba: dict, th: dict, episodes: int, seed_start: int, seed: int = 0):
	"""Run one arm on the evaluation seeds; returns per-episode rows and the per-plan commitments."""
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	planner2 = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard")) if arm.gate in DECISION else None
	gen = torch.Generator().manual_seed(seed)
	discount = float(agent.discount)
	q = random_q(arm.budget) if arm.gate == "random" else None
	ctrl = BudgetControl(arm.budget) if arm.gate not in (None, "random") else None
	rows, rec = [], dict(ep=[], t=[], k=[], t_plan=[], t_gate=[])
	sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
	for i in range(episodes):
		obs, done, t, ep_reward, calls, j = env.reset(seed=seed_start + i), False, 0, 0.0, 0, 0
		start = time.perf_counter()
		while not done:
			sync()
			t_a = time.perf_counter()
			shift_warm_start(planner, j)
			planner.act(obs, t0=calls == 0)
			plan = planner.last_plan.detach().cpu().clamp(-1, 1)
			sync()
			t_b = time.perf_counter()
			calls += 1
			roll, pe_th = None, None
			if arm.gate is None:
				k = arm.k
			elif arm.gate == "random":
				k = 1
				while k < K_MAX and bool(torch.rand(1, generator=gen) < q):
					k += 1
			elif arm.fast:
				k = commit_len((ir_fast if arm.fast2 else ir_only)(agent, obs, plan, lba["L"]), th[arm.gate][ctrl.level()], gen)
			elif arm.gate == "PE":
				k = K_MAX
				_, roll, _ = score_plan(agent, obs, plan, lba)
				pe_th = th["PE"][ctrl.level()]
			elif arm.gate in DECISION:
				d = decision_scores(agent, sim, planner2, planner.prev_mean, plan)
				k = commit_len(d[arm.gate], th[arm.gate][ctrl.level()], gen)
			elif arm.gate == "oracle":
				steps, roll, _ = score_plan(agent, obs, plan, lba)
				r_true, _ = sim.rollout(sim.save(), plan.view(plan.shape[0], 1, -1))
				k = commit_len(plan_error(roll, r_true[:, 0], discount), th["oracle"][ctrl.level()], gen)
			else:
				steps, roll, _ = score_plan(agent, obs, plan, lba, pool=planner.last_pool if arm.gate == "RS" else None)
				k = commit_len(steps[arm.gate], th[arm.gate][ctrl.level()], gen)
				if arm.abort:
					pe_th = th["PE"][key(arm.budget)]
			sync()
			rec["t_plan"].append(t_b - t_a)
			rec["t_gate"].append(time.perf_counter() - t_b)
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
			if ctrl is not None and not done:     # a plan cut short by the episode's end says nothing about the gate
				ctrl.update(j)
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
	from src.preflight.run_preflight import collect, load_agent
	from src.tools.plan_diag import SimSnapshot

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", choices=["stock", "grounded", "grounded_noaux", "stock_aux"], required=True)
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", default=None,
	               help="episodes_<model>.pt saved by run_preflight (fits the LBA); if absent, 50 real episodes are "
	                    "collected on seeds 2000-2049 (the protocol run_preflight used)")
	p.add_argument("--study", choices=["main", "confirm", "frontier", "timing", "timing2", "general"], default="main")
	p.add_argument("--dba_episodes", type=int, default=20, help="confirm: episodes that train the decision-aware audit")
	p.add_argument("--dba_seed_start", type=int, default=6800)
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
		if args.episodes_file:
			data = torch.load(args.episodes_file)
		else:
			data = collect(agent, args.task, 50, 2000)
			torch.save(data, out / "episodes.pt")
			print(f"collected 50 real episodes, success {float(data['success'][:, -1].float().mean()):.2f}", flush=True)
		lba = fit_lba(agent, data, H_LONG, beta=1.0)
		(out / "lba.pkl").write_bytes(pickle.dumps(lba))
		(out / "lba.json").write_text(json.dumps({k: lba[k] for k in ("eps_clean", "eps_hall", "L", "n_pos")}, indent=1))
		print(f"LBA fitted on {data['obs'].shape[0]} episodes ({lba['n_pos']} positive steps)", flush=True)

	env = ManiSkill3Env(args.task, seed=args.seed_start)
	sim = SimSnapshot(env)
	decision = args.study == "confirm"
	# 0b. confirm: the decision-aware audit, trained on its own episodes (separate seeds)
	if decision and "dba" not in lba:
		tr = calibrate(agent, env, sim, lba, args.dba_episodes, args.dba_seed_start, decision=True)
		lba["dba"] = fit_dba(tr)
		torch.save(tr, out / "dba_train.pt")
		(out / "lba.pkl").write_bytes(pickle.dumps(lba))
		print(f"DBA fitted on {tr['act'].shape[0]} decisions", flush=True)
	# 1. thresholds from closed-loop k = 1 calibration episodes (separate seeds)
	if args.study == "frontier" and not (out / "thresholds_extra.json").exists():
		# reuses the confirmation job's directory (lba.pkl with the DBA, thresholds.json for every existing gate and
		# level); only the new scores IR1, FI, RS need calibrating, on the same calibration seeds
		cal = calibrate(agent, env, sim, lba, args.cal_episodes, args.cal_seed_start, pool=True)
		ex = {g: v for g, v in thresholds_from(cal).items() if g in EXTRA}
		torch.save(cal, out / "calibration_extra.pt")
		(out / "thresholds_extra.json").write_text(json.dumps(ex, indent=1))
		print(f"extra thresholds for {sorted(ex)}", flush=True)
	if (out / "thresholds.json").exists():
		th = json.loads((out / "thresholds.json").read_text())
	else:
		cal = calibrate(agent, env, sim, lba, args.cal_episodes, args.cal_seed_start, decision=decision)
		th = thresholds_from(cal)
		torch.save(cal, out / "calibration.pt")
		(out / "thresholds.json").write_text(json.dumps(th, indent=1))
		print(f"thresholds: {json.dumps(th)}", flush=True)
	if (out / "thresholds_extra.json").exists():
		th.update(json.loads((out / "thresholds_extra.json").read_text()))

	if args.study in ("timing", "timing2", "general"):   # the lean scorer must give the IR of flip_scores
		scorer = ir_only if args.study == "timing" else ir_fast
		from src.planning.audited_planner import AuditedPlanner, PlannerConfig
		pl = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
		obs, worst = env.reset(seed=args.cal_seed_start), 0.0
		for c in range(5):
			pl.act(obs, t0=c == 0)
			plan = pl.last_plan.detach().cpu().clamp(-1, 1)
			z0 = agent.model.encode(obs.to(next(agent.model.parameters()).device).view(1, -1), None)
			ref = flip_scores(agent, z0, plan.to(z0.device), lba["L"])["IR"]
			worst = max(worst, float((scorer(agent, obs, plan, lba["L"]) - ref).abs().max() / (ref.abs().max() + 1e-6)))
			obs = env.step(plan[0])[0]
		print(f"lean IR check: max relative difference = {worst:.2e}", flush=True)
		assert worst < (1e-5 if args.study == "timing" else 1e-3), "lean IR scorer disagrees with flip_scores"
		print(f"ensemble heads: {1 + agent.num_aux_dynamics}", flush=True)

	# 2. arms, same evaluation seeds
	arms = arms_for(has_ensemble, args.study)
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

	if args.study == "frontier":    # the directory also holds the confirmation arms (same robot, same seeds)
		arms = arms_for(has_ensemble, "confirm") + arms
	write_outputs(out, arms)
	print("done", flush=True)


if __name__ == "__main__":
	main()
