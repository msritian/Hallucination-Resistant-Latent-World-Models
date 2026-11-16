"""Robust imagined regret: is the gap between IR and the critic-regret oracle (reg) the robot not knowing where it
will really be, and can a horizon-faithful error estimate (the learned Bellman audit) close it?

IR judges plan action j+1 at the imagined states z^m_{j+1} of the dynamics heads. reg judges it at the REAL state.
If the imagined state is wrong by up to r, a first-order bound on the regret at the real state is

    RIR[j] = mean over heads m, critic heads h of max(0, g_mh + r_j * ||P grad_z g_mh||),
    g_mh   = Q_h(z^m_{j+1}, pi(z^m_{j+1})) - Q_h(z^m_{j+1}, a_{j+1}),

where P projects onto the SimNorm simplex groups (the latent stays a product of simplices). r_j = 0 gives IR exactly.
"Errors that change the decision" = sensitivity of the decision to the state x the size of the state error.

Gates (all before acting except the oracles, which use a simulator copy):
  IRs   oracle: IR at the encoded REAL states (decomposes reg - IR into a state part and an action-proposal part)
  RIRo  oracle radius: r_j = the true latent error at step j (ceiling of the robust idea)
  RIR   r_j = conditional 80% quantile of the latent error given the learned Bellman audit's score (LBA)
  RIRt  r_j = the same, given the spread of the independently rolled dynamics heads (trajectory ensemble, Dt)
  RIRc  r_j = the 80% quantile of the latent error at horizon index j (horizon only, no detector)
Radius maps and thresholds come from the k = 1 calibration episodes (same seeds as every earlier study).
"""
import torch

from src.auditor.signals import member_rollouts, pi_mean, q_heads

ROBUST = ("IRs", "RIRo", "RIR", "RIRt", "RIRc")
ORACLE = ("IRs", "RIRo")
BINS = 10
Q = 0.8


def _project(g: torch.Tensor, group: int) -> torch.Tensor:
	"""Tangent projection onto the product of simplices: remove each SimNorm group's mean."""
	s = g.shape
	g = g.view(*s[:-1], -1, group)
	return (g - g.mean(-1, keepdim=True)).view(s)


def parts(agent, obs: torch.Tensor, plan: torch.Tensor, L: int) -> dict:
	"""Per plan: g [K, M, L], gradient norms s [K, M, L], trajectory spread Dt [L], main-head states z0 [L, d]."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	plan = plan.to(dev)
	H, A = plan.shape
	with torch.no_grad():
		z_init = model.encode(obs.to(dev).view(1, -1), None)
		trajs = member_rollouts(agent.aux_dynamics_heads(), z_init, plan.view(H, 1, A))[:, 1:L + 1, 0]   # [M, L, d]
	M, _, d = trajs.shape
	dt = trajs.std(0).norm(dim=-1) if M > 1 else torch.zeros(L, device=dev)
	g, s = _g_and_s(model, cfg, trajs.reshape(M * L, d), plan[1:L + 1].unsqueeze(0).expand(M, L, A).reshape(M * L, A))
	return dict(g=g.view(-1, M, L).cpu(), s=s.view(-1, M, L).cpu(), Dt=dt.cpu(), z0=trajs[0].detach())


def _g_and_s(model, cfg, z: torch.Tensor, a_plan: torch.Tensor):
	"""Per critic head h: g_h(z) = Q_h(z, pi(z)) - Q_h(z, a_plan) and ||P grad_z g_h(z)|| for each row of z."""
	z = z.detach().clone().requires_grad_(True)
	with torch.enable_grad():
		g = q_heads(model, z, pi_mean(model, z), cfg) - q_heads(model, z, a_plan, cfg)             # [K, N, 1]
		g = g[..., 0]
		norms = []
		for h in range(g.shape[0]):
			(gr,) = torch.autograd.grad(g[h].sum(), z, retain_graph=h < g.shape[0] - 1)
			norms.append(_project(gr, cfg.simnorm_dim).norm(dim=-1))
	return g.detach(), torch.stack(norms).detach()


def bound(p: dict, r: torch.Tensor) -> torch.Tensor:
	"""RIR [L] for radii r [L]; r = 0 gives IR."""
	return (p["g"] + r.view(1, 1, -1) * p["s"]).clamp(min=0).mean((0, 1))


@torch.no_grad()
def true_parts(agent, sim, plan: torch.Tensor, z0: torch.Tensor, L: int) -> dict:
	"""From a simulator copy: true latent error PE [L] of the main head and IR at the encoded real states IRs [L]."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	_, o = sim.rollout(sim.save(), plan.view(plan.shape[0], 1, -1))
	z_true = model.encode(o[:L, 0].to(dev), None)
	pe = (z_true - z0).norm(dim=-1)
	a_plan = plan[1:L + 1].to(dev)
	irs = (q_heads(model, z_true, pi_mean(model, z_true), cfg) - q_heads(model, z_true, a_plan, cfg)).clamp(min=0)
	return dict(PE=pe.cpu(), IRs=irs.mean(0)[..., 0].cpu())


@torch.no_grad()
def live_ir(agent, obs: torch.Tensor, a: torch.Tensor) -> float:
	"""Live IR: IR of plan action a at the robot's CURRENT real observation (encoder + critic only, no simulator).
	Under the sequential commitment rule this reproduces the IRs oracle while acting (results_robust_regret.md)."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	z = model.encode(obs.to(dev).view(1, -1), None)
	a = a.to(dev).view(1, -1)
	g = (q_heads(model, z, pi_mean(model, z), cfg) - q_heads(model, z, a, cfg)).clamp(min=0)
	return float(g.mean())


@torch.no_grad()
def irs_only(agent, sim, plan: torch.Tensor, L: int) -> torch.Tensor:
	"""IRs [L] of one plan from a simulator copy (calibration only)."""
	model, cfg = agent.model, agent.cfg
	dev = next(model.parameters()).device
	_, o = sim.rollout(sim.save(), plan.view(plan.shape[0], 1, -1))
	z = model.encode(o[:L, 0].to(dev), None)
	a = plan[1:L + 1].to(dev)
	return (q_heads(model, z, pi_mean(model, z), cfg) - q_heads(model, z, a, cfg)).clamp(min=0).mean(0)[..., 0].cpu()


def calibrate_live(agent, env, sim, L: int, episodes: int, seed_start: int) -> torch.Tensor:
	"""IRs scores [N, L] on closed-loop k = 1 calibration episodes (the protocol of src.commit.calibrate)."""
	from src.monitor import H_LONG
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	S = []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			S.append(irs_only(agent, sim, planner.last_plan.detach().cpu(), L))
			obs, _, done, _ = env.step(action)
			t += 1
	return torch.stack(S)


def fit_radius(score: torch.Tensor, pe: torch.Tensor) -> dict:
	"""Conditional Q-quantile of the latent error given a score: BINS quantile bins of the pooled score [N, L]."""
	x, y = score.reshape(-1).float(), pe.reshape(-1).float()
	edges = torch.quantile(x, torch.linspace(0, 1, BINS + 1)[1:-1]).tolist()
	b = torch.bucketize(x, torch.tensor(edges))
	r = [float(torch.quantile(y[b == i], Q)) if (b == i).any() else float(torch.quantile(y, Q)) for i in range(BINS)]
	return dict(edges=edges, r=r)


def radius(m: dict, score: torch.Tensor) -> torch.Tensor:
	return torch.tensor(m["r"])[torch.bucketize(score.float(), torch.tensor(m["edges"]))]


def gate_score(gate: str, p: dict, rmaps: dict, lba_steps=None, tp=None) -> torch.Tensor:
	if gate == "IRs":
		return tp["IRs"]
	if gate == "RIRo":
		return bound(p, tp["PE"])
	if gate == "RIR":
		return bound(p, radius(rmaps["LBA"], lba_steps))
	if gate == "RIRt":
		return bound(p, radius(rmaps["Dt"], p["Dt"]))
	if gate == "RIRc":
		return bound(p, torch.tensor(rmaps["c"]))
	raise ValueError(gate)


def score(agent, sim, obs, plan, lba: dict, rmaps: dict, gate: str) -> torch.Tensor:
	"""The gate's per-step score [L] for one plan at run time."""
	from src.commit import score_plan
	L = lba["L"]
	p = parts(agent, obs, plan, L)
	tp = true_parts(agent, sim, plan, p["z0"], L) if gate in ORACLE else None
	lba_steps = score_plan(agent, obs, plan, lba)[0]["LBA"] if gate == "RIR" else None
	return gate_score(gate, p, rmaps, lba_steps, tp)


def calibrate(agent, env, sim, lba: dict, episodes: int, seed_start: int):
	"""Closed-loop k = 1 episodes (the calibration protocol of src.commit.calibrate): every plan's parts, LBA scores,
	true latent errors and IRs. Returns (radius maps, {gate: scores [N, L]}, raw record)."""
	from src.commit import decision_scores, score_plan
	from src.monitor import H_LONG
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	planner2 = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	L = lba["L"]
	P, LB, TP, REG = [], [], [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			plan = planner.last_plan.detach().cpu()
			p = parts(agent, obs, plan, L)
			TP.append(true_parts(agent, sim, plan, p["z0"], L))
			p.pop("z0")
			P.append(p)
			LB.append(score_plan(agent, obs, plan, lba)[0]["LBA"])
			REG.append(decision_scores(agent, sim, planner2, planner.prev_mean, plan)["reg"])   # offline check only
			obs, _, done, _ = env.step(action)
			t += 1
	pe = torch.stack([tp["PE"] for tp in TP])
	lb = torch.stack(LB)
	dt = torch.stack([p["Dt"] for p in P])
	rmaps = dict(LBA=fit_radius(lb, pe), Dt=fit_radius(dt, pe),
	             c=[float(torch.quantile(pe[:, j], Q)) for j in range(L)])
	scores = {g: torch.stack([gate_score(g, p, rmaps, lb[n], TP[n]) for n, p in enumerate(P)]) for g in ROBUST}
	scores["IR0"] = torch.stack([bound(p, torch.zeros(L)) for p in P])                 # must equal IR (check)
	raw = dict(reg=torch.stack(REG), PE=pe, LBA=lb, Dt=dt, IRs=torch.stack([tp["IRs"] for tp in TP]),
	           g=torch.stack([p["g"] for p in P]), s=torch.stack([p["s"] for p in P]))
	return rmaps, scores, raw
