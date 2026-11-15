"""Commitment study on DeepMind Control with a RELEASED TD-MPC2 checkpoint (docs/preregistration_dmc_commit.md).

Same rules as src/commit.py (ManiSkill): MPPI with a 12-step horizon plans from the real observation; the robot executes
k plan actions blind, then replans. Gates choose k before acting from the plan's per-step scores, with thresholds from
k = 1 calibration episodes and the same budget controller, so every gate uses the same planning compute.
Score = episode return / 1000 (DMC returns lie in [0, 1000]).

Released checkpoints have one dynamics head, so every signal here is single-head:
  IR   imagined regret: mean over critic heads h of max(0, Q_h(z_{j+1}, pi(z_{j+1})) - Q_h(z_{j+1}, a_{j+1}))
  ENT  policy uncertainty at the imagined state: mean log-std of pi(z_{j+1}) (the action-entropy signal of adaptive
       action chunking for VLA policies)
  B    critic disagreement: std over critic heads of Q_h(z_{j+1}, a_{j+1})
  FI   latent-flow instability |z_{j+2} - 2 z_{j+1} + z_j|
  RS   rank instability: share of the 15 best other MPPI candidates whose remaining actions, scored from z_{j+1}, beat
       the plan's
  PE   prediction error while acting: |enc(real obs after j+1 actions) - z_{j+1}|
Oracles (simulator copy): oracle (return error of the plan), act (|fresh action - plan action| after replanning from
the real state), reg (critic regret of the stale action at the real state).

Run with Hydra overrides, e.g.
	python -m src.commit_dmc task=mt30 model_size=48 checkpoint=/path/mt30-48M.pt +out=dcm +tasks_subset=walker-walk
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TDMPC2_DIR = str(Path(__file__).resolve().parent.parent / "third_party" / "tdmpc2" / "tdmpc2")
sys.path.insert(0, TDMPC2_DIR)

import hydra  # noqa: E402

from src.auditor.signals import decode, pi_mean, q_heads  # noqa: E402
from src.commit import K_MAX, LEVELS, BudgetControl, commit_len, key, match, passes, random_q  # noqa: E402

H = 12
L = K_MAX - 1
GATES = ("IR", "ENT", "B", "FI", "RS")
ARMS = [("k1", None, 1), ("k2", None, 2), ("k4", None, 4), ("random@2", "random", 2), ("random@4", "random", 4),
        ("IR@2", "IR", 2), ("IR@4", "IR", 4), ("oracle@4", "oracle", 4), ("act@4", "act", 4), ("reg@4", "reg", 4),
        ("ENT@4", "ENT", 4), ("B@4", "B", 4), ("FI@4", "FI", 4), ("RS@4", "RS", 4), ("PE@4", "PE", 4)]
# disturbance study (docs/preregistration_dmc_push.md): budget 6, with the bound gate BND
ARMS_PUSH = [("k1", None, 1), ("k6", None, 6), ("random@6", "random", 6), ("IR@6", "IR", 6), ("PE@6", "PE", 6),
             ("BND@6", "BND", 6), ("reg@6", "reg", 6)]
RS_ELITES = 16
PUSH_P, PUSH_DUR = 0.02, 5                 # disturbance study: push start probability per step, duration in steps


# ------------------------------------------------------------------------------------------------ simulator copy
class DMCSim:
	"""Save / restore the current task's dm_control physics (and step counters) inside TD-MPC2's env wrappers."""

	def __init__(self, env):
		self.top = env
		self.sched = None                        # disturbance study: [T, 3] force on the root body per env step
		e = env
		while not hasattr(e, "envs"):
			e = e.env
		self.mt = e

	def _parts(self):
		timeout = self.mt._env                   # TensorWrapper(Timeout(DMControlWrapper(action_scale(control.Env))))
		while not hasattr(timeout, "_max_episode_steps"):
			timeout = timeout.env
		ctrl = timeout.env.env                   # DMControlWrapper.env = action_scale wrapper
		while True:                              # unwrap fully: wrappers delegate reads (__getattr__) but not writes,
			try:                                 # and keep the wrapped env in a slot (not in vars())
				ctrl = object.__getattribute__(ctrl, "_env")
			except AttributeError:
				break
		return timeout, ctrl

	def reseed(self, seed: int):
		_, ctrl = self._parts()
		ctrl.task._random = np.random.RandomState(seed)

	def save(self):
		timeout, ctrl = self._parts()
		return ctrl.physics.get_state().copy(), timeout._t, ctrl._step_count, ctrl._reset_next_step

	def restore(self, snap):
		timeout, ctrl = self._parts()
		state, t, sc, rn = snap
		with ctrl.physics.reset_context():
			ctrl.physics.set_state(state)
		timeout._t, ctrl._step_count, ctrl._reset_next_step = t, sc, rn

	def remaining(self) -> int:
		timeout, _ = self._parts()
		return timeout.max_episode_steps - timeout._t

	def set_pushes(self, seed: int, alpha: float, p: float = PUSH_P, dur: int = PUSH_DUR):
		"""Seeded push schedule for one episode: at each env step a push starts with probability p and lasts dur steps;
		force = alpha * (robot weight) in a random 3D direction, applied to body 1 (the root). alpha = 0: no pushes."""
		if alpha <= 0:
			self.sched = None
			return
		timeout, ctrl = self._parts()
		T = timeout.max_episode_steps
		rs = np.random.RandomState(seed + 104729)
		w = alpha * float(ctrl.physics.model.body_mass.sum()) * 9.81
		f = np.zeros((T, 3))
		for t in np.nonzero(rs.rand(T) < p)[0]:
			u = rs.randn(3)
			f[t:t + dur] = w * u / np.linalg.norm(u)
		self.sched = f

	def step(self, a):
		"""env.step with the scheduled push for the current step (none outside the disturbance study)."""
		if self.sched is not None:
			timeout, ctrl = self._parts()
			xf = ctrl.physics.data.xfrc_applied
			xf[:] = 0
			xf[1, :3] = self.sched[min(timeout._t, len(self.sched) - 1)]
		return self.top.step(a)

	def rollout(self, actions: torch.Tensor):
		"""Execute actions [n, A] (n <= remaining steps) from the current state; returns rewards [n], obs [n, D]; restores."""
		snap = self.save()
		rew, obs = [], []
		for a in actions:
			o, r, _, _ = self.step(a.cpu())
			rew.append(float(r))
			obs.append(torch.as_tensor(o).float().cpu())
		self.restore(snap)
		return torch.tensor(rew), (torch.stack(obs) if obs else torch.zeros(0))


# ------------------------------------------------------------------------------------------------ planner
class Planner:
	"""TD-MPC2's MPPI (tdmpc2.py _plan, eval mode) with a 12-step horizon; keeps the chosen plan and the final pool."""

	def __init__(self, agent, ti: int, disc: float):
		self.agent, self.model, self.cfg = agent, agent.model, agent.cfg
		self.dev = next(agent.model.parameters()).device
		self.task = torch.tensor([ti], device=self.dev)
		self.disc = disc
		self.prev_mean = torch.zeros(H, self.cfg.action_dim, device=self.dev)

	@torch.no_grad()
	def value(self, z, actions):
		G, d = 0, 1.0
		tk = self.task.expand(z.shape[0])
		for t in range(actions.shape[0]):
			G = G + d * decode(self.model.reward(z, actions[t], tk), self.cfg)
			z = self.model.next(z, actions[t], tk)
			d *= self.disc
		a, _ = self.model.pi(z, tk)
		return G + d * self.model.Q(z, a, tk, return_type="avg")

	@torch.no_grad()
	def act(self, obs, t0=False):
		cfg, A, mask = self.cfg, self.cfg.action_dim, self.model._action_masks[self.task]
		z = self.model.encode(obs.to(self.dev).view(1, -1), self.task)
		pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=self.dev)
		_z = z.repeat(cfg.num_pi_trajs, 1)
		tk = self.task.expand(cfg.num_pi_trajs)
		for t in range(H - 1):
			pi_actions[t], _ = self.model.pi(_z, tk)
			_z = self.model.next(_z, pi_actions[t], tk)
		pi_actions[-1], _ = self.model.pi(_z, tk)
		z = z.repeat(cfg.num_samples, 1)
		mean = torch.zeros(H, A, device=self.dev)
		std = torch.full((H, A), cfg.max_std, dtype=torch.float, device=self.dev)
		if not t0:
			mean[:-1] = self.prev_mean[1:]
		actions = torch.empty(H, cfg.num_samples, A, device=self.dev)
		actions[:, :cfg.num_pi_trajs] = pi_actions
		for _ in range(cfg.iterations):
			r = torch.randn(H, cfg.num_samples - cfg.num_pi_trajs, A, device=self.dev)
			actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
			actions = actions * mask
			value = self.value(z, actions).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			score = torch.exp(cfg.temperature * (elite_value - elite_value.max(0).values))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(cfg.min_std, cfg.max_std) * mask
			mean = mean * mask
		idx = torch.multinomial(score.squeeze(1), 1)
		plan = elite_actions[:, idx].squeeze(1).clamp(-1, 1)
		self.prev_mean.copy_(mean)
		self.last_plan, self.last_pool = plan, dict(actions=actions.clone(), value=value.clone())
		return plan

	def shift(self, j: int):
		if j > 1:
			s = j - 1
			self.prev_mean[:-s] = self.prev_mean[s:].clone()
			self.prev_mean[-s:] = 0


# ------------------------------------------------------------------------------------------------ scores of one plan
@torch.no_grad()
def plan_scores(pl: Planner, obs, plan, rs: bool = True) -> tuple:
	"""Per-step scores [L] of the before-acting gates, and the imagined latents z_0..z_H and rewards."""
	m, cfg, tk = pl.model, pl.cfg, pl.task
	z = [m.encode(obs.to(pl.dev).view(1, -1), tk)]
	r_hat = []
	for t in range(H):
		r_hat.append(float(decode(m.reward(z[-1], plan[t].view(1, -1), tk), cfg)))
		z.append(m.next(z[-1], plan[t].view(1, -1), tk))
	Z = torch.cat(z)                                                        # [H+1, d]
	zf = Z[1:L + 1]
	tl = tk.expand(L)
	_, info = m.pi(zf, tl)
	pa = info["mean"]
	qa = q_heads(m, zf, pa, cfg, tl)                                        # [K, L, 1]
	qp = q_heads(m, zf, plan[1:L + 1], cfg, tl)
	ndim = pl.model._action_masks[pl.task].sum()
	s = dict(IR=(qa - qp).clamp(min=0).mean(0).view(-1),
	         ENT=(info["log_std"].sum(-1) / ndim).view(-1),
	         B=qp.std(0).view(-1),
	         FI=torch.stack([(Z[min(j + 2, H)] - 2 * Z[j + 1] + Z[j]).norm() for j in range(L)]))
	if rs:
		acts, val = pl.last_pool["actions"], pl.last_pool["value"].view(-1)
		top = torch.topk(val, RS_ELITES - 1).indices
		cand = torch.cat([plan.view(H, 1, -1), acts[:, top]], 1)
		out = []
		for j in range(L):
			v = pl.value(Z[j + 1:j + 2].repeat(cand.shape[1], 1), cand[j + 1:]).view(-1)
			out.append((v[1:] > v[0]).float().mean())
		s["RS"] = torch.stack(out)
	return {k: v.float().cpu() for k, v in s.items()}, Z, torch.tensor(r_hat)


@torch.no_grad()
def oracles(pl: Planner, pl2: Planner, sim: DMCSim, obs, plan, r_hat, Z, decision: bool) -> dict:
	"""Return error E [L], prediction error PE [L] and (decision=True) act, reg [L]; zero beyond the episode's end."""
	n = min(L + 1, sim.remaining())
	E, PE, act, reg = (torch.zeros(L) for _ in range(4))
	rew, o = sim.rollout(plan[:n])
	m = n - 1                                                               # steps j with a real state s_{j+1}
	if m > 0:
		d = r_hat[1:n] - rew[1:n]
		E[:m] = torch.cumsum(pl.disc ** torch.arange(m, dtype=d.dtype) * d, 0).abs()
	if n > 0:
		zt = pl.model.encode(o[:min(n, L)].to(pl.dev), pl.task.expand(min(n, L)))
		PE[:min(n, L)] = (zt - Z[1:min(n, L) + 1]).norm(dim=-1).cpu()
	if decision:
		for j in range(1, m + 1):
			pm = pl.prev_mean.clone()
			if j > 1:
				pm[:-(j - 1)] = pl.prev_mean[j - 1:]
				pm[-(j - 1):] = 0
			pl2.prev_mean.copy_(pm)
			a_star = pl2.act(o[j - 1])[0]
			act[j - 1] = float((a_star - plan[j]).norm())
			zr = pl.model.encode(o[j - 1].to(pl.dev).view(1, -1), pl.task).repeat(2, 1)
			q = q_heads(pl.model, zr, torch.stack([a_star, plan[j]]), pl.cfg, pl.task.expand(2)).mean(0).view(-1)
			reg[j - 1] = float(q[0] - q[1])
	return dict(E=E, PE=PE, act=act, reg=reg)


# ------------------------------------------------------------------------------------------------ calibration, arms
def reset(env, sim, ti, seed, push=0.0):
	torch.manual_seed(seed)
	obs = env.reset(task_idx=ti)
	sim.reseed(seed)
	obs = env.reset(task_idx=ti)
	sim.set_pushes(seed, push)
	return obs


@torch.no_grad()
def calibrate(env, sim, pl, pl2, ti, episodes, seed0, push=0.0):
	rec = {}
	for i in range(episodes):
		obs, done, t = reset(env, sim, ti, seed0 + i, push), False, 0
		while not done:
			plan = pl.act(obs, t0=t == 0)
			s, Z, r_hat = plan_scores(pl, obs, plan)
			s.update(oracles(pl, pl2, sim, obs, plan, r_hat, Z, decision=True))
			for k, v in s.items():
				rec.setdefault(k, []).append(v)
			obs, _, done, _ = sim.step(plan[0].cpu())
			t += 1
	return {k: torch.stack(v) for k, v in rec.items()}


def fit_bound(cal: dict) -> tuple:
	"""Weights (a, b) of the bound gate BND_j = a * IR_j + b * PE_j (regret <= IR + Lip * dist + slack): non-negative
	least squares of the reg oracle on [IR, PE, 1] over the calibration steps (IR_j, PE_j, reg_j all judge plan
	action j+1 at state j+1). Falls back to IR alone if both weights are 0."""
	from scipy.optimize import nnls
	ir, pe, reg = (cal[k][:, :L].reshape(-1).double().numpy() for k in ("IR", "PE", "reg"))
	(a, b, _), _ = nnls(np.stack([ir, pe, np.ones_like(ir)], 1), reg)
	return (float(a), float(b)) if a + b > 0 else (1.0, 0.0)


def thresholds(cal: dict, bound: tuple = None) -> dict:
	src = {g: cal["E" if g == "oracle" else g] for g in (*GATES, "oracle", "act", "reg", "PE")}
	if bound is not None:
		src["BND"] = bound[0] * cal["IR"] + bound[1] * cal["PE"]
	return {g: {key(b): match(torch.cummax(s[:, :L].float(), 1).values, b) for b in LEVELS} for g, s in src.items()}


@torch.no_grad()
def run_arm(env, sim, pl, pl2, ti, name, gate, budget, th, episodes, seed0, gseed, push=0.0, bound=None):
	gen = torch.Generator().manual_seed(gseed)
	q = random_q(budget) if gate == "random" else None
	ctrl = BudgetControl(budget) if gate not in (None, "random") else None
	rows = []
	for i in range(episodes):
		obs, done, t, ret, calls, j = reset(env, sim, ti, seed0 + i, push), False, 0, 0.0, 0, 0
		start = time.perf_counter()
		while not done:
			pl.shift(j)
			plan = pl.act(obs, t0=calls == 0)
			calls += 1
			pe_th, Z, bnd = None, None, None
			if gate is None:
				k = budget
			elif gate == "random":
				k = 1
				while k < K_MAX and bool(torch.rand(1, generator=gen) < q):
					k += 1
			else:
				thr = th[gate][ctrl.level()]
				if gate == "PE":
					_, Z, _ = plan_scores(pl, obs, plan, rs=False)
					k, pe_th = K_MAX, thr
				elif gate == "BND":    # checked while acting: running max of a * IR_j + b * PE_j, as calibrated
					s, Z, _ = plan_scores(pl, obs, plan, rs=False)
					k, bnd, run = K_MAX, s["IR"], 0.0
				elif gate in ("oracle", "act", "reg"):
					s, Z, r_hat = plan_scores(pl, obs, plan, rs=False)
					o = oracles(pl, pl2, sim, obs, plan, r_hat, Z, decision=gate != "oracle")
					k = commit_len(o["E" if gate == "oracle" else gate], thr, gen)
				else:
					s, _, _ = plan_scores(pl, obs, plan, rs=gate == "RS")
					k = commit_len(s[gate], thr, gen)
			j = 0
			while True:
				obs, r, done, _ = sim.step(plan[j].cpu())
				ret, t, j = ret + float(r), t + 1, j + 1
				if done or j >= k:
					break
				if pe_th is not None:
					e = float((pl.model.encode(obs.to(pl.dev).view(1, -1), pl.task)[0] - Z[j]).norm())
					if not passes(e, pe_th, gen):
						break
				if bnd is not None:    # deciding plan action j at the real state s_j: IR[j-1], PE at z_j
					e = float((pl.model.encode(obs.to(pl.dev).view(1, -1), pl.task)[0] - Z[j]).norm())
					run = max(run, bound[0] * float(bnd[j - 1]) + bound[1] * e)
					if not passes(run, thr, gen):
						break
			if ctrl is not None and not done:
				ctrl.update(j)
		rows.append(dict(arm=name, episode=i, seed=seed0 + i, success=ret / 1000.0, reward=ret, steps=t,
		                 plan_calls=calls, mean_k=t / calls, seconds=time.perf_counter() - start))
	return rows


@hydra.main(config_name="config", config_path=TDMPC2_DIR, version_base=None)
def main(cfg):
	from common.parser import parse_cfg
	from envs import make_env
	from tdmpc2 import TDMPC2

	out = Path(cfg.out)
	episodes, cal_eps = int(cfg.get("episodes", 30)), int(cfg.get("cal_episodes", 5))
	subset = str(cfg.get("tasks_subset"))
	arms_only = cfg.get("arms", None)
	fixed = cfg.get("fixed_ks", None)          # headroom pilot: only fixed-k arms, no calibration
	seed0 = int(cfg.get("seed0", 6000))
	push = float(cfg.get("push", 0.0))        # disturbance study: push strength (fraction of the robot's weight)
	study = str(cfg.get("study", "main"))
	cfg = parse_cfg(cfg)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	sim = DMCSim(env)
	for task in subset.split("+"):
		ti = list(cfg.tasks).index(task)
		disc = float(agent.discount[ti])
		pl, pl2 = Planner(agent, ti, disc), Planner(agent, ti, disc)
		d = out / task
		(d / "arms").mkdir(parents=True, exist_ok=True)
		bound = None
		if (d / "bound.json").exists():
			bound = tuple(json.loads((d / "bound.json").read_text()))
		if fixed:
			th = {}
		elif (d / "thresholds.json").exists():
			th = json.loads((d / "thresholds.json").read_text())
		else:
			cal = calibrate(env, sim, pl, pl2, ti, cal_eps, 6900, push)
			if study == "push":
				bound = fit_bound(cal)
				(d / "bound.json").write_text(json.dumps(bound))
				print(f"{task}: bound gate weights a={bound[0]:.4g} b={bound[1]:.4g}", flush=True)
			th = thresholds(cal, bound)
			torch.save(cal, d / "calibration.pt")
			E = cal["E"][:, :L]
			(d / "lba.json").write_text(json.dumps(dict(eps_hall=float(torch.quantile(E, 0.75)),
			                                            eps_clean=float(torch.quantile(E, 0.5)))))
			(d / "thresholds.json").write_text(json.dumps(th))
			print(f"{task}: calibrated on {E.shape[0]} plans", flush=True)
		arms = [(f"k{k}", None, int(k)) for k in str(fixed).split("+")] if fixed else (ARMS_PUSH if study == "push" else ARMS)
		for n, (name, gate, budget) in enumerate(arms):
			f = d / "arms" / f"{name}.pt"
			if f.exists() or (arms_only and name not in str(arms_only).split("+")):
				continue
			rows = run_arm(env, sim, pl, pl2, ti, name, gate, budget, th, episodes, seed0, n, push, bound)
			torch.save(dict(rows=rows), f)
			print(f"{task:20s} {name:10s} return {np.mean([r['reward'] for r in rows]):6.1f}  "
			      f"actions per plan {sum(r['steps'] for r in rows) / sum(r['plan_calls'] for r in rows):.2f}  "
			      f"{np.mean([r['seconds'] for r in rows]):.1f} s/episode", flush=True)
	print("done", flush=True)


if __name__ == "__main__":
	main()
