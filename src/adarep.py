"""AdaReP-style replanning gate (Cheng et al., arXiv 2606.23079), adapted to the equal-budget commitment protocol.

AdaReP replans when the deviation d of the real latent from the cached rollout exceeds an adaptive tolerance
    eps_{t+1} = eps_0 exp(-a_d dbar_{t+1}) exp(-a_L Lbar_t),
    d_{t+1} = ||z_{t+1} - zhat_{t+1}||,   Lhat_t = ||z_{t+1} - z_t|| / (||u_t|| + gamma),
with dbar, Lbar sliding-window means (window W). Taking logs, "d > eps" is "s > log eps_0" with
    s = log d + a_d dbar + a_L Lbar,
so we use s as the gate score and let the budget controller pick log eps_0 like every other gate's threshold.
Choices not fixed by the paper's abstract-level description (stated in docs/preregistration_live_ir.md, amendment 1):
W = 3, gamma = 1e-3, a_d = 1 / median(d), a_L = 1 / median(Lhat) on the calibration plans; the window covers the
steps of the current plan. With a_d = a_L = 0 the gate is PE.
"""
import math

import torch

W = 3
GAMMA = 1e-3


def score(d_hist: list, l_hist: list, norm: dict) -> float:
	db = sum(d_hist[-W:]) / len(d_hist[-W:])
	lb = sum(l_hist[-W:]) / len(l_hist[-W:])
	return math.log(d_hist[-1] + 1e-8) + db / norm["d"] + lb / norm["l"]


class Tracker:
	"""Online state of the gate within one plan (reset at every new plan)."""

	def __init__(self, agent, obs: torch.Tensor, roll_z: torch.Tensor, plan: torch.Tensor, norm: dict, th: dict):
		self.agent, self.roll_z, self.plan, self.norm, self.th = agent, roll_z, plan, norm, th
		self.dev = next(agent.model.parameters()).device
		self.z_prev = self._enc(obs)
		self.d, self.l, self.max = [], [], -math.inf

	@torch.no_grad()
	def _enc(self, obs):
		return self.agent.model.encode(obs.to(self.dev).view(1, -1), None)[0]

	def step(self, obs: torch.Tensor, j: int) -> float:
		"""After executing plan actions 0..j-1 and observing obs: running max of the score judging action j."""
		z = self._enc(obs)
		self.d.append(float((z - self.roll_z[j].to(self.dev)).norm()))
		self.l.append(float((z - self.z_prev).norm()) / (float(self.plan[j - 1].norm()) + GAMMA))
		self.z_prev = z
		self.max = max(self.max, score(self.d, self.l, self.norm))
		return self.max


@torch.no_grad()
def calibrate(agent, env, sim, lba: dict, episodes: int, seed_start: int, steps: int):
	"""Closed-loop k = 1 calibration episodes: per plan, the raw d and Lhat of steps 1..steps along the REAL rollout of
	the plan (simulator copy), the imagined main-head rollout as the cached plan. Returns (d [N, S], l [N, S])."""
	from src.commit import score_plan
	from src.monitor import H_LONG
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	planner = AuditedPlanner(agent, PlannerConfig(horizon=H_LONG, score="standard"))
	dev = next(agent.model.parameters()).device
	D, Lh = [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		while not done:
			action = planner.act(obs, t0=t == 0)
			plan = planner.last_plan.detach().cpu().clamp(-1, 1)
			_, roll, _ = score_plan(agent, obs, plan, lba)
			_, o = sim.rollout(sim.save(), plan.view(plan.shape[0], 1, -1))
			z = agent.model.encode(torch.cat([obs.view(1, -1), o[:steps, 0]]).to(dev), None)     # [S+1, d]
			d = (z[1:] - roll.z[1:steps + 1, 0].to(dev)).norm(dim=-1)
			l = (z[1:] - z[:-1]).norm(dim=-1) / (plan[:steps].to(dev).norm(dim=-1) + GAMMA)
			D.append(d.cpu())
			Lh.append(l.cpu())
			obs, _, done, _ = env.step(action)
			t += 1
	return torch.stack(D), torch.stack(Lh)


def scores_from(D: torch.Tensor, Lh: torch.Tensor, norm: dict) -> torch.Tensor:
	"""Gate scores [N, S] (step j judges action j+1) from calibration d, Lhat with the within-plan window."""
	N, S = D.shape
	out = torch.empty(N, S)
	for n in range(N):
		for j in range(S):
			out[n, j] = score(D[n, :j + 1].tolist(), Lh[n, :j + 1].tolist(), norm)
	return out
