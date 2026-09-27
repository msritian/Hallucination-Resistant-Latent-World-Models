"""Evaluation-time MPPI planner with a separate planning horizon and switchable plan scoring (execution_final.md §3.5, §3.7, §7).

Mirrors TDMPC2._plan (third_party/tdmpc2 @ e9f5932) but with its own horizon and scoring:
	standard: sum of imagined rewards + terminal value, with the mean-of-5-heads critic and the policy's mean action
	trust:    audited trust score with a chosen detection signal (A, B, C, D, E, M) and per-step thresholds tau(t)
	elvis:    ELVIS-style UCB λ-return
All scoring modes share the same rollout (dynamics head 0), so arms differ only in how plans are scored.
"""
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch

import src.tdmpc2_path  # noqa: F401
from common import math
from src.auditor.elvis import RunningNorm, elvis_lambdas, elvis_return
from src.auditor.signals import audit_rollout, bellman_target_spread, decode, dynamics_disagreement, pi_mean, q_heads
from src.auditor.trust import effective_horizon, trust_weighted_return, trust_weights

SCORES = ("standard", "trust", "elvis")
SIGNALS = ("A", "B", "C", "D", "E", "M")


@dataclass
class PlannerConfig:
	horizon: int = 12
	score: str = "trust"
	signal: str = "A"
	kappa: float = 1.0
	tau_pct: int = 95          # which calibrated percentile of the clean residuals defines "normal"
	beta: float = 1.0          # UCB weight for Signal E and ELVIS
	lambda_min: float = 0.0    # ELVIS
	lambda_max: float = 1.0    # ELVIS

	def name(self) -> str:
		if self.score == "standard":
			return f"standard_H{self.horizon}"
		if self.score == "elvis":
			return f"elvis_H{self.horizon}_lmin{self.lambda_min}_lmax{self.lambda_max}_b{self.beta}"
		return f"trust_{self.signal}_H{self.horizon}_k{self.kappa}_p{self.tau_pct}"


class AuditedPlanner:
	def __init__(self, agent, pcfg: PlannerConfig, taus: Optional[dict] = None):
		assert pcfg.score in SCORES and pcfg.signal in SIGNALS
		self.agent, self.p = agent, pcfg
		self.cfg, self.model = agent.cfg, agent.model
		self.device = next(agent.model.parameters()).device
		self.discount = float(agent.discount)
		H = pcfg.horizon
		if pcfg.score == "trust":
			needed = {"C": ["A", "B"]}.get(pcfg.signal, [pcfg.signal])
			for k in needed:
				assert taus is not None and k in taus and len(taus[k]) >= H, f"need tau for signal {k} up to step {H}"
			self.taus = {k: torch.as_tensor(taus[k][:H], dtype=torch.float32, device=self.device).view(H, 1, 1) for k in needed}
		if pcfg.signal in ("D", "M") and pcfg.score == "trust":
			assert agent.num_aux_dynamics > 0, "Signals D and M need the auxiliary dynamics heads (Run 2 models)"
		self.norm = RunningNorm()
		self.prev_mean = torch.zeros(H, self.cfg.action_dim, device=self.device)
		self.stats = defaultdict(list)

	# ------------------------------------------------------------------ scoring
	def _standard(self, z, actions):
		G, disc = 0, 1.0
		for t in range(actions.shape[0]):
			G = G + disc * decode(self.model.reward(z, actions[t], None), self.cfg)
			z = self.model.next(z, actions[t], None)
			disc *= self.discount
		v = q_heads(self.model, z, pi_mean(self.model, z), self.cfg).mean(0)
		return G + disc * v, None

	def _signal(self, roll, actions):
		x = {}
		name = self.p.signal
		if name in ("A", "C"):
			x["A"] = roll.delta
		if name in ("B", "C"):
			x["B"] = roll.critic_spread()
		if name == "E":
			x["E"] = roll.ucb(self.p.beta)[1:]
		if name == "D":
			x["D"] = dynamics_disagreement(self.agent.aux_dynamics_heads(), roll.z, actions)
		if name == "M":
			x["M"] = bellman_target_spread(self.agent.aux_dynamics_heads(), self.model, self.cfg, roll, actions)
		s = {k: v / self.taus[k] for k, v in x.items()}
		return torch.maximum(s["A"], s["B"]) if name == "C" else s[name]

	def _trust(self, z, actions):
		roll = audit_rollout(self.model, self.cfg, z, actions, self.discount)
		w = trust_weights(self._signal(roll, actions), self.p.kappa)
		score, omega = trust_weighted_return(roll.r_hat, roll.q_sa, roll.q_terminal, w, self.discount)
		return score, omega

	def _elvis(self, z, actions):
		roll = audit_rollout(self.model, self.cfg, z, actions, self.discount)
		ucb = roll.ucb(self.p.beta)
		lam = elvis_lambdas(ucb, self.norm, self.p.lambda_min, self.p.lambda_max)
		self.norm.update(ucb)
		return elvis_return(roll.r_hat, roll.v_mean, lam, self.discount), None

	def estimate_value(self, z, actions):
		fn = {"standard": self._standard, "trust": self._trust, "elvis": self._elvis}[self.p.score]
		return fn(z, actions)

	# ------------------------------------------------------------------ MPPI (mirrors TDMPC2._plan)
	@torch.no_grad()
	def plan(self, obs, t0=False, eval_mode=True):
		cfg, H, A = self.cfg, self.p.horizon, self.cfg.action_dim
		z = self.model.encode(obs, None)
		if cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=self.device)
			_z = z.repeat(cfg.num_pi_trajs, 1)
			for t in range(H - 1):
				pi_actions[t], _ = self.model.pi(_z, None)
				_z = self.model.next(_z, pi_actions[t], None)
			pi_actions[-1], _ = self.model.pi(_z, None)

		z = z.repeat(cfg.num_samples, 1)
		mean = torch.zeros(H, A, device=self.device)
		std = torch.full((H, A), cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self.prev_mean[1:]
		actions = torch.empty(H, cfg.num_samples, A, device=self.device)
		if cfg.num_pi_trajs > 0:
			actions[:, :cfg.num_pi_trajs] = pi_actions

		omega_elite = None
		for _ in range(cfg.iterations):
			r = torch.randn(H, cfg.num_samples - cfg.num_pi_trajs, A, device=self.device)
			actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)

			value, omega = self.estimate_value(z, actions)
			value = value.nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			if omega is not None:
				omega_elite = omega[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(cfg.temperature * (elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(cfg.min_std, cfg.max_std)

		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		chosen = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std0 = chosen[0], std[0]
		if not eval_mode:
			a = a + std0 * torch.randn(A, device=self.device)
		self.prev_mean.copy_(mean)
		if omega_elite is not None:
			self.stats["h_eff"].append(float((effective_horizon(omega_elite) * score.view(-1, 1)).sum()))
			self.stats["trust_collapsed"].append(float(((omega_elite[-1] < 0.01).float() * score.view(-1, 1)).sum()))
		return a.clamp(-1, 1)

	def act(self, obs, t0=False, eval_mode=True):
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if self.device.type == "cuda":
			torch.cuda.synchronize()
		start = time.perf_counter()
		a = self.plan(obs, t0=t0, eval_mode=eval_mode)
		if self.device.type == "cuda":
			torch.cuda.synchronize()
		self.stats["plan_ms"].append(1000 * (time.perf_counter() - start))
		return a.cpu()

	def reset_stats(self):
		self.stats = defaultdict(list)
