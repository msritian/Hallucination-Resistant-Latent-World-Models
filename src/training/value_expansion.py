"""Hallucination-aware value expansion for TD-MPC2 (docs/preregistration_value_expansion.md).

Standard TD-MPC2 target for a real transition:  G0 = r + gamma * Q_tgt(z', pi(z'))   (z' = encoded real next state)
Value expansion (MVE) imagines j steps with the policy from z' and uses
	Gj = r + gamma * ( sum_{k<j} gamma^k r_hat_k + gamma^j Q_tgt(z_j, pi(z_j)) ),   j = 0..h.
Hallucination-aware mixing: imagined step k is trusted with probability t_k; the target is the expected return when
imagination is cut at the first untrusted step:
	G = sum_{j<h} T_j (1 - t_j) Gj + T_h Gh,   T_j = prod_{k<j} t_k.
Modes: none (G0), fixed (t = 1 -> Gh), const (t = 0.5), lba (t from the Bellman audit), D (t from ensemble
disagreement). For lba and D the raw score goes through the same per-step empirical-CDF mapping, t = 1 - CDF(score),
so both have the same average trust and differ only in WHERE they trust.
"""
import torch

from src.auditor.signals import decode
from src.planning.gpu_audit import GPUTrees, gpu_features
from src.preflight.run_preflight import fit_lba, plan_signals

MODES = ("none", "fixed", "const", "lba", "D")


def mix(G: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
	"""G [h+1, N, 1] (targets G0..Gh), t [h, N, 1] trust per imagined step -> [N, 1]."""
	h = t.shape[0]
	T = torch.cat([torch.ones_like(t[:1]), torch.cumprod(t, 0)], 0)          # T_0..T_h
	w = torch.cat([T[:h] * (1 - t), T[h:]], 0)                                 # stop-at-j probabilities, sum to 1
	return (w * G).sum(0)


class ValueExpansion:
	def __init__(self, mode: str, h: int = 5, start_step: int = 50_000, reservoir: int = 50_000, seed: int = 0):
		assert mode in MODES
		self.mode, self.h, self.start_step, self.R = mode, h, start_step, reservoir
		self.active, self.trees, self.bias = False, None, None
		self.buf = None                                   # [h, R] recent raw scores (per-step CDF)
		self.gen = torch.Generator().manual_seed(seed)
		self.stats = {}

	def refresh(self, agent, episodes):
		"""Refit the audit on the newest real episodes (lba mode); activate expansion."""
		if self.mode == "lba":
			from src.training.audit_replay import episodes_to_data
			data = episodes_to_data(episodes[-50:])
			lba = fit_lba(agent, data, self.h, beta=1.0)
			dev = next(agent.model.parameters()).device
			self.trees, self.bias = GPUTrees(lba["clf"], dev), lba["bias"].to(dev)
		self.active = True

	@torch.no_grad()
	def _score(self, agent, z, actions):
		"""Raw per-step score [h, N] of the imagined policy rollout (higher = less trustworthy)."""
		if self.mode == "lba":
			sig, roll = plan_signals(agent, z, actions, 1.0, bias=self.bias, ensemble=False)
			L = self.h - 1
			X = gpu_features(sig, roll, L)
			p = self.trees.predict_proba(X.reshape(-1, X.shape[-1])).view(L, -1)
			return torch.cat([p, p[-1:]], 0)                                     # last step: reuse the last audited one
		sig, _ = plan_signals(agent, z, actions, 1.0, ensemble=True)
		return sig["D"].float()

	def _trust(self, score):
		"""t = 1 - per-step empirical CDF of the score over a reservoir of recent scores."""
		h, N = score.shape
		s = score.detach()
		self.buf = s if self.buf is None else torch.cat([self.buf, s], 1)[:, -self.R:]
		srt = self.buf.sort(1).values
		cdf = torch.stack([torch.searchsorted(srt[k].contiguous(), s[k].contiguous()) for k in range(h)]).float() / srt.shape[1]
		return (1 - cdf).clamp(0, 1)

	@torch.no_grad()
	def target(self, agent, next_z, reward, terminated, task=None):
		"""Replaces TDMPC2._td_target. next_z [T, B, d] real next latents, reward/terminated [T, B, 1]."""
		model, cfg, g = agent.model, agent.cfg, agent.discount
		T, B, d = next_z.shape
		z = next_z.reshape(T * B, d)
		zs, acts, r_hat = [z], [], []
		for k in range(self.h):
			a, _ = model.pi(zs[-1], task)
			acts.append(a)
			r_hat.append(decode(model.reward(zs[-1], a, task), cfg))
			zs.append(model.next(zs[-1], a, task))
		v = []
		for j in range(self.h + 1):
			a, _ = model.pi(zs[j], task)
			v.append(model.Q(zs[j], a, task, return_type="min", target=True))
		r = reward.reshape(T * B, 1)
		nt = (1 - terminated).reshape(T * B, 1)
		G, acc = [], torch.zeros_like(r)
		for j in range(self.h + 1):
			G.append(r + g * nt * (acc + g ** j * v[j]))
			if j < self.h:
				acc = acc + g ** j * r_hat[j]
		G = torch.stack(G)                                                         # [h+1, N, 1]
		if self.mode == "fixed":
			t = torch.ones(self.h, T * B, 1, device=z.device)
		elif self.mode == "const":
			t = torch.full((self.h, T * B, 1), 0.5, device=z.device)
		else:
			t = self._trust(self._score(agent, z, torch.stack(acts))).unsqueeze(-1)
		self.stats = dict(mean_trust=float(t.mean()))
		return mix(G, t).reshape(T, B, 1)
