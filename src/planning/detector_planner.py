"""MPPI planners that use the learned Bellman audit (LBA) on every candidate plan (pre-registered in
docs/preregistration_detector_planning.md).

Base score: the lambda-return  G_t = r_t + gamma((1 - lambda_t) V(z_{t+1}) + lambda_t G_{t+1})  (our best earlier
planner uses lambda_t = 0.8 everywhere). The audit gives p_t, the probability that the imagined return up to z_{t+1}
is wrong, for every candidate and step. Two ways to use it:

  lam = "lba":   lambda_t = lambda_max * (1 - p_t)  -> stop trusting a candidate's imagination where it is hallucinated
                 and fall back on the critic's value of the last trusted state (per candidate, per step).
  drop > 0:      in every MPPI iteration, remove the fraction ``drop`` of candidates the audit finds most suspicious
                 (max_t p_t) before the elites are chosen.
Controls: "lba_shuffled" (the same p profiles, randomly reassigned across candidates) and random dropping.
"""
from dataclasses import dataclass

import torch

from src.auditor.elvis import elvis_return
from src.auditor.signals import audit_rollout
from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from src.planning.gpu_audit import candidate_audit

LAMS = ("const", "lba", "lba_shuffled", "lba_pure")


@dataclass(frozen=True)
class DetectorRule:
	name: str
	horizon: int
	lam: str = "const"           # const | lba | lba_shuffled | lba_pure (lambda_t = 1 - p_t)
	lam_max: float = 0.8
	drop: float = 0.0            # fraction of candidates removed per MPPI iteration
	drop_random: bool = False    # control: remove a random subset of the same size


class DetectorPlanner(AuditedPlanner):
	def __init__(self, agent, rule: DetectorRule, audit=None, seed: int = 0):
		"""``audit``: dict(trees=GPUTrees, bias=[H]) for this horizon; needed unless the rule never reads p."""
		assert rule.lam in LAMS and 0.0 <= rule.drop < 1.0
		super().__init__(agent, PlannerConfig(horizon=rule.horizon, score="elvis", lambda_signal="const",
		                                      lambda_max=rule.lam_max))
		self.rule, self.audit = rule, audit
		self.needs_p = rule.lam != "const" or (rule.drop > 0 and not rule.drop_random)
		if self.needs_p:
			assert audit is not None and len(audit["bias"]) == rule.horizon, "audit must match the planning horizon"
		self.gen = torch.Generator().manual_seed(seed)

	def estimate_value(self, z, actions):
		r, (H, N) = self.rule, actions.shape[:2]
		if self.needs_p:
			p, roll = candidate_audit(self.agent, self.audit["trees"], self.audit["bias"], z, actions)   # p [H-1, N]
		else:
			p, roll = None, audit_rollout(self.model, self.cfg, z, actions, self.discount)
		if r.lam == "const":
			lam = torch.full((H, N, 1), r.lam_max, device=self.device)
		else:
			q = p
			if r.lam == "lba_shuffled":
				q = p[:, torch.randperm(N, generator=self.gen).to(self.device)]
			q = torch.cat([q, q[-1:]], 0)                                    # lambda_{H-1}: last audited step
			lam = ((1.0 if r.lam == "lba_pure" else r.lam_max) * (1 - q)).unsqueeze(-1)
			self.stats["mean_lambda"].append(float(lam.mean()))
		score = elvis_return(roll.r_hat, roll.v_mean, lam, self.discount)
		if r.drop > 0:
			k = int(round(r.drop * N))
			if r.drop_random:
				idx = torch.randperm(N, generator=self.gen)[:k].to(self.device)
			else:
				idx = p.max(0).values.topk(k).indices
			score = score.clone()
			score[idx] = float("-inf")
		if p is not None:
			self.stats["mean_p"].append(float(p.mean()))
		return score, None
