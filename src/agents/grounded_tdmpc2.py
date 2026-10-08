"""TD-MPC2 with a grounded critic/policy and optional auxiliary dynamics heads (execution_final.md §3.1–3.2).

Run 1 (stock):    grounded=False, num_aux_dynamics=0  -> identical to TD-MPC2's _update.
Run 2 (grounded): grounded=True,  num_aux_dynamics=4  -> Q and π trained on encoded real latents;
                  4 extra dynamics heads trained on detached inputs (they never shape the encoder).

`_update` mirrors TDMPC2._update at the pinned commit (third_party/tdmpc2 @ e9f5932); changes are marked [CHANGED].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.tdmpc2_path  # noqa: F401
from common import init, layers, math
from tdmpc2 import TDMPC2


def make_aux_dynamics(cfg, n: int) -> nn.ModuleList:
	heads = nn.ModuleList([
		layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		for _ in range(n)
	])
	heads.apply(init.weight_init)
	return heads


class GroundedTDMPC2(TDMPC2):
	def __init__(self, cfg):
		self.grounded = bool(cfg.get("grounded", False))
		self.num_aux_dynamics = int(cfg.get("num_aux_dynamics", 0))
		super().__init__(cfg)  # compiles self._update (our override) with mode="reduce-overhead" if cfg.compile
		# Optional compile mode for the update (planning keeps TD-MPC2's "reduce-overhead"). With "reduce-overhead"
		# (CUDA graphs) every recompilation left the update permanently slower in our runs; "default" was ~1.8x slower
		# per update but stable across recompiles (src/tools/update_bench.py). The mode does not change the math.
		mode = cfg.get("compile_mode", "reduce-overhead")
		if cfg.compile and mode != "reduce-overhead":
			self._update = torch.compile(type(self)._update.__get__(self), mode=mode)
		if self.num_aux_dynamics > 0:
			self.model._dyn_aux = make_aux_dynamics(cfg, self.num_aux_dynamics).to(self.device)
			self.optim.add_param_group({"params": self.model._dyn_aux.parameters(), "lr": self.cfg.lr})

	def aux_dynamics_heads(self):
		"""All dynamics heads as callables (z, a) -> z', head 0 first (for Signals D and M)."""
		heads = [lambda z, a: self.model.next(z, a, None)]
		if self.num_aux_dynamics > 0:
			heads += [(lambda h: (lambda z, a: h(torch.cat([z, a], -1))))(h) for h in self.model._dyn_aux]
		return heads

	def checkpoint_state(self) -> dict:
		return {
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
			"scale": self.scale.state_dict(),
		}

	value_expansion = None   # src.training.value_expansion.ValueExpansion, set by the trainer (hallucination-aware targets)

	def _td_target(self, next_z, reward, terminated, task):
		ve = self.value_expansion
		if ve is not None and ve.active and ve.mode != "none":
			return ve.target(self, next_z, reward, terminated, task)
		return super()._td_target(next_z, reward, terminated, task)

	def load_checkpoint_state(self, state: dict) -> None:
		self.model.load_state_dict(state["model"])
		self.optim.load_state_dict(state["optim"])
		self.pi_optim.load_state_dict(state["pi_optim"])
		self.scale.load_state_dict(state["scale"])

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# [CHANGED] Auxiliary dynamics heads: own rollouts from the detached start latent (no encoder gradient).
		aux_consistency_loss = 0.
		if self.num_aux_dynamics > 0:
			for head in self.model._dyn_aux:
				z_k = zs[0].detach()
				for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
					z_k = head(torch.cat([z_k, _action], -1))
					aux_consistency_loss = aux_consistency_loss + F.mse_loss(z_k, _next_z) * self.cfg.rho**t
			aux_consistency_loss = aux_consistency_loss / self.cfg.horizon

		# [CHANGED] Grounded: Q is trained on encoded real latents (encoder gradient blocked), not rolled-out ones.
		_zs = zs[:-1]
		z_real = torch.cat([zs[:1].detach(), next_z], dim=0) if self.grounded else None
		qs = self.model.Q(z_real[:-1] if self.grounded else _zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * (consistency_loss + aux_consistency_loss) +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# [CHANGED] Grounded: π is updated on encoded real latents instead of rolled-out ones.
		pi_info = self.update_pi(z_real if self.grounded else zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = {
			"consistency_loss": consistency_loss,
			"aux_consistency_loss": aux_consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		}
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return {k: v.detach().mean() if isinstance(v, torch.Tensor) else torch.tensor(v) \
			for k, v in info.items()}
