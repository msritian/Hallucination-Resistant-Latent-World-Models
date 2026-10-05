"""Detector-guided replay: train the world model more on the experience where its imagination fails
(pre-registered in docs/preregistration_audit_replay.md).

Every ``refresh`` steps the sampler scores every stored 12-step window of real experience:
  lba: max_t p_t of the learned Bellman audit (refitted on the newest episodes, same recipe as the detection results)
  D:   max_t dynamics-ensemble disagreement (published-style control)
and keeps the ``top`` fraction of windows as the priority set. Each training batch then takes ``frac`` of its
sequences from the priority set (a random training slice inside a random priority window) and the rest uniformly
from the replay buffer, exactly as TD-MPC2 samples.
"""
import torch

from src.planning.gpu_audit import GPUTrees, gpu_features
from src.preflight.run_preflight import fit_lba, rollout_signals


def episodes_to_data(episodes: list) -> dict:
	"""Trainer episodes (TensorDicts with a NaN first action/reward) -> obs [E, T+1, D], action [E, T, A], reward [E, T]."""
	return dict(obs=torch.stack([e["obs"] for e in episodes]).float(),
	            action=torch.stack([e["action"][1:] for e in episodes]).float(),
	            reward=torch.stack([e["reward"][1:] for e in episodes]).float())


class AuditReplay:
	def __init__(self, buffer, mode: str, frac: float = 0.5, top: float = 0.3, H: int = 12, fit_episodes: int = 50,
	             chunk: int = 4096, seed: int = 0):
		assert mode in ("lba", "D") and 0 < frac < 1 and 0 < top <= 1
		self.buffer, self.mode, self.frac, self.top, self.H = buffer, mode, frac, top, H
		self.fit_episodes, self.chunk = fit_episodes, chunk
		self.gen = torch.Generator().manual_seed(seed)
		self.data, self.windows = None, None
		self.stats = {}

	@torch.no_grad()
	def refresh(self, agent, episodes: list):
		"""Re-score every stored window with the current model; keep the most suspicious ``top`` fraction."""
		dev = next(agent.model.parameters()).device
		data = episodes_to_data(episodes)
		E, T = data["action"].shape[:2]
		H = self.H
		if self.mode == "lba":
			recent = {k: v[-self.fit_episodes:] for k, v in data.items()}
			lba = fit_lba(agent, recent, H, beta=1.0)
			trees, bias = GPUTrees(lba["clf"], dev), lba["bias"].to(dev)
		starts = torch.arange(T - H + 1)
		scores = []
		for e0 in range(0, E, max(1, self.chunk // len(starts))):
			eps = torch.arange(e0, min(E, e0 + max(1, self.chunk // len(starts))))
			o = torch.stack([data["obs"][eps][:, s:s + H + 1] for s in starts], 1).flatten(0, 1).transpose(0, 1)   # [H+1, M, D]
			a = torch.stack([data["action"][eps][:, s:s + H] for s in starts], 1).flatten(0, 1).transpose(0, 1)    # [H, M, A]
			z = agent.model.encode(o.to(dev), None)
			if self.mode == "lba":
				sig, _, roll = rollout_signals(agent, z, a.to(dev), 1.0, bias=bias, ensemble=False)
				X = gpu_features(sig, roll, H - 1)
				s = trees.predict_proba(X.reshape(-1, X.shape[-1])).view(H - 1, -1).max(0).values
			else:
				sig, _, _ = rollout_signals(agent, z, a.to(dev), 1.0)
				s = sig["D"].max(0).values.float()
			scores.append(s.cpu())
		scores = torch.cat(scores)                                                   # [E * (T-H+1)], episode-major
		k = max(1, int(round(self.top * len(scores))))
		idx = scores.topk(k).indices
		self.windows = torch.stack([idx // len(starts), idx % len(starts)], 1)    # (episode, start)
		self.data = data
		self.stats = dict(windows=len(scores), priority_windows=k, score_mean=float(scores.mean()),
		                  score_priority_min=float(scores[idx].min()))

	def sample(self):
		"""Same tuple as Buffer.sample(): obs [h+1, B, D], action [h, B, A], reward [h, B, 1], terminated, task."""
		obs, action, reward, terminated, task = self.buffer.sample()
		if self.windows is None:
			return obs, action, reward, terminated, task
		h, B = action.shape[0], action.shape[1]
		n = int(round(self.frac * B))
		pick = self.windows[torch.randint(len(self.windows), (n,), generator=self.gen)]
		start = pick[:, 1] + torch.randint(0, self.H - h + 1, (n,), generator=self.gen)
		ep = pick[:, 0]
		t = start.view(1, -1) + torch.arange(h + 1).view(-1, 1)                    # [h+1, n]
		p_obs = self.data["obs"][ep.view(1, -1), t]                                  # [h+1, n, D]
		p_act = self.data["action"][ep.view(1, -1), t[:-1]]
		p_rew = self.data["reward"][ep.view(1, -1), t[:-1]].unsqueeze(-1)
		dev = obs.device
		obs = torch.cat([p_obs.to(dev, obs.dtype), obs[:, n:]], 1).contiguous()
		action = torch.cat([p_act.to(dev, action.dtype), action[:, n:]], 1).contiguous()
		reward = torch.cat([p_rew.to(dev, reward.dtype), reward[:, n:]], 1).contiguous()
		terminated = torch.cat([torch.zeros_like(reward[:, :n]), terminated[:, n:]], 1).contiguous()
		return obs, action, reward, terminated, task


@torch.no_grad()
def imagination_error(agent, episodes: list, H: int = 12) -> dict:
	"""How wrong the model's imagination is on real episodes: mean |imagined - real| discounted return over H-1 steps
	(the simulator answer key of the detection results), along the actions actually taken."""
	from src.preflight.run_preflight import return_error, reward_windows, windows
	dev = next(agent.model.parameters()).device
	data = episodes_to_data(episodes)
	ids = range(data["obs"].shape[0])
	o, a, _ = windows(data, H, ids)
	_, _, roll = rollout_signals(agent, agent.model.encode(o.to(dev), None), a.to(dev), 1.0, ensemble=False)
	E = return_error(roll, reward_windows(data, H, ids), float(agent.discount))
	return dict(return_error_last=float(E[-1].mean()), return_error_mean=float(E.mean()),
	            return_error_by_step=[float(x) for x in E.mean(1)])
