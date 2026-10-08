"""Track 2 (Horde): plug-in per-signal Bellman audits for a FROZEN world model (docs/preregistration_horde.md).

Signals (cumulants): every state-observation dimension, standardised: c_t = std(o_{t+1}). On the robot's stored real
experience (frozen encoder/dynamics), train
  C(z, a) -> R^D   predicted next signals             (the analogue of the reward head)
  G(z, a) -> R^D   general value functions: G = c + gamma_c * G(z', pi(z'))   (one critic per signal, Horde-style)
Along an imagined rollout from a real start, per-signal residual
  delta^d_k = | G_d(z_k, a_k) - (C_d(z_k, a_k) + gamma_c * G_d(z_{k+1}, pi(z_{k+1}))) |
and per-signal answer key (imagined vs real discounted signal return along the same actions)
  E^d_k = | sum_{j<=k} gamma_c^j (C_d(z_j, a_j) - c_d,t+j) |.
Tests (25 calibration / 25 test episodes of the robot's saved real episodes):
  reward-free  trees on signal-residual features only (no reward, no reward critic) predict the REWARD hallucination labels
  per-signal   AUROC of delta^d for its own label E^d (mean over signals)
  localise     on windows where some signal is hallucinated, does argmax_d delta^d (scaled) hit argmax_d E^d (scaled)?
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.auditor.signals import pi_mean
from src.preflight.metrics import auroc, stratified_auroc
from src.preflight.run_preflight import (CRITIC_FEATURES, _features, calibration_bias, return_error, reward_windows,
                                         rollout_signals, windows, make_gbt)


def mlp(i, o, h=256):
	return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.Mish(), nn.Linear(h, h), nn.LayerNorm(h), nn.Mish(), nn.Linear(h, o))


class Horde(nn.Module):
	def __init__(self, latent, act, D):
		super().__init__()
		self.C, self.G = mlp(latent + act, D), mlp(latent + act, D)

	def c(self, z, a):
		return self.C(torch.cat([z, a], -1))

	def g(self, z, a):
		return self.G(torch.cat([z, a], -1))


def train_horde(agent, data, gamma_c=0.9, updates=20000, batch=1024, lr=3e-4, seed=0):
	"""Fit C and G on encoded REAL transitions; the world model and policy stay frozen."""
	torch.manual_seed(seed)
	dev = next(agent.model.parameters()).device
	obs, act = data["obs"], data["action"]                          # [E, T+1, D], [E, T, A]
	mu, sd = obs.flatten(0, 1).mean(0), obs.flatten(0, 1).std(0) + 1e-6
	with torch.no_grad():
		z = torch.cat([agent.model.encode(obs[i].to(dev), None).cpu() for i in range(obs.shape[0])]).view(obs.shape[0], obs.shape[1], -1)
	Z0, Z1 = z[:, :-1].flatten(0, 1), z[:, 1:].flatten(0, 1)
	A0, Cn = act.flatten(0, 1), ((obs[:, 1:] - mu) / sd).flatten(0, 1)
	H = Horde(z.shape[-1], act.shape[-1], obs.shape[-1]).to(dev)
	target = copy.deepcopy(H.G).requires_grad_(False)
	opt = torch.optim.Adam(H.parameters(), lr=lr)
	N = Z0.shape[0]
	for step in range(updates):
		i = torch.randint(N, (batch,))
		z0, z1, a0, c = Z0[i].to(dev), Z1[i].to(dev), A0[i].to(dev), Cn[i].to(dev)
		with torch.no_grad():
			tgt = c + gamma_c * target(torch.cat([z1, pi_mean(agent.model, z1)], -1))
		loss = ((H.c(z0, a0) - c) ** 2).mean() + ((H.g(z0, a0) - tgt) ** 2).mean()
		opt.zero_grad(); loss.backward(); opt.step()
		with torch.no_grad():
			for p, q in zip(target.parameters(), H.G.parameters()):
				p.lerp_(q, 0.01)
	H.eval()
	return H, mu, sd


@torch.no_grad()
def signal_audit(agent, H, mu, sd, data, eps, gamma_c=0.9, Hh=12):
	"""Per-window, per-step, per-signal residuals and answer keys for the given episodes: [L, M, D] each."""
	dev = next(agent.model.parameters()).device
	o, a, _ = windows(data, Hh, eps)
	z = agent.model.encode(o.to(dev), None)
	zs = [z[0]]
	for k in range(Hh):
		zs.append(agent.model.next(zs[-1], a[k].to(dev), None))
	L = Hh - 1
	c_hat = torch.stack([H.c(zs[k], a[k].to(dev)) for k in range(Hh)])                       # [H, M, D]
	g = torch.stack([H.g(zs[k], a[k].to(dev)) for k in range(Hh)])
	g_next = torch.stack([H.g(zs[k + 1], pi_mean(agent.model, zs[k + 1])) for k in range(Hh)])
	delta = (g - (c_hat + gamma_c * g_next)).abs()[:L].cpu()                                   # [L, M, D]
	# per-signal ensemble disagreement (rival localiser): spread over dynamics heads of the next-step signal prediction
	ens = None
	if agent.num_aux_dynamics > 0:
		heads = agent.aux_dynamics_heads()
		ens = torch.stack([torch.stack([H.c(h(zs[k], a[k].to(dev)), a[k + 1].to(dev)) for h in heads]).std(0)
		                   for k in range(L)]).cpu()                                                # [L, M, D]
	c_real = ((o[1:] - mu) / sd)                                                                 # [H, M, D] real next signals
	disc = gamma_c ** torch.arange(Hh, dtype=torch.float32).view(-1, 1, 1)
	E = torch.cumsum(disc * (c_hat.cpu() - c_real), 0).abs()[:L]
	return delta, E, ens


def obs_names(task):
	"""Names of the flattened state-observation dims (ManiSkill flattens the state dict in key order); [] if unavailable."""
	try:
		import gymnasium as gym
		import mani_skill.envs  # noqa: F401
		env = gym.make(task, num_envs=1, obs_mode="state_dict", sim_backend="cpu", control_mode="pd_ee_delta_pose")
		obs, _ = env.reset(seed=0)
		env.close()
		names = []
		def walk(prefix, x):
			if isinstance(x, dict):
				for k, v in x.items():
					walk(f"{prefix}/{k}" if prefix else k, v)
			else:
				n = int(torch.as_tensor(x).reshape(-1).numel())
				names.extend([f"{prefix}[{i}]" for i in range(n)])
		walk("", obs)
		return names
	except Exception as exc:
		print(f"signal names unavailable: {exc!r}", flush=True)
		return []


def main(argv=None):
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", required=True)
	p.add_argument("--train_ckpt", required=True, help="checkpoint.pt with the robot's stored training episodes")
	p.add_argument("--train_episodes", type=int, default=3000, help="newest stored episodes used to fit the heads")
	p.add_argument("--episodes_file", required=True, help="the robot's 50 saved evaluation episodes (with rewards)")
	p.add_argument("--updates", type=int, default=20000)
	p.add_argument("--out", required=True)
	a = p.parse_args(argv)
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	agent = load_agent(a.model, a.run, a.task)
	dev = next(agent.model.parameters()).device

	src = torch.load(a.train_ckpt, map_location="cpu", weights_only=False)["episodes"]
	n = src["reward"].shape[0]
	tr = dict(obs=src["obs"][n - a.train_episodes:].float(), action=src["action"][n - a.train_episodes:, 1:].float())
	H, mu, sd = train_horde(agent, tr, updates=a.updates)
	print(f"Horde heads trained on {tr['obs'].shape[0]} episodes, {tr['obs'].shape[-1]} signals", flush=True)

	data = torch.load(a.episodes_file)
	Ne = data["obs"].shape[0]
	cal, ev = list(range(Ne // 2)), list(range(Ne // 2, Ne))
	dc, Ec, ens_c = signal_audit(agent, H, mu, sd, data, cal)
	de, Ee, ens_e = signal_audit(agent, H, mu, sd, data, ev)
	L = dc.shape[0]
	steps_e = torch.arange(L).view(-1, 1).expand(L, de.shape[1])
	res = {}

	# per-signal detection: each residual vs its own label (thresholds per signal from calibration)
	per = []
	for d in range(dc.shape[-1]):
		e0 = Ec[0, :, d]
		hi, lo = torch.quantile(e0, 0.99), torch.quantile(e0, 0.9)
		y, keep = Ee[..., d] > hi, (Ee[..., d] > hi) | (Ee[..., d] <= lo)
		if y[keep].sum() >= 5 and (~y[keep]).sum() >= 5:
			per.append(stratified_auroc(de[..., d][keep], y[keep], steps_e[keep]))
			res.setdefault("_dims", []).append(d)
	res["per_signal_auroc_mean"] = sum(per) / len(per) if per else float("nan")
	res["per_signal_auroc"] = per

	# localisation: scale residuals and errors by their calibration medians, compare argmax on hallucinated windows
	s_d = dc.flatten(0, 1).median(0).values + 1e-6
	s_e = Ec.flatten(0, 1).median(0).values + 1e-6
	hall = (Ee / s_e).max(-1).values > torch.quantile((Ec / s_e).max(-1).values, 0.99)
	hit = ((de / s_d).argmax(-1) == (Ee / s_e).argmax(-1))[hall]
	res["localisation_top1"] = float(hit.float().mean()) if hall.any() else float("nan")
	res["localisation_chance"] = 1.0 / dc.shape[-1]
	# baselines: always guess the signal most often wrong on calibration; per-signal ensemble disagreement
	hall_c = (Ec / s_e).max(-1).values > torch.quantile((Ec / s_e).max(-1).values, 0.99)
	majority = int(torch.bincount((Ec / s_e).argmax(-1)[hall_c], minlength=dc.shape[-1]).argmax()) if hall_c.any() else 0
	res["localisation_majority"] = float(((Ee / s_e).argmax(-1)[hall] == majority).float().mean()) if hall.any() else float("nan")
	if ens_e is not None:
		s_n = ens_c.flatten(0, 1).median(0).values + 1e-6
		res["localisation_ensemble"] = float(((ens_e / s_n).argmax(-1) == (Ee / s_e).argmax(-1))[hall].float().mean()) if hall.any() else float("nan")
	# which signals are most often the wrong one (named when names are available)
	if hall.any():
		cnt = torch.bincount((Ee / s_e).argmax(-1)[hall], minlength=dc.shape[-1])
		res["most_wrong_signals"] = [[int(i), int(cnt[i])] for i in cnt.argsort(descending=True)[:5]]
	res["per_signal_auroc_by_dim"] = {}

	# reward-free detection of REWARD hallucination: trees on signal-residual summaries only
	g = float(agent.discount)
	def reward_labels(eps):
		o, ac, _ = windows(data, 12, eps)
		_, _, roll = rollout_signals(agent, agent.model.encode(o.to(dev), None), ac.to(dev), 1.0, ensemble=False)
		return return_error(roll, reward_windows(data, 12, eps), g), roll
	Erc, roll_c = reward_labels(cal); Ere, roll_e = reward_labels(ev)
	def summ(delta):   # per step: max/mean/median over signals, and their running maxima, plus step
		f = torch.stack([delta.max(-1).values, delta.mean(-1), delta.median(-1).values], -1)
		return torch.cat([f, torch.cummax(f, 0).values, torch.arange(L).view(-1, 1, 1).expand(L, f.shape[1], 1).float()], -1)
	e1 = Erc[0].flatten()
	hi, lo = torch.quantile(e1, 0.99).item(), torch.quantile(e1, 0.9).item()
	def lab(E):
		return E > hi, (E > hi) | (E <= lo)
	yc, kc = lab(Erc); ye, ke = lab(Ere)
	Xc, Xe = summ(dc), summ(de)
	clf = make_gbt().fit(Xc.reshape(-1, Xc.shape[-1])[kc.reshape(-1)].numpy(), yc.reshape(-1)[kc.reshape(-1)].numpy())
	pr = torch.as_tensor(clf.predict_proba(Xe.reshape(-1, Xe.shape[-1]).numpy())[:, 1]).view(L, -1).float()
	res["reward_free_auroc"] = stratified_auroc(pr[ke], ye[ke], steps_e[ke])
	# same trees on the reward-critic audit (our main detector) for reference, same windows and labels
	bias = calibration_bias(agent, agent.model.encode(windows(data, 12, cal)[0].to(dev), None), windows(data, 12, cal)[1].to(dev), 1.0, 12)
	def lba_X(eps):
		o, ac, _ = windows(data, 12, eps)
		sig, _, roll = rollout_signals(agent, agent.model.encode(o.to(dev), None), ac.to(dev), 1.0, bias=bias.to(dev), ensemble=False)
		return _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, L, True)
	Lc, Le = lba_X(cal), lba_X(ev)
	clf2 = make_gbt().fit(Lc[kc.reshape(-1)].numpy(), yc.reshape(-1)[kc.reshape(-1)].numpy())
	p2 = torch.as_tensor(clf2.predict_proba(Le.numpy())[:, 1]).view(L, -1).float()
	res["reward_audit_auroc"] = stratified_auroc(p2[ke], ye[ke], steps_e[ke])
	Xb = torch.cat([Lc, Xc.reshape(-1, Xc.shape[-1])], -1); Xbe = torch.cat([Le, Xe.reshape(-1, Xe.shape[-1])], -1)
	clf3 = make_gbt().fit(Xb[kc.reshape(-1)].numpy(), yc.reshape(-1)[kc.reshape(-1)].numpy())
	p3 = torch.as_tensor(clf3.predict_proba(Xbe.numpy())[:, 1]).view(L, -1).float()
	res["combined_auroc"] = stratified_auroc(p3[ke], ye[ke], steps_e[ke])
	# reward-free rival: ensemble disagreement D (raw, per step) on the same windows and labels
	def d_scores(eps):
		o, ac, _ = windows(data, 12, eps)
		sig, _, _ = rollout_signals(agent, agent.model.encode(o.to(dev), None), ac.to(dev), 1.0)
		return sig["D"][:L].cpu() if "D" in sig else None
	Dd = d_scores(ev)
	if Dd is not None:
		res["ensemble_D_auroc"] = stratified_auroc(Dd[ke], ye[ke], steps_e[ke])
	names = obs_names(a.task)
	res["signal_names"] = names if len(names) == dc.shape[-1] else []   # only if the layout matches exactly
	res["per_signal_auroc_by_dim"] = {(res["signal_names"][d] if res["signal_names"] else str(d)): v
	                                  for d, v in zip(res.pop("_dims", []), res["per_signal_auroc"])}
	(out / "horde.json").write_text(json.dumps(res, indent=1))
	print(json.dumps({k: v for k, v in res.items() if k != "per_signal_auroc"}, indent=1), flush=True)


if __name__ == "__main__":
	main()
