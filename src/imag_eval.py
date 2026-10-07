"""Track 1: when to trust imagined policy evaluation (docs/preregistration_imag_eval.md).

Candidate policies act on the world model's latent state: a_t = clip(pi_mean(z_t) + sigma * eps_t + b) with eps_t drawn
ONCE per (start seed, t), so every imagined episode has an exact real twin (same start state, same noise sequence).
For each policy and start seed:
  real:     run the policy in the simulator (z_t = encode(o_t))          -> real discounted return R
  imagined: run it in the world model from encode(o_0) for the same T   -> imagined discounted return R_hat
            and score the imagined episode with every detector (max over consecutive 12-step audit chunks):
            LBA (ours), D, M (ensembles), B (critic spread)
Analysis (src/tools/imag_eval_report.py): does the score predict |R_hat - R|; risk-coverage of trusted episodes; policy
ranking and value estimates from trusted episodes only.
"""
import argparse
import json
from pathlib import Path

import torch

from src.auditor.signals import pi_mean
from src.preflight.run_preflight import CRITIC_FEATURES, _features, fit_lba, plan_signals

POLICIES = {   # name: (sigma, bias on xyz translation, random)
	"pi": (0.0, 0.0, False), "pi_n0.3": (0.3, 0.0, False), "pi_n0.6": (0.6, 0.0, False), "pi_n1.0": (1.0, 0.0, False),
	"pi_bias0.3": (0.0, 0.3, False), "random": (0.0, 0.0, True),
}


def act(model, z, eps_t, spec):
	sigma, bias, rnd = spec
	if rnd:
		return eps_t.clamp(-1, 1)
	a = pi_mean(model, z) + sigma * eps_t
	if bias:
		a = a.clone()
		a[..., :3] = a[..., :3] + bias
	return a.clamp(-1, 1)


@torch.no_grad()
def real_episode(agent, env, seed, eps, spec):
	dev = next(agent.model.parameters()).device
	obs, done, t, R, g = env.reset(seed=seed), False, 0, 0.0, 1.0
	o0 = obs.clone()
	while not done:
		a = act(agent.model, agent.model.encode(obs.to(dev).view(1, -1), None), eps[t].to(dev).view(1, -1), spec)
		obs, r, done, info = env.step(a.view(-1).cpu())
		R += g * float(r); g *= float(agent.discount); t += 1
	return o0, R, float(info["success"]), t


@torch.no_grad()
def imagined_episode(agent, o0, T, eps, spec, lba, H=12):
	dev = next(agent.model.parameters()).device
	model, g = agent.model, float(agent.discount)
	z = model.encode(o0.to(dev).view(1, -1), None)
	scores = {k: [] for k in ("LBA", "D", "M", "B")}
	R_hat, t = 0.0, 0
	while t < T:
		h = min(H, T - t)
		acts, zs = [], z
		for k in range(h):                                   # the policy acts on the imagined state
			a = act(model, zs, eps[t + k].to(dev).view(1, -1), spec)
			acts.append(a)
			zs = model.next(zs, a, None)
		acts = torch.stack(acts)                             # [h, 1, A]
		sig, roll = plan_signals(agent, z, acts, 1.0, bias=lba["bias"].to(dev)[:h])
		R_hat += sum(g ** (t + k) * float(roll.r_hat[k]) for k in range(h))
		if h == H:                                           # full audit chunk (the audit was fitted on 12-step windows)
			X = _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, H - 1, True)
			scores["LBA"].append(float(lba["clf"].predict_proba(X.numpy())[:, 1].max()))
			for k in ("D", "M", "B"):
				if k in sig:
					scores[k].append(float(sig[k][:H - 1].max()))
		z, t = roll.z[-1], t + h
	return R_hat, {k: max(v) if v else float("nan") for k, v in scores.items()}


def main(argv=None):
	from src.envs.maniskill3 import ManiSkill3Env
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", required=True)
	p.add_argument("--starts", type=int, default=30)
	p.add_argument("--seed_start", type=int, default=9000)
	p.add_argument("--out", required=True)
	a = p.parse_args(argv)
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	agent = load_agent(a.model, a.run, a.task)
	lba = fit_lba(agent, torch.load(a.episodes_file), 12, beta=1.0)
	env = ManiSkill3Env(a.task, seed=a.seed_start)
	A = agent.cfg.action_dim
	rows = []
	for name, spec in POLICIES.items():
		for i in range(a.starts):
			seed = a.seed_start + i
			gen = torch.Generator().manual_seed(seed)
			eps = (torch.rand(env.max_episode_steps + 1, A, generator=gen) * 2 - 1) if spec[2] else \
				torch.randn(env.max_episode_steps + 1, A, generator=gen)
			o0, R, succ, T = real_episode(agent, env, seed, eps, spec)
			R_hat, sc = imagined_episode(agent, o0, T, eps, spec, lba)
			rows.append(dict(policy=name, seed=seed, real_return=R, success=succ, imagined_return=R_hat, **sc))
		m = [r for r in rows if r["policy"] == name]
		print(f"{name:12s} real {sum(r['real_return'] for r in m) / len(m):6.2f}  imagined "
		      f"{sum(r['imagined_return'] for r in m) / len(m):6.2f}  success {sum(r['success'] for r in m) / len(m):.2f}", flush=True)
		(out / "rows.json").write_text(json.dumps(rows))
	env.close()
	print("done", flush=True)


if __name__ == "__main__":
	main()
