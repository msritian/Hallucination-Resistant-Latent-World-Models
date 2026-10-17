"""Phase A: does the Bellman audit detect test-time distribution shift, and only when it matters?
(docs/preregistration_shift.md)

A trained robot (TD-MPC2's own planner) acts in shifted versions of its task:
  none            nominal (false-alarm reference)
  gain:<g>        weaker/stronger motors: executed action = g * action
  bias:<b>        actuator drift: executed action[:3] += b (end-effector translation)
  mass:<m>        task objects m times heavier (best effort through the simulator API; actual masses are logged)
  obs:<o>         sensor offset on the agent's observation only (the physics is unchanged)
For every 12-step window of real experience we compute, from the window's start and actions only:
  LBA   our learned Bellman audit (fitted on the robot's NOMINAL saved episodes), max over steps
  D, M, B  ensemble / critic-spread baselines, max over steps
and, using the observed future (an after-the-fact monitor):
  PE1   one-step latent prediction error  ||dynamics(z_t, a_t) - encode(o_{t+1})||
  PE    mean latent prediction error over the window
Outputs per condition: success, detection AUROC (shifted vs nominal windows), detection delay, false alarms.
"""
import argparse
import json
from pathlib import Path

import torch

from src.preflight.metrics import auroc
from src.preflight.run_preflight import CRITIC_FEATURES, _features, fit_lba, rollout_signals, windows

SIGNALS = ("LBA", "D", "M", "B", "PE1", "PE")


def make_env(task, seed, cond: str):
	from src.envs.maniskill3 import ManiSkill3Env

	kind, _, val = cond.partition(":")
	env = ManiSkill3Env(task, seed=seed)
	if kind == "none":
		return env, {}
	v = float(val)
	info = {}
	base_step, base_reset = env.step, env.reset

	if kind == "gain":
		env.step = lambda a: base_step((a * v).clamp(-1, 1))
	elif kind == "bias":
		def step(a):
			a = a.clone()
			a[:3] = a[:3] + v
			return base_step(a.clamp(-1, 1))
		env.step = step
	elif kind == "obs":
		g = torch.Generator().manual_seed(0)
		offset = None
		def shift_obs(o):
			nonlocal offset
			if offset is None:   # a fixed offset on 3 random observation dims (sensor miscalibration)
				offset = torch.zeros_like(o)
				offset[torch.randperm(o.numel(), generator=g)[:3]] = v
				info["obs_dims"] = torch.nonzero(offset).flatten().tolist()
			return o + offset
		env.reset = lambda seed=None: shift_obs(base_reset(seed))
		def step(a):
			o, r, d, i = base_step(a)
			return shift_obs(o), r, d, i
		env.step = step
	elif kind == "mass":
		def reset(seed=None):
			o = base_reset(seed)
			u = env.env.unwrapped
			masses = []
			for name in ("cubeA", "cubeB", "cube", "obj", "peg"):
				actor = getattr(u, name, None)
				for body in getattr(actor, "_bodies", []) if actor is not None else []:
					try:
						m0 = float(body.mass)
						body.set_mass(m0 * v) if hasattr(body, "set_mass") else setattr(body, "mass", m0 * v)
						masses.append((name, m0, float(body.mass)))
					except Exception as exc:   # logged, never fatal: the summary shows whether the shift was applied
						info["mass_error"] = repr(exc)
			info["masses"] = masses
			return o
		env.reset = reset
	else:
		raise ValueError(cond)
	return env, info


@torch.no_grad()
def collect(agent, task, cond, episodes, seed_start):
	env, info = make_env(task, seed_start, cond)
	obs_all, act_all, rew_all, succ = [], [], [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		O, A, R = [obs], [], []
		while not done:
			a = agent.act(obs, t0=t == 0, eval_mode=True)
			obs, r, done, inf = env.step(a)
			O.append(obs); A.append(a); R.append(float(r)); t += 1
		obs_all.append(torch.stack(O)); act_all.append(torch.stack(A)); rew_all.append(torch.tensor(R))
		succ.append(float(inf["success"]))
	env.close()
	return dict(obs=torch.stack(obs_all), action=torch.stack(act_all), reward=torch.stack(rew_all),
	            success=torch.tensor(succ)), info


@torch.no_grad()
def window_scores(agent, data, lba, H=12):
	"""Per-window scores [M] and per-window (episode, start) for every 12-step window of real experience."""
	dev = next(agent.model.parameters()).device
	E, T = data["action"].shape[:2]
	o, a, ep = windows(data, H, range(E))
	z_real = agent.model.encode(o.to(dev), None)
	sig, err, roll = rollout_signals(agent, z_real, a.to(dev), 1.0, bias=lba["bias"].to(dev))
	L = H - 1
	X = _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, L, True)
	p = torch.as_tensor(lba["clf"].predict_proba(X.numpy())[:, 1]).view(L, -1)
	out = {"LBA": p.max(0).values.float()}
	for k in ("D", "M", "B"):
		if k in sig:
			out[k] = sig[k][:L].max(0).values.cpu().float()
	out["PE1"] = err[0].cpu().float()
	out["PE"] = err.mean(0).cpu().float()
	start = torch.tensor([s for _ in range(E) for s in range(T - H + 1)])
	return out, ep, start


def detection(nom: dict, sh: dict, nom_idx, sh_idx, T_win: int) -> dict:
	"""Window AUROC (shifted vs nominal), and per episode the first window start where the running mean of the score
	exceeds the nominal 95th percentile of running means (delay; false alarms measured on nominal episodes)."""
	res = {}
	for k in SIGNALS:
		if k not in nom or k not in sh:
			continue
		s = torch.cat([sh[k], nom[k]]); y = torch.cat([torch.ones(len(sh[k])), torch.zeros(len(nom[k]))]).bool()
		def running(scores, idx):
			ep, st = idx
			runs = {}
			for e in torch.unique(ep):
				m = ep == e
				order = st[m].argsort()
				x = scores[m][order]
				runs[int(e)] = torch.cumsum(x, 0) / torch.arange(1, len(x) + 1)
			return runs
		rn, rs = running(nom[k], nom_idx), running(sh[k], sh_idx)
		thr = torch.quantile(torch.cat(list(rn.values())), 0.95)
		first = lambda r: next((t for t, v in enumerate(r) if v > thr), None)
		delays = [first(r) for r in rs.values()]
		fa = [first(r) is not None for r in rn.values()]
		res[k] = dict(auroc=float(auroc(s, y)), detected_episodes=sum(d is not None for d in delays) / len(delays),
		              median_delay_steps=float(torch.tensor([d for d in delays if d is not None]).float().median())
		              if any(d is not None for d in delays) else None, false_alarm_episodes=sum(fa) / len(fa))
	return res


def main(argv=None):
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", required=True)
	p.add_argument("--episodes_file", required=True, help="the robot's NOMINAL saved episodes (fits the audit)")
	p.add_argument("--conditions", default="none,gain:0.5,bias:0.3,mass:3,obs:0.5")
	p.add_argument("--episodes", type=int, default=20)
	p.add_argument("--seed_start", type=int, default=8000)
	p.add_argument("--out", required=True)
	a = p.parse_args(argv)
	out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
	agent = load_agent(a.model, a.run, a.task)
	lba = fit_lba(agent, torch.load(a.episodes_file), 12, beta=1.0)
	scores, idx, summary = {}, {}, {}
	for cond in a.conditions.split(","):
		data, info = collect(agent, a.task, cond, a.episodes, a.seed_start)
		s, ep, st = window_scores(agent, data, lba)
		scores[cond], idx[cond] = s, (ep, st)
		summary[cond] = dict(success=float(data["success"].mean()), info=info,
		                     mean_scores={k: float(v.mean()) for k, v in s.items()})
		print(f"{cond:10s} success {summary[cond]['success']:.2f}  " +
		      "  ".join(f"{k} {v:.3f}" for k, v in summary[cond]["mean_scores"].items()) + f"  {info}", flush=True)
		torch.save(dict(scores=scores, idx=idx, summary=summary), out / "shift.pt")
	for cond in scores:
		if cond != "none":
			summary[cond]["detection"] = detection(scores["none"], scores[cond], idx["none"], idx[cond], 12)
	(out / "summary.json").write_text(json.dumps(summary, indent=1))
	print(json.dumps({c: summary[c].get("detection") for c in summary}, indent=1), flush=True)


if __name__ == "__main__":
	main()
