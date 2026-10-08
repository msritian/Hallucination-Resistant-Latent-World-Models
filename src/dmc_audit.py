"""Bellman audit of RELEASED pretrained TD-MPC2 world models on DeepMind Control (docs/preregistration_dmc_audit.md).

Loads a checkpoint exactly as TD-MPC2's evaluate.py does (its own config, env and agent), then per task:
  1. collect `episodes` real episodes with the agent's own planner (observations, actions, simulator rewards);
  2. first half: calibrate the audit (thresholds, bias, trees: same recipe as our ManiSkill results);
  3. second half: score every 12-step window (starts every `stride` steps) with the audit and with the baselines that
     need no extra models (critic-head spread B, ELVIS-style E, raw one-step A, anchored Aa); simulator labels.
No extra models exist for released checkpoints, so ensemble baselines are not applicable (the point of the study).

Run with Hydra overrides from TD-MPC2's config plus our keys, e.g.
	python -m src.dmc_audit task=mt30 model_size=48 checkpoint=/path/mt30-48M.pt +out=dmc +episodes=20 +tasks_subset=all
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TDMPC2_DIR = str(Path(__file__).resolve().parent.parent / "third_party" / "tdmpc2" / "tdmpc2")
sys.path.insert(0, TDMPC2_DIR)

import hydra  # noqa: E402

from src.preflight.metrics import stratified_auroc  # noqa: E402
from src.preflight.run_preflight import (CRITIC_FEATURES, _features, fit_lba, return_error, reward_windows,  # noqa: E402
                                         rollout_signals, windows)

H = 12


def strided(data: dict, stride: int) -> dict:
	"""Keep windows starting every `stride` steps by cutting each episode into overlapping chunks of H+1 states."""
	obs, act, rew = data["obs"], data["action"], data["reward"]
	T = act.shape[1]
	starts = list(range(0, T - H + 1, stride))
	return dict(obs=torch.stack([obs[:, s:s + H + 1] for s in starts], 1).flatten(0, 1),
	            action=torch.stack([act[:, s:s + H] for s in starts], 1).flatten(0, 1),
	            reward=torch.stack([rew[:, s:s + H] for s in starts], 1).flatten(0, 1),
	            episode=torch.arange(obs.shape[0]).repeat_interleave(len(starts)))


@torch.no_grad()
def collect(env, agent, task_idx, episodes, seed0):
	obs_all, act_all, rew_all = [], [], []
	for i in range(episodes):
		torch.manual_seed(seed0 + i)
		obs, done, t = env.reset(task_idx=task_idx) if task_idx is not None else env.reset(), False, 0
		O, A, R = [torch.as_tensor(obs).float().cpu()], [], []
		while not done:
			a = agent.act(obs, t0=t == 0, task=task_idx)
			obs, r, done, info = env.step(a)
			O.append(torch.as_tensor(obs).float().cpu()); A.append(torch.as_tensor(a).float().cpu()); R.append(float(r)); t += 1
		obs_all.append(torch.stack(O)); act_all.append(torch.stack(A)); rew_all.append(torch.tensor(R))
	return dict(obs=torch.stack(obs_all), action=torch.stack(act_all), reward=torch.stack(rew_all))


@torch.no_grad()
def audit_task(agent, view, data, task_idx, stride):
	dev = next(agent.model.parameters()).device
	E = data["obs"].shape[0]
	cal = strided({k: v[:E // 2] for k, v in data.items()}, stride)
	ev = strided({k: v[E // 2:] for k, v in data.items()}, stride)
	lba = fit_lba(view, cal, H, 1.0, task=task_idx)
	out = {"n_pos_cal": lba["n_pos"]}
	M = ev["obs"].shape[0]
	tk = None if task_idx is None else torch.full((M,), int(task_idx), device=dev, dtype=torch.long)
	o, a, _ = windows(ev, H, range(M))
	sig, _, roll = rollout_signals(view, agent.model.encode(o.to(dev), tk), a.to(dev), 1.0, bias=lba["bias"].to(dev),
	                               ensemble=False, task=tk)
	L = H - 1
	X = _features({k: v.cpu() for k, v in sig.items()}, roll, CRITIC_FEATURES, L, True)
	scores = {"LBA": torch.as_tensor(lba["clf"].predict_proba(X.numpy())[:, 1]).view(L, -1).float()}
	for k in ("B", "E", "A", "Aa"):
		scores[k] = sig[k][:L].cpu().float()
	Eev = return_error(roll, reward_windows(ev, H, range(M)), view.discount)
	# labels exactly as in the ManiSkill results: thresholds from calibration one-step errors
	o_c, a_c, _ = windows(cal, H, range(cal["obs"].shape[0]))
	tkc = None if task_idx is None else torch.full((o_c.shape[1],), int(task_idx), device=dev, dtype=torch.long)
	_, _, roll_c = rollout_signals(view, agent.model.encode(o_c.to(dev), tkc), a_c.to(dev), 1.0, ensemble=False, task=tkc)
	Ecal = return_error(roll_c, reward_windows(cal, H, range(cal["obs"].shape[0])), view.discount)
	hi, lo = torch.quantile(Ecal[0].flatten(), 0.99), torch.quantile(Ecal[0].flatten(), 0.9)
	inc = lambda v: torch.cat([v[:1], v[1:] - v[:-1]], 0)
	big = torch.quantile(inc(Ecal).flatten(), 0.99)
	pos, keep = Eev > hi, (Eev > hi) | (Eev <= lo)
	sudden = pos & torch.cummax((inc(Eev) > big).int(), 0).values.bool()
	steps = torch.arange(L).view(-1, 1).expand_as(pos)
	for name, p in (("all", pos), ("sudden", sudden), ("gradual", pos & ~sudden)):
		k = p | (keep & ~pos)
		out[name] = {d: (stratified_auroc(s[k], p[k], steps[k]) if p[k].any() and (~p[k]).any() else float("nan"))
		             for d, s in scores.items()}
	out["positive_rate"] = float(pos[keep].float().mean())
	out["return"] = float(data["reward"].sum(1).mean())
	return out


@hydra.main(config_name="config", config_path=TDMPC2_DIR, version_base=None)
def main(cfg):
	from common.parser import parse_cfg
	from envs import make_env
	from tdmpc2 import TDMPC2

	out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
	episodes, stride, seed0 = int(cfg.get("episodes", 20)), int(cfg.get("stride", 5)), int(cfg.get("seed0", 11000))
	subset = str(cfg.get("tasks_subset", "all"))
	cfg = parse_cfg(cfg)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
	for i, task in enumerate(tasks):
		if subset != "all" and task not in subset.split("+"):
			continue
		if task in results:
			continue
		ti = i if cfg.multitask else None
		disc = agent.discount[ti] if cfg.multitask else agent.discount
		view = SimpleNamespace(model=agent.model, cfg=agent.cfg, discount=float(disc), num_aux_dynamics=0)
		data = collect(env, agent, ti, episodes, seed0)
		results[task] = audit_task(agent, view, data, ti, stride)
		r = results[task]["all"]
		print(f"{task:26s} return {results[task]['return']:7.1f}  pos {results[task]['positive_rate']:.3f}  "
		      + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
		(out / "results.json").write_text(json.dumps(results, indent=1))
	print("done", flush=True)


if __name__ == "__main__":
	main()
