"""Pessimistic planning for offline-trained TD-MPC2 (docs/preregistration_offline_pessimism.md).

MPPI (H=3, TD-MPC2's own horizon) scores each candidate with its imagined return; a pessimistic arm subtracts
	lambda * std(score) * zscore(penalty)
over the candidates of the same MPPI iteration (scale-free; only differences between candidates matter for the choice).
Penalties (per candidate, max over imagined steps):
	LBA  the Bellman audit p_k (fitted ONLY on the offline dataset, as in a real offline setting)
	D    dynamics-ensemble disagreement (MOReL / PETS-style)
	M    Bellman-target spread over dynamics heads (MOBILE-style)
	B    spread of the critic heads (Q-ensemble, edge-of-reach style)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.planning.audited_planner import AuditedPlanner, PlannerConfig
from src.planning.gpu_audit import GPUTrees, gpu_features
from src.preflight.run_preflight import fit_lba, plan_signals

PENALTIES = ("LBA", "D", "M", "B")
LAMBDAS = (0.5, 1.0, 2.0)


class PessimisticPlanner(AuditedPlanner):
	def __init__(self, agent, penalty=None, lam=0.0, audit=None, horizon=3):
		super().__init__(agent, PlannerConfig(horizon=horizon, score="standard"))
		self.penalty, self.lam, self.audit = penalty, lam, audit

	def estimate_value(self, z, actions):
		H = actions.shape[0]
		sig, roll = plan_signals(self.agent, z, actions, 1.0,
		                         bias=None if self.audit is None else self.audit["bias"],
		                         ensemble=self.penalty in ("D", "M"))
		disc = self.discount ** torch.arange(H, device=self.device, dtype=torch.float32).view(-1, 1, 1)
		score = (disc * roll.r_hat).sum(0) + self.discount ** H * roll.v_mean[-1]          # [N, 1]
		if self.penalty is None or self.lam == 0:
			return score, None
		L = H - 1
		if self.penalty == "LBA":
			X = gpu_features(sig, roll, L)
			pen = self.audit["trees"].predict_proba(X.reshape(-1, X.shape[-1])).view(L, -1).max(0).values
		else:
			pen = sig[self.penalty][:L].max(0).values.float()
		pen = (pen - pen.mean()) / (pen.std() + 1e-6)
		self.stats["pen_std"].append(float(score.std()))
		return score - self.lam * score.std() * pen.view(-1, 1), None


def run(env, planner, episodes, seed_start, name):
	rows = []
	for i in range(episodes):
		obs, done, t, start = env.reset(seed=seed_start + i), False, 0, time.perf_counter()
		while not done:
			obs, _, done, info = env.step(planner.act(obs, t0=t == 0, eval_mode=True))
			t += 1
		rows.append(dict(arm=name, episode=i, seed=seed_start + i, success=float(info["success"]), seconds=time.perf_counter() - start))
	return rows


def main(argv=None):
	from src.envs.maniskill3 import ManiSkill3Env
	from src.monitor import write_outputs
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="stock_aux")
	p.add_argument("--task", required=True)
	p.add_argument("--dataset", required=True, help="offline_episodes.pt written by the offline trainer")
	p.add_argument("--fit_episodes", type=int, default=300)
	p.add_argument("--episodes", type=int, default=50)
	p.add_argument("--seed_start", type=int, default=10000)
	p.add_argument("--out", required=True)
	a = p.parse_args(argv)
	out = Path(a.out); (out / "arms").mkdir(parents=True, exist_ok=True)
	agent = load_agent(a.model, a.run, a.task)
	dev = next(agent.model.parameters()).device
	data = torch.load(a.dataset)
	idx = torch.randperm(data["obs"].shape[0], generator=torch.Generator().manual_seed(0))[:a.fit_episodes]
	lba = fit_lba(agent, {k: v[idx] for k, v in data.items()}, 3, beta=1.0)
	trees = GPUTrees(lba["clf"], dev)
	err = trees.check(lba["clf"], lba["X_sample"])
	audit = dict(trees=trees, bias=lba["bias"].to(dev))
	print(f"audit fitted on {len(idx)} dataset episodes ({lba['n_pos']} positive steps); GPU trees max error {err:.1e}", flush=True)
	arms = [("none", None, 0.0)] + [(f"{k}_l{l}", k, l) for k in PENALTIES for l in LAMBDAS
	                                if k == "LBA" or k == "B" or agent.num_aux_dynamics > 0]
	env = ManiSkill3Env(a.task, seed=a.seed_start)
	for name, pen, lam in arms:
		f = out / "arms" / f"{name}.pt"
		if f.exists():
			continue
		rows = run(env, PessimisticPlanner(agent, pen, lam, audit), a.episodes, a.seed_start, name)
		torch.save(dict(rows=rows), f)
		print(f"{name:10s} success {sum(r['success'] for r in rows) / len(rows):.2f}  "
		      f"{sum(r['seconds'] for r in rows) / len(rows):.1f} s/episode", flush=True)
	env.close()
	write_outputs(out, [type("A", (), {"name": n})() for n, _, _ in arms])
	(out / "audit.json").write_text(json.dumps(dict(n_pos=lba["n_pos"], fit_episodes=len(idx))))
	print("done", flush=True)


if __name__ == "__main__":
	main()
