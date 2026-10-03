"""Model rollout vs simulator rollout from the same initial state and the same action sequence.

For one task and one trained TD-MPC2 model:
1. Probe: collect real episodes, encode every real state, and fit a linear map latent -> state observation (ridge
   regression, held-out R^2 reported). It is fitted on REAL latents only and used to read out what an imagined latent
   "means" physically (TD-MPC2 has no decoder).
2. Rollouts: in fresh episodes, at every ``every``-th step, snapshot the simulator and compare, for ``horizon`` steps,
   the model's open-loop imagination with the simulator along the SAME actions, for
     - policy:     the actions the agent actually executed next (the simulator result is the real episode itself);
     - candidates: action sequences from the planner's search at that state (elite and random MPPI samples),
                   executed in the simulator from the saved state.
   Per step it records predicted vs real reward, critic value of the imagined vs real state, latent error, probed
   state (imagined) vs real state, distance of the imagined latent to the nearest real latent (distribution shift),
   and the one-step and anchored Bellman residuals.

Outputs (out/): data.pt (all arrays), summary.json, report.md (tables and concrete examples).
Plots: python -m src.tools.plot_rollout_vs_sim out/   (needs matplotlib; run locally)

Example:
	python -m src.tools.rollout_vs_sim --model run/final_model.pt --run grounded --task StackCube-v1 --out rvs_stack
"""
import argparse
import json
from pathlib import Path

import torch

from src.auditor.signals import audit_rollout
from src.preflight.run_preflight import collect, value_fn
from src.tools.plan_diag import SimSnapshot


# ------------------------------------------------------------------------------------------------ observation names
def obs_names(task: str, dim: int) -> list:
	"""Names of the flattened state observation, from ManiSkill's state_dict layout (falls back to indices)."""
	try:
		import gymnasium as gym
		import mani_skill.envs  # noqa: F401
		from src.envs.lavapipe import register_lavapipe
		env = gym.make(task, num_envs=1, obs_mode="state_dict", control_mode="pd_ee_delta_pose", sim_backend="cpu",
		               render_backend=register_lavapipe())
		obs, _ = env.reset(seed=0)
		env.close()
		names = []

		def walk(prefix, x):
			if isinstance(x, dict):
				for k, v in x.items():
					walk(f"{prefix}.{k}" if prefix else k, v)
			else:
				n = int(torch.as_tensor(x).reshape(-1).numel())
				names.extend([f"{prefix}[{i}]" for i in range(n)] if n > 1 else [prefix])

		walk("", obs)
		if len(names) == dim:
			return names
	except Exception as e:  # noqa: BLE001
		print(f"obs names unavailable ({e}); using indices")
	return [f"obs[{i}]" for i in range(dim)]


# ------------------------------------------------------------------------------------------------ probe
def fit_probe(z: torch.Tensor, x: torch.Tensor, ridge: float = 1e-3, holdout: float = 0.2, seed: int = 0) -> dict:
	"""Ridge regression x ~ [z, 1] W. Returns W, the held-out R^2 per dimension, and standardization stats."""
	g = torch.Generator().manual_seed(seed)
	perm = torch.randperm(len(z), generator=g)
	n_te = int(holdout * len(z))
	te, tr = perm[:n_te], perm[n_te:]
	Z = torch.cat([z, torch.ones(len(z), 1)], 1).double()
	X = x.double()
	A = Z[tr].T @ Z[tr] + ridge * torch.eye(Z.shape[1], dtype=torch.float64)
	W = torch.linalg.solve(A, Z[tr].T @ X[tr])
	pred = Z[te] @ W
	ss_res = ((pred - X[te]) ** 2).sum(0)
	ss_tot = ((X[te] - X[te].mean(0)) ** 2).sum(0).clamp(min=1e-12)
	return dict(W=W.float(), r2=(1 - ss_res / ss_tot).float())


def probe(W: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
	return torch.cat([z, torch.ones(*z.shape[:-1], 1, dtype=z.dtype)], -1) @ W.to(z.dtype)


def nn_distance(q: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
	"""Distance from each query latent [..., d] to its nearest neighbour in ``bank`` [N, d]."""
	flat = q.reshape(-1, q.shape[-1])
	out = torch.cat([torch.cdist(flat[i:i + chunk], bank).min(1).values for i in range(0, len(flat), chunk)])
	return out.reshape(q.shape[:-1])


# ------------------------------------------------------------------------------------------------ collection
@torch.no_grad()
def collect_rollouts(agent, planner, task: str, episodes: int, seed_start: int, every: int, H: int, n_cand: int):
	from src.envs.maniskill3 import ManiSkill3Env

	env = ManiSkill3Env(task, seed=seed_start)
	sim = SimSnapshot(env)
	gen = torch.Generator().manual_seed(0)
	starts, det_err = [], []
	for i in range(episodes):
		obs, done, t = env.reset(seed=seed_start + i), False, 0
		ep_obs, ep_act, ep_rew = [obs], [], []
		pending = []
		while not done:
			planner.act(obs, t0=t == 0, eval_mode=True)
			action = agent.act(obs, t0=t == 0, eval_mode=True)
			if t % every == 0 and t + H <= env.max_episode_steps:
				snap = sim.save()
				pool = planner.last_pool
				value = pool["value"].squeeze(-1).cpu()
				order = value.argsort(descending=True)
				rand = order[n_cand:][torch.randperm(len(order) - n_cand, generator=gen)[:n_cand]]
				idx = torch.cat([order[:n_cand], rand])
				acts = pool["actions"][:, idx.to(pool["actions"].device)].cpu()
				if acts.shape[0] < H:  # planner horizon shorter than H: extend with the agent's policy mean later
					raise ValueError("planner horizon must be >= --horizon")
				acts = acts[:H]
				rew, obs_true = sim.rollout(snap, acts)
				_, check = sim.rollout(snap, action.view(1, 1, -1))
				pending.append(dict(t=t, obs0=obs.cpu(), cand_actions=acts, cand_r=rew, cand_obs=obs_true,
				                    cand_kind=torch.cat([torch.zeros(n_cand), torch.ones(len(rand))]).long(), check=check[0, 0]))
			obs_next, reward, done, info = env.step(action)
			if pending and pending[-1]["t"] == t:
				det_err.append(float((pending[-1].pop("check") - obs_next).abs().max()))
			ep_obs.append(obs_next)
			ep_act.append(action)
			ep_rew.append(float(reward))
			obs = obs_next
			t += 1
		ep_obs, ep_act, ep_rew = torch.stack(ep_obs), torch.stack(ep_act), torch.tensor(ep_rew)
		for p in pending:  # the policy's own future: the real episode from t on
			s = p["t"]
			p.update(episode=i, pol_actions=ep_act[s:s + H].unsqueeze(1), pol_r=ep_rew[s:s + H].unsqueeze(1),
			         pol_obs=ep_obs[s + 1:s + H + 1].unsqueeze(1), success=info["success"])
			starts.append(p)
		print(f"episode {i}: {len(starts)} start states, success {info['success']:.0f}, "
		      f"max determinism error {max(det_err) if det_err else float('nan'):.2e}", flush=True)
	env.close()
	return starts, det_err


# ------------------------------------------------------------------------------------------------ analysis
@torch.no_grad()
def analyze(agent, starts: list, W: torch.Tensor, bank: torch.Tensor, bank_holdout: torch.Tensor) -> dict:
	dev = next(agent.model.parameters()).device
	g = float(agent.discount)
	enc = lambda o: agent.model.encode(o.to(dev), None)
	out = {}
	for kind in ("policy", "candidates"):
		key = "pol" if kind == "policy" else "cand"
		acts = torch.cat([s[f"{key}_actions"] for s in starts], 1).to(dev)                    # [H, M, A]
		obs_true = torch.cat([s[f"{key}_obs"] for s in starts], 1)                              # [H, M, D]
		r_true = torch.cat([s[f"{key}_r"] for s in starts], 1)                                  # [H, M]
		K = [s[f"{key}_actions"].shape[1] for s in starts]
		obs0 = torch.cat([s["obs0"].view(1, -1).expand(k, -1) for s, k in zip(starts, K)])
		start_id = torch.cat([torch.full((k,), j) for j, k in enumerate(K)])
		H, M = acts.shape[:2]
		z0 = enc(obs0)
		roll = audit_rollout(agent.model, agent.cfg, z0, acts, g)
		z_hat = roll.z[1:].cpu()                                                                # [H, M, d]
		z_real = enc(obs_true.reshape(-1, obs_true.shape[-1])).reshape(H, M, -1).cpu()
		v_hat = roll.v_mean[1:].squeeze(-1).cpu()
		v_real = value_fn(agent, z_real.to(dev)).cpu()
		r_hat = roll.r_hat.squeeze(-1).cpu()
		disc = g ** torch.arange(H, dtype=torch.float32).view(-1, 1)
		ret_err = torch.cumsum(disc * (r_hat - r_true), 0)                                      # signed
		d = dict(
			actions=acts.cpu(), r_hat=r_hat, r_true=r_true, v_hat=v_hat, v_real=v_real,
			latent_err=(z_hat - z_real).norm(dim=-1), return_err=ret_err,
			state_hat=probe(W, z_hat.double()).float(), state_true=obs_true,
			state_probe_real=probe(W, z_real.double()).float(),
			nn_dist_imagined=nn_distance(z_hat, bank), nn_dist_real=nn_distance(z_real, bank),
			residual_A=roll.delta.squeeze(-1).cpu(), residual_anchored=roll.anchored_residual().squeeze(-1).cpu(),
			start_id=start_id, kind=(torch.cat([s["cand_kind"] for s in starts]) if key == "cand" else torch.zeros(M).long()),
		)
		out[kind] = d
	out["nn_dist_heldout_real"] = nn_distance(bank_holdout, bank)
	return out


def q(x: torch.Tensor, p: float) -> float:
	return float(torch.quantile(x.float().flatten(), p))


def summarize(res: dict, names: list, r2: torch.Tensor, det_err: list) -> dict:
	S = dict(determinism_max_err=max(det_err) if det_err else float("nan"),
	         probe_r2_median=float(r2.median()), probe_r2_by_dim={n: float(v) for n, v in zip(names, r2)})
	S["nn_dist_heldout_real_median"] = q(res["nn_dist_heldout_real"], 0.5)
	for kind in ("policy", "candidates"):
		d = res[kind]
		H = d["r_hat"].shape[0]
		groups = {"all": torch.ones(d["r_hat"].shape[1], dtype=torch.bool)}
		if kind == "candidates":
			groups = {"elite": d["kind"] == 0, "random": d["kind"] == 1}
		for gname, m in groups.items():
			per = []
			for t in range(H):
				per.append(dict(
					step=t + 1,
					reward_abs_err=float((d["r_hat"][t, m] - d["r_true"][t, m]).abs().median()),
					return_abs_err=float(d["return_err"][t, m].abs().median()),
					return_bias=float(d["return_err"][t, m].mean()),
					value_bias=float((d["v_hat"][t, m] - d["v_real"][t, m]).mean()),
					latent_err=float(d["latent_err"][t, m].median()),
					state_err_probe=float((d["state_hat"][t, m] - d["state_true"][t, m]).norm(dim=-1).median()),
					state_err_probe_real=float((d["state_probe_real"][t, m] - d["state_true"][t, m]).norm(dim=-1).median()),
					nn_dist_imagined=float(d["nn_dist_imagined"][t, m].median()),
					nn_dist_real=float(d["nn_dist_real"][t, m].median()),
				))
			S[f"{kind}/{gname}"] = per
	return S


def examples(res: dict, names: list, n: int = 3) -> list:
	"""The policy-action rollouts whose imagined return drifts most from reality, with the state dims that diverge most."""
	d = res["policy"]
	H = d["r_hat"].shape[0]
	worst = d["return_err"][-1].abs().argsort(descending=True)[:n]
	out = []
	for m in worst.tolist():
		gap = (d["state_hat"][:, m] - d["state_true"][:, m]).abs().max(0).values
		dims = gap.argsort(descending=True)[:4].tolist()
		rows = []
		for t in range(H):
			rows.append(dict(step=t + 1, r_hat=float(d["r_hat"][t, m]), r_true=float(d["r_true"][t, m]),
			                 v_hat=float(d["v_hat"][t, m]), v_real=float(d["v_real"][t, m]),
			                 residual_A=float(d["residual_A"][t, m]), residual_anchored=float(d["residual_anchored"][t, m]),
			                 latent_err=float(d["latent_err"][t, m]),
			                 state={names[k]: [float(d["state_hat"][t, m, k]), float(d["state_true"][t, m, k])] for k in dims}))
		out.append(dict(rollout=m, start_id=int(d["start_id"][m]), final_return_err=float(d["return_err"][-1, m]),
		                diverging_dims=[names[k] for k in dims], steps=rows))
	return out


def write_report(out: Path, S: dict, ex: list, args):
	L = [f"# Model rollout vs simulator: {args.task}, {args.run} model", "",
	     f"Same initial state, same action sequence, {args.horizon} steps. Simulator replay determinism error: "
	     f"{S['determinism_max_err']:.2e}. Linear probe latent -> state, held-out R^2 on real latents (median over dims): "
	     f"{S['probe_r2_median']:.3f}.", "",
	     "Distribution shift: median distance from a latent to its nearest REAL latent (bank of encoded real states); "
	     f"held-out real states: {S['nn_dist_heldout_real_median']:.4f}.", ""]
	for key in [k for k in S if "/" in k]:
		L += [f"## {key}", "", "| step | reward abs err | return abs err | return bias | value bias | latent err | "
		      "state err (probe of imagined) | state err (probe of real) | NN dist imagined | NN dist real |",
		      "| :-: |" + " :-: |" * 9]
		for r in S[key]:
			if r["step"] in (1, 2, 3, 6, 9, 12, 16, 20, 24) or r["step"] == len(S[key]):
				L.append(f"| {r['step']} | {r['reward_abs_err']:.3f} | {r['return_abs_err']:.3f} | {r['return_bias']:+.3f} | "
				         f"{r['value_bias']:+.3f} | {r['latent_err']:.3f} | {r['state_err_probe']:.3f} | "
				         f"{r['state_err_probe_real']:.3f} | {r['nn_dist_imagined']:.3f} | {r['nn_dist_real']:.3f} |")
		L.append("")
	L += ["## Concrete examples (policy actions, largest final return error)", ""]
	for e in ex:
		L += [f"**Rollout {e['rollout']}** (start {e['start_id']}): imagined minus real return after {args.horizon} steps "
		      f"= {e['final_return_err']:+.3f}. Most diverging state dims: {', '.join(e['diverging_dims'])}.", "",
		      "| step | reward imagined / real | value imagined / real | residual A | anchored | " +
		      " | ".join(f"{k} imagined / real" for k in e["diverging_dims"][:2]) + " |",
		      "| :-: | :-: | :-: | :-: | :-: | :-: | :-: |"]
		for r in e["steps"]:
			if r["step"] in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24):
				st = " | ".join(f"{r['state'][k][0]:+.3f} / {r['state'][k][1]:+.3f}" for k in e["diverging_dims"][:2])
				L.append(f"| {r['step']} | {r['r_hat']:.3f} / {r['r_true']:.3f} | {r['v_hat']:.2f} / {r['v_real']:.2f} | "
				         f"{r['residual_A']:.3f} | {r['residual_anchored']:.3f} | {st} |")
		L.append("")
	(out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None):
	from src.planning.audited_planner import AuditedPlanner, PlannerConfig
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--model", required=True)
	p.add_argument("--run", default="grounded")
	p.add_argument("--task", default="StackCube-v1")
	p.add_argument("--out", required=True)
	p.add_argument("--horizon", type=int, default=24)
	p.add_argument("--episodes", type=int, default=10)
	p.add_argument("--every", type=int, default=5)
	p.add_argument("--n_cand", type=int, default=8)
	p.add_argument("--probe_episodes", type=int, default=30)
	p.add_argument("--seed_start", type=int, default=9000)
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)

	agent = load_agent(args.model, args.run, args.task)
	dev = next(agent.model.parameters()).device

	# 1. probe on real latents
	pdata = collect(agent, args.task, args.probe_episodes, args.seed_start + 500)
	obs = pdata["obs"].reshape(-1, pdata["obs"].shape[-1])
	with torch.no_grad():
		z = agent.model.encode(obs.to(dev), None).cpu()
	pr = fit_probe(z, obs)
	names = obs_names(args.task, obs.shape[-1])
	perm = torch.randperm(len(z), generator=torch.Generator().manual_seed(1))
	bank, bank_holdout = z[perm[len(z) // 10:]], z[perm[:len(z) // 10]]
	print(f"probe fitted on {len(z)} real latents; held-out R^2 median {float(pr['r2'].median()):.3f}")

	# 2. rollouts vs simulator
	planner = AuditedPlanner(agent, PlannerConfig(horizon=args.horizon, score="standard"))
	starts, det_err = collect_rollouts(agent, planner, args.task, args.episodes, args.seed_start, args.every,
	                                   args.horizon, args.n_cand)
	res = analyze(agent, starts, pr["W"], bank, bank_holdout)
	S = summarize(res, names, pr["r2"], det_err)
	ex = examples(res, names)
	torch.save(dict(res=res, names=names, probe_r2=pr["r2"], summary=S, examples=ex, args=vars(args)), out / "data.pt")
	(out / "summary.json").write_text(json.dumps(dict(summary=S, examples=ex), indent=1))
	write_report(out, S, ex, args)
	print((out / "report.md").read_text())


if __name__ == "__main__":
	main()
