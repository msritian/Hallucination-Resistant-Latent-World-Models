"""Value-error diagnostic (analysis only; does not change the pre-registered Phase 2 decision).

Question: does the Bellman residual detect imagined-step errors that change *value* (decision-relevant), even where
it misses raw latent errors? Reuses the episodes saved by run_preflight.

For open-loop replays along real actions (same windows and calibration/evaluation split as the preflight):
  latent error   e_{t+1} = ||z_hat_{t+1} - z_{t+1}||
  value error    v_{t+1} = |V(z_hat_{t+1}) - V(z_{t+1})|,  V(z) = mean-head Q(z, mu_pi(z))
  critic noise   delta_real_t = |Q(z_hat_t, a_t) - (r_hat_t + gamma V(z_{t+1}))|   (residual with the REAL next state)

Decomposition behind it: delta_t = |(Q - r_hat - gamma V(z_{t+1})) - gamma (V(z_hat_{t+1}) - V(z_{t+1}))|,
i.e. critic inconsistency on real data plus gamma times the value error of the imagined step.
"""
import argparse
import json
from pathlib import Path

import torch

from src.auditor.signals import pi_mean, q_heads
from src.preflight.analysis import calibrate_all, natural_labels, normalize
from src.preflight.metrics import bootstrap_auroc
from src.preflight.run_preflight import CAL, rollout_signals, windows


def _rank(x: torch.Tensor) -> torch.Tensor:
	x = x.double().flatten()
	order = torch.argsort(x)
	ranks = torch.empty_like(x)
	ranks[order] = torch.arange(1, len(x) + 1, dtype=torch.float64)
	uniq, inv, counts = torch.unique(x[order], return_inverse=True, return_counts=True)
	if len(uniq) < len(x):
		cum = torch.cumsum(counts, 0).double()
		ranks[order] = (cum - (counts.double() - 1) / 2)[inv]
	return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
	ra, rb = _rank(a), _rank(b)
	ra, rb = ra - ra.mean(), rb - rb.mean()
	den = (ra.norm() * rb.norm()).item()
	return float((ra @ rb).item() / den) if den > 0 else float("nan")


def within_step_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
	"""Mean over imagined steps of the Spearman correlation within that step ([H, M] inputs)."""
	vals = [spearman(a[t], b[t]) for t in range(a.shape[0])]
	vals = [v for v in vals if v == v]
	return sum(vals) / len(vals) if vals else float("nan")


def value(agent, z: torch.Tensor) -> torch.Tensor:
	return q_heads(agent.model, z, pi_mean(agent.model, z), agent.cfg).mean(0).squeeze(-1)


@torch.no_grad()
def _replays(agent, data, episode_ids, H, beta):
	dev = next(agent.model.parameters()).device
	o, a, ep_id = windows(data, H, episode_ids)
	z_real = agent.model.encode(o.to(dev), None)
	sig, err, roll = rollout_signals(agent, z_real, a.to(dev), beta)
	v_real = value(agent, z_real[1:])                                   # [H, M]
	v_err = (roll.v_mean[1:].squeeze(-1) - v_real).abs()
	delta_real = (roll.q_sa - (roll.r_hat + roll.discount * v_real.unsqueeze(-1))).abs().squeeze(-1)
	cpu = lambda x: x.detach().cpu()
	return {k: cpu(v) for k, v in sig.items()}, cpu(err), cpu(v_err), cpu(delta_real), ep_id


def diagnose_model(agent, data: dict, H: int = 12, beta: float = 1.0) -> dict:
	E = data["obs"].shape[0]
	cal_eps, ev_eps = list(range(E // 2)), list(range(E // 2, E))

	sig_c, err_c, verr_c, _, _ = _replays(agent, data, cal_eps, H, beta)
	taus, eps = calibrate_all(sig_c, err_c, CAL)
	v1 = verr_c[0].flatten().float()
	eps_v = dict(eps_v_clean=torch.quantile(v1, CAL["eps_clean_pct"] / 100).item(),
	             eps_v_hall=torch.quantile(v1, CAL["eps_hall_pct"] / 100).item())

	sig, err, v_err, delta_real, ep_id = _replays(agent, data, ev_eps, H, beta)
	s = normalize(sig, taus)
	groups = ep_id.view(1, -1).expand_as(err)
	steps = torch.arange(err.shape[0]).view(-1, 1).expand_as(err)

	lat_lab, lat_keep = natural_labels(err, eps["eps_clean"], eps["eps_hall"])
	val_lab, val_keep = natural_labels(v_err, eps_v["eps_v_clean"], eps_v["eps_v_hall"])

	def strat(x, lab, keep):
		p, lo, hi = bootstrap_auroc(x[keep], lab[keep], n_boot=500, groups=groups[keep], strata=steps[keep])
		return dict(auroc=p, ci95=[lo, hi], n_pos=int(lab[keep].sum()), n_neg=int((~lab[keep]).sum()))

	per_signal = {}
	for n, x in s.items():
		per_signal[n] = dict(
			type4_latent=strat(x, lat_lab, lat_keep),
			type4_value=strat(x, val_lab, val_keep),
			spearman_latent=within_step_spearman(x, err),
			spearman_value=within_step_spearman(x, v_err),
		)

	lat_pos, val_pos = lat_lab & lat_keep, val_lab & val_keep
	raw_delta = sig["A"]
	return dict(
		per_signal=per_signal,
		thresholds=dict(**eps, **eps_v),
		overlap=dict(
			latent_positives=int(lat_pos.sum()), value_positives=int(val_pos.sum()),
			frac_latent_pos_that_change_value=float((lat_pos & (v_err > eps_v["eps_v_hall"])).sum() / max(int(lat_pos.sum()), 1)),
			frac_value_pos_that_are_latent_pos=float((val_pos & (err > eps["eps_hall"])).sum() / max(int(val_pos.sum()), 1)),
			spearman_latent_vs_value_error=within_step_spearman(err, v_err),
		),
		critic_noise=dict(
			median_delta_with_real_next_state=float(delta_real.median()),
			median_delta_on_value_error_positives=float(raw_delta[val_pos].median()) if val_pos.any() else float("nan"),
			median_delta_on_value_error_negatives=float(raw_delta[val_lab.logical_not() & val_keep].median()),
			median_value_error=float(v_err.median()),
		),
	)


def write_report(out: Path, report: dict):
	L = ["# Value-error diagnostic (analysis only; the Phase 2 decision stands)", ""]
	for model, r in report.items():
		o, c, th = r["overlap"], r["critic_noise"], r["thresholds"]
		L += [f"## {model} model", "",
		      f"- Latent-error positives: {o['latent_positives']}; value-error positives: {o['value_positives']}",
		      f"- Fraction of latent-error positives that also change value: **{o['frac_latent_pos_that_change_value']:.2f}**",
		      f"- Fraction of value-error positives that are also latent-error positives: {o['frac_value_pos_that_are_latent_pos']:.2f}",
		      f"- Within-step Spearman(latent error, value error): {o['spearman_latent_vs_value_error']:.2f}",
		      f"- Critic noise: median residual with the REAL next state {c['median_delta_with_real_next_state']:.4f}; "
		      f"median residual on value-error positives {c['median_delta_on_value_error_positives']:.4f} vs negatives "
		      f"{c['median_delta_on_value_error_negatives']:.4f}; median value error {c['median_value_error']:.4f}",
		      f"- Thresholds: latent {th['eps_clean']:.3f}/{th['eps_hall']:.3f}, value {th['eps_v_clean']:.4f}/{th['eps_v_hall']:.4f}", "",
		      "| Signal | AUROC: latent-error labels | AUROC: value-error labels | Spearman w/ latent error | Spearman w/ value error |",
		      "| :-: | :-: | :-: | :-: | :-: |"]
		for n, p in r["per_signal"].items():
			a, b = p["type4_latent"], p["type4_value"]
			L.append(f"| {n} | {a['auroc']:.3f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] | {b['auroc']:.3f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}] "
			         f"| {p['spearman_latent']:.2f} | {p['spearman_value']:.2f} |")
		L.append("")
	(out / "diagnostic.md").write_text("\n".join(L))


def main(argv=None):
	from src.preflight.run_preflight import load_agent

	p = argparse.ArgumentParser()
	p.add_argument("--task", required=True)
	p.add_argument("--grounded_model", required=True)
	p.add_argument("--stock_model", default=None)
	p.add_argument("--episodes_dir", required=True, help="preflight output folder with episodes_<model>.pt")
	p.add_argument("--out", required=True)
	p.add_argument("--horizon", type=int, default=12)
	args = p.parse_args(argv)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	report = {}
	for name, path, run in [("grounded", args.grounded_model, "grounded"), ("stock", args.stock_model, "stock")]:
		if path is None:
			continue
		agent = load_agent(path, run, args.task)
		data = torch.load(Path(args.episodes_dir) / f"episodes_{name}.pt", map_location="cpu")
		report[name] = diagnose_model(agent, data, args.horizon)
		print(json.dumps(report[name]["overlap"], indent=1), json.dumps(report[name]["critic_noise"], indent=1))
		del agent
		torch.cuda.empty_cache()
	(out / "diagnostic.json").write_text(json.dumps(report, indent=1))
	write_report(out, report)
	print((out / "diagnostic.md").read_text())


if __name__ == "__main__":
	main()
