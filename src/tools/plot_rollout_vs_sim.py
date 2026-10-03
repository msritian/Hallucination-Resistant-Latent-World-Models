"""Plots for ``rollout_vs_sim`` results (run locally; needs matplotlib).

	python -m src.tools.plot_rollout_vs_sim rvs_stack/          -> rvs_stack/fig_*.png
"""
import sys
from pathlib import Path

import torch


def band(ax, x, y, label, color):
	"""Median line with an interquartile band; y is [H, M]."""
	med = y.float().median(1).values
	lo, hi = torch.quantile(y.float(), 0.25, dim=1), torch.quantile(y.float(), 0.75, dim=1)
	ax.plot(x, med, color=color, label=label)
	ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def main(argv=None):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out = Path((argv or sys.argv[1:])[0])
	D = torch.load(out / "data.pt", weights_only=False)
	res, names, args = D["res"], D["names"], D["args"]
	pol, cand = res["policy"], res["candidates"]
	H = pol["r_hat"].shape[0]
	x = list(range(1, H + 1))
	groups = [("policy actions", pol, None, "tab:blue"), ("planner candidates: elite", cand, cand["kind"] == 0, "tab:orange"),
	          ("planner candidates: random", cand, cand["kind"] == 1, "tab:red")]
	sel = lambda d, m, k: d[k] if m is None else d[k][:, m]

	# 1. How far imagination drifts from the simulator, per step
	fig, axes = plt.subplots(1, 4, figsize=(20, 4))
	panels = [("|imagined - real return|", lambda d, m: sel(d, m, "return_err").abs()),
	          ("latent error ||z_hat - enc(s)||", lambda d, m: sel(d, m, "latent_err")),
	          ("state error (probe of imagined latent)", lambda d, m: (sel(d, m, "state_hat") - sel(d, m, "state_true")).norm(dim=-1)),
	          ("value: imagined - real state (bias)", lambda d, m: sel(d, m, "v_hat") - sel(d, m, "v_real"))]
	for ax, (title, fn) in zip(axes, panels):
		for label, d, m, c in groups:
			band(ax, x, fn(d, m), label, c)
		if "probe" in title:
			ax.plot(x, (pol["state_probe_real"] - pol["state_true"]).norm(dim=-1).median(1).values, "k--", label="probe of REAL latent (probe error floor)")
		ax.set_title(title)
		ax.set_xlabel("imagined step")
	axes[0].legend(fontsize=8)
	axes[2].legend(fontsize=8)
	fig.suptitle(f"{args['task']} ({args['run']}): model vs simulator, same start state and actions (median, IQR)")
	fig.tight_layout()
	fig.savefig(out / "fig_drift.png", dpi=130)

	# 2. Distribution shift: imagined latents leave the region of real latents
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	for label, d, m, c in groups:
		band(axes[0], x, sel(d, m, "nn_dist_imagined"), label, c)
	band(axes[0], x, pol["nn_dist_real"], "real states (same steps)", "tab:green")
	axes[0].axhline(float(res["nn_dist_heldout_real"].median()), color="k", ls="--", label="held-out real states")
	axes[0].set_title("distance to nearest REAL latent")
	axes[0].set_xlabel("imagined step")
	axes[0].legend(fontsize=8)
	axes[1].hist(res["nn_dist_heldout_real"].numpy(), bins=50, alpha=0.5, density=True, label="held-out real")
	axes[1].hist(pol["nn_dist_imagined"][-1].numpy(), bins=50, alpha=0.5, density=True, label=f"imagined, step {H} (policy actions)")
	axes[1].hist(cand["nn_dist_imagined"][-1].numpy(), bins=50, alpha=0.5, density=True, label=f"imagined, step {H} (candidates)")
	axes[1].set_title("distribution of nearest-real-latent distance")
	axes[1].legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(out / "fig_distribution_shift.png", dpi=130)

	# 3. Concrete examples
	for i, e in enumerate(D["examples"]):
		m = e["rollout"]
		dims = [names.index(k) for k in e["diverging_dims"][:2]]
		fig, axes = plt.subplots(1, 4, figsize=(20, 3.6))
		axes[0].plot(x, pol["r_hat"][:, m], "o-", label="imagined")
		axes[0].plot(x, pol["r_true"][:, m], "s-", label="simulator")
		axes[0].set_title("reward")
		axes[1].plot(x, pol["v_hat"][:, m], "o-", label="critic on imagined state")
		axes[1].plot(x, pol["v_real"][:, m], "s-", label="critic on real state")
		axes[1].set_title("value")
		for k, style in zip(dims, ("-", "--")):
			axes[2].plot(x, pol["state_hat"][:, m, k], "o" + style, label=f"{names[k]} imagined (probe)")
			axes[2].plot(x, pol["state_true"][:, m, k], "s" + style, label=f"{names[k]} simulator")
		axes[2].set_title("most diverging state dims")
		axes[3].plot(x, pol["residual_A"][:, m], "o-", label="one-step residual")
		axes[3].plot(x, pol["residual_anchored"][:, m], "s-", label="anchored residual")
		axes[3].set_title("Bellman residuals (no simulator needed)")
		for ax in axes:
			ax.set_xlabel("imagined step")
			ax.legend(fontsize=7)
		fig.suptitle(f"Example {i + 1}: imagined minus real return after {H} steps = {e['final_return_err']:+.2f}")
		fig.tight_layout()
		fig.savefig(out / f"fig_example_{i + 1}.png", dpi=130)
	print(f"wrote plots to {out}")


if __name__ == "__main__":
	main()
