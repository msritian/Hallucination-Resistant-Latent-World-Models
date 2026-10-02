"""Task-relevant view of ``rollout_vs_sim`` results: object positions in centimetres instead of all state dims.

Uses only the saved data.pt (no model, no simulator). The linear probe (latent -> state, fitted on real latents only)
reads the object's position out of imagined latents; only dimensions the probe reads reliably on real states
(held-out R^2 >= 0.9) are used.

	python -m src.tools.rvs_task_view rvs_results/rvs_seed_grounded_stack_s20/rvs/
Outputs: report_task.md, fig_task_drift.png, fig_task_shift.png, fig_task_example_{policy,elite}.png
"""
import sys
from pathlib import Path

import torch

# Object-position dims per task (x, y, z in metres), and a relation that defines success.
TASK_DIMS = {
	"StackCube-v1": dict(obj=["extra.cubeA_pose[0]", "extra.cubeA_pose[1]", "extra.cubeA_pose[2]"],
	                     rel=["extra.cubeA_to_cubeB_pos[0]", "extra.cubeA_to_cubeB_pos[1]", "extra.cubeA_to_cubeB_pos[2]"],
	                     obj_label="cube A", rel_label="cube A to cube B offset"),
	"PickSingleYCB-v1": dict(obj=["extra.obj_pose[0]", "extra.obj_pose[1]", "extra.obj_pose[2]"],
	                         rel=["extra.obj_to_goal_pos[0]", "extra.obj_to_goal_pos[1]", "extra.obj_to_goal_pos[2]"],
	                         obj_label="object", rel_label="object to goal offset"),
}


def load(out: Path):
	D = torch.load(out / "data.pt", weights_only=False)
	task = D["args"]["task"]
	spec = TASK_DIMS[task]
	names, r2 = D["names"], D["probe_r2"]
	idx = lambda keys: [names.index(k) for k in keys if k in names and float(r2[names.index(k)]) >= 0.9]
	return D, task, spec, idx(spec["obj"]), idx(spec["rel"]), r2


def pos_err_cm(d, m, dims, which="state_hat"):
	"""Euclidean error (cm) between probed and true object position, [H, M]."""
	a = d[which][:, :, dims] if m is None else d[which][:, m][:, :, dims]
	b = d["state_true"][:, :, dims] if m is None else d["state_true"][:, m][:, :, dims]
	return 100 * (a - b).norm(dim=-1)


def main(argv=None):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out = Path((argv or sys.argv[1:])[0])
	D, task, spec, obj, rel, r2 = load(out)
	res = D["res"]
	pol, cand = res["policy"], res["candidates"]
	H = pol["r_hat"].shape[0]
	x = list(range(1, H + 1))
	groups = [("agent's own actions", pol, None, "tab:blue"), ("planner candidates: elite", cand, cand["kind"] == 0, "tab:orange"),
	          ("planner candidates: random", cand, cand["kind"] == 1, "tab:red")]
	sel = lambda d, m, k: d[k] if m is None else d[k][:, m]
	L = [f"# Task-relevant view: {task} ({D['args']['run']} model)", "",
	     f"Object-position dims used (probe held-out R^2 >= 0.9): {[D['names'][i] for i in obj]}; "
	     f"relation dims: {[D['names'][i] for i in rel]}.", "",
	     "## Object-position error, imagined vs simulator (median cm)", "",
	     f"Probe error floor = probe applied to the REAL latent ({spec['obj_label']}).", "",
	     "| step | own actions | elite candidates | random candidates | probe floor | NN dist imagined (own) | NN dist real (own) |",
	     "| :-: | :-: | :-: | :-: | :-: | :-: | :-: |"]
	e_pol = pos_err_cm(pol, None, obj)
	e_el, e_rd = pos_err_cm(cand, cand["kind"] == 0, obj), pos_err_cm(cand, cand["kind"] == 1, obj)
	floor = pos_err_cm(pol, None, obj, "state_probe_real")
	for t in range(H):
		if t + 1 in (1, 3, 6, 9, 12, 16, 20, 24):
			L.append(f"| {t + 1} | {e_pol[t].median():.1f} | {e_el[t].median():.1f} | {e_rd[t].median():.1f} | "
			         f"{floor[t].median():.1f} | {pol['nn_dist_imagined'][t].median():.3f} | {pol['nn_dist_real'][t].median():.3f} |")

	# Figure: task-relevant drift + optimism
	fig, axes = plt.subplots(1, 3, figsize=(17, 4))
	for label, d, m, c in groups:
		e = pos_err_cm(d, m, obj)
		axes[0].plot(x, e.median(1).values, color=c, label=label)
		axes[0].fill_between(x, torch.quantile(e, 0.25, 1), torch.quantile(e, 0.75, 1), color=c, alpha=0.15)
		rb = sel(d, m, "return_err")
		axes[1].plot(x, rb.mean(1), color=c, label=label)
		vb = sel(d, m, "v_hat") - sel(d, m, "v_real")
		axes[2].plot(x, vb.mean(1), color=c, label=label)
	axes[0].plot(x, floor.median(1).values, "k--", label="probe error floor (real latents)")
	axes[0].set_title(f"{spec['obj_label']} position error, imagined vs real (cm)")
	axes[1].set_title("imagined minus real return (mean; >0 = too optimistic)")
	axes[2].set_title("critic value: imagined minus real state (mean)")
	for ax in axes:
		ax.set_xlabel("imagined step")
		ax.axhline(0, color="grey", lw=0.5)
		ax.legend(fontsize=8)
	fig.suptitle(f"{task}: model vs simulator from the same state with the same actions")
	fig.tight_layout()
	fig.savefig(out / "fig_task_drift.png", dpi=130)

	# Figure: distribution shift, imagined vs real states at the same steps
	fig, ax = plt.subplots(figsize=(7, 4))
	for label, d, m, c in groups:
		ax.plot(x, sel(d, m, "nn_dist_imagined").median(1).values, color=c, label=f"imagined ({label})")
		ax.plot(x, sel(d, m, "nn_dist_real").median(1).values, color=c, ls="--", label=f"real ({label})")
	ax.set_title("distance to nearest training-episode latent (median)")
	ax.set_xlabel("step")
	ax.legend(fontsize=7)
	fig.tight_layout()
	fig.savefig(out / "fig_task_shift.png", dpi=130)

	# Examples: worst own-action rollout and most over-optimistic elite candidate
	def example(d, m_idx, title, fname):
		zrel = rel[-1] if rel else None
		fig, axes = plt.subplots(1, 4, figsize=(20, 3.6))
		axes[0].plot(x, d["r_hat"][:, m_idx], "o-", label="imagined")
		axes[0].plot(x, d["r_true"][:, m_idx], "s-", label="simulator")
		axes[0].set_title("reward")
		k = obj[-1]
		axes[1].plot(x, 100 * d["state_hat"][:, m_idx, k], "o-", label="imagined (probe)")
		axes[1].plot(x, 100 * d["state_true"][:, m_idx, k], "s-", label="simulator")
		axes[1].set_title(f"{spec['obj_label']} height (cm)")
		if zrel is not None:
			dist_h = 100 * d["state_hat"][:, m_idx][:, rel].norm(dim=-1)
			dist_t = 100 * d["state_true"][:, m_idx][:, rel].norm(dim=-1)
			axes[2].plot(x, dist_h, "o-", label="imagined (probe)")
			axes[2].plot(x, dist_t, "s-", label="simulator")
			axes[2].set_title(f"{spec['rel_label']} distance (cm)")
		axes[3].plot(x, d["residual_A"][:, m_idx], "o-", label="one-step residual")
		axes[3].plot(x, d["residual_anchored"][:, m_idx], "s-", label="anchored residual")
		axes[3].set_title("Bellman residuals (no simulator needed)")
		for ax in axes:
			ax.set_xlabel("imagined step")
			ax.legend(fontsize=7)
		fig.suptitle(title)
		fig.tight_layout()
		fig.savefig(out / fname, dpi=130)
		rows = [f"| {t + 1} | {d['r_hat'][t, m_idx]:.2f} / {d['r_true'][t, m_idx]:.2f} | "
		        f"{100 * d['state_hat'][t, m_idx, k]:.1f} / {100 * d['state_true'][t, m_idx, k]:.1f} | "
		        + (f"{100 * d['state_hat'][t, m_idx][rel].norm():.1f} / {100 * d['state_true'][t, m_idx][rel].norm():.1f} | " if rel else "– | ")
		        + f"{d['residual_A'][t, m_idx]:.2f} | {d['residual_anchored'][t, m_idx]:.2f} |"
		        for t in range(H) if t + 1 in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24)]
		return ["", f"### {title}", "",
		        f"| step | reward imagined / real | {spec['obj_label']} height cm imagined / real | "
		        f"{spec['rel_label']} cm imagined / real | one-step residual | anchored residual |",
		        "| :-: | :-: | :-: | :-: | :-: | :-: |"] + rows

	m_pol = int(pol["return_err"][-1].abs().argmax())
	elite = torch.nonzero(cand["kind"] == 0).flatten()
	m_el = int(elite[cand["return_err"][-1, elite].argmax()])
	L += ["", "## Concrete examples"]
	L += example(pol, m_pol, f"Agent's own actions: largest drift (imagined minus real return {float(pol['return_err'][-1, m_pol]):+.2f})",
	             "fig_task_example_policy.png")
	L += example(cand, m_el, f"Elite planner candidate: most over-optimistic (imagined minus real return {float(cand['return_err'][-1, m_el]):+.2f})",
	             "fig_task_example_elite.png")
	(out / "report_task.md").write_text("\n".join(L) + "\n")
	print((out / "report_task.md").read_text())


if __name__ == "__main__":
	main()
