"""Report for the DMC commitment study (docs/preregistration_dmc_commit.md): D1-D4, per-task scores, accuracy.

Usage: python -m src.tools.dmc_commit_report results/dcm_*.tar.gz   (score = return / 1000)"""
import sys

import torch

from src.commit_dmc import ARMS, GATES, L
from src.preflight.metrics import stratified_auroc
from src.tools.commit_report import actions_per_plan, diff_ci, fmt_ci, load, success


def accuracy(cal: dict, target: str) -> dict:
	E = cal["E" if target == "return" else "act"][:, :L]
	valid = E > 0                                            # zero = beyond the episode's end
	hi, md = torch.quantile(E[valid], 0.75), torch.quantile(E[valid], 0.5)
	pos, neg = (E > hi) & valid, (E <= md) & valid
	keep = pos | neg
	steps = torch.arange(L).view(1, -1).expand_as(E)
	return {g: stratified_auroc(cal[g][:, :L][keep], pos[keep], steps[keep]) for g in (*GATES, "PE")}


def main(paths):
	tasks = load(paths)
	names = list(tasks)
	short = [n.replace("dcm_", "") for n in names]
	arms = [a for a, _, _ in ARMS]
	print(f"# DMC commitment study: results\n\n{len(names)} tasks, released TD-MPC2 mt30-48M. Score = return / 1000.\n")
	print("## 1. Score per arm\n")
	print("| Arm | " + " | ".join(short) + " | Mean | Actions/plan |")
	print("| :-- | " + " | ".join(":-:" for _ in names) + " | :-: | :-: |")
	for a in arms:
		v = [sum(success(tasks[n], a).values()) / len(success(tasks[n], a)) for n in names]
		apl = sum(actions_per_plan(tasks[n], a) for n in names) / len(names)
		print(f"| {a} | " + " | ".join(f"{x:.3f}" for x in v) + f" | **{sum(v) / len(v):.3f}** | {apl:.2f} |")

	ok = lambda d: d["models"] > 0 and d["ci95"][0] > 0
	print("\n## 2. Pre-registered claims\n")
	for tag, a, b in (("D1", "IR@4", "random@4"), ("D2", "act@4", "oracle@4"), ("D3", "IR@4", "ENT@4"), ("D4", "IR@4", "k4")):
		d = diff_ci(tasks, a, b)
		print(f"- **{tag}** {a} − {b} = {fmt_ci(d)} → {'passed' if ok(d) else 'failed'}")
	print("\n## 3. Reported, not claimed\n")
	pairs = [("k1", "k4"), ("k1", "k2")] + [(f"{g}@4", "random@4") for g in ("act", "reg", "oracle", "ENT", "B", "FI", "RS", "PE")]
	pairs += [(f"IR@4", f"{g}@4") for g in ("act", "reg", "B", "FI", "RS", "PE")]
	pairs += [("IR@2", "random@2"), ("IR@2", "k2"), ("IR@2", "k1"), ("random@4", "k4")]
	for a, b in pairs:
		print(f"- {a} − {b}: {fmt_ci(diff_ci(tasks, a, b))}")
	for target in ("return", "decision"):
		acc = {n: accuracy(tasks[n]["cal"], target) for n in names}
		keys = list(acc[names[0]])
		print(f"\n## 4{'ab'[target == 'decision']}. Accuracy on {target} labels (calibration plans, AUROC)\n")
		print("| Task | " + " | ".join(keys) + " |")
		print("| :-- | " + " | ".join(":-:" for _ in keys) + " |")
		for n, s in zip(names, short):
			print(f"| {s} | " + " | ".join(f"{acc[n][k]:.3f}" for k in keys) + " |")
		print("| **Mean** | " + " | ".join(f"**{sum(acc[n][k] for n in names) / len(names):.3f}**" for k in keys) + " |")


if __name__ == "__main__":
	main(sys.argv[1:])
