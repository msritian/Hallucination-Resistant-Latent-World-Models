"""Pool the run-time warning-light results over robots and test the pre-registered claims
(docs/preregistration_monitor.md).

	python -m src.tools.monitor_report results/mon_*.tar.gz > docs/monitor_results.md

W1  accuracy on the plans the robot acts on: mean step-stratified AUROC of the LBA above each raw monitor.
W2  acting on the warning helps:   success(H12+LBA@0.25) - success(H12) > 0          (95% CI above 0)
W3  the detector matters:          success(H12+LBA@0.25) - success(H12+random@0.25) > 0   (95% CI above 0)
W4  better than other monitors:    success(H12+LBA@0.25) - success(H12+X@0.25) > 0 for X in A, B, E, D, M (mean)
Differences are paired by episode seed within each robot, averaged within robots, then over robots; CIs resample
episodes within each robot.
"""
import json
import math
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from src.preflight.metrics import stratified_auroc
from src.tools.pool_lambda import pooled_ci

MONITORS = ("LBA", "A", "B", "E", "D", "M")
MID = "0.25"


def load(paths) -> dict:
	"""{robot: dict(arms={name: dict(rows, rec)}, lba=..., thresholds=...)} from result dirs or tarballs."""
	out = {}
	for p in map(Path, paths):
		root = p
		if p.suffixes[-2:] == [".tar", ".gz"]:
			root = Path(tempfile.mkdtemp())
			with tarfile.open(p) as tf:
				tf.extractall(root)
		mon = next(d for d in [root, *root.rglob("*")] if d.is_dir() and (d / "arms").is_dir())
		name = p.name.replace(".tar.gz", "").replace("mon_", "")
		out[name] = dict(arms={f.stem: torch.load(f, weights_only=False) for f in sorted((mon / "arms").glob("*.pt"))},
		                 lba=json.loads((mon / "lba.json").read_text()),
		                 thresholds=json.loads((mon / "thresholds.json").read_text()))
	return out


def fmt(v):
	return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def success(robot: dict, arm: str) -> dict:
	return {r["seed"]: r["success"] for r in robot["arms"][arm]["rows"]} if arm in robot["arms"] else {}


def diff_ci(robots: dict, a: str, b: str) -> dict:
	per = []
	for r in robots.values():
		sa, sb = success(r, a), success(r, b)
		seeds = sorted(set(sa) & set(sb))
		if seeds:
			per.append(torch.tensor([sa[s] - sb[s] for s in seeds]))
	return pooled_ci(per)


def accuracy(robot: dict, exact_only: bool = False) -> dict:
	"""Step-stratified AUROC of each monitor's per-step score for real plan errors (H12 arm, simulator replays).

	``exact_only``: only decisions where the simulator snapshot reproduced the real step exactly (error <= 1e-4)."""
	rec = robot["arms"]["H12"]["rec"]
	E = rec["E"]
	lba = robot["lba"]
	pos, neg = E > lba["eps_hall"], E <= lba["eps_clean"]
	keep = pos | neg
	if exact_only and len(rec["det_err"]) == E.shape[0]:
		keep = keep & (rec["det_err"] <= 1e-4).view(-1, 1)
	steps = torch.arange(E.shape[1]).view(1, -1).expand_as(E)
	out = {}
	for k in MONITORS:
		if k in rec["steps"]:
			s = rec["steps"][k]
			out[k] = stratified_auroc(s[keep], pos[keep], steps[keep]) if pos[keep].any() and neg[keep].any() else float("nan")
	out["positives"] = int(pos.sum())
	out["decisions"] = int(E.shape[0])
	return out


def run(robots: dict) -> str:
	L = ["# Run-time warning light: results", "",
	     f"{len(robots)} robots. Success = task success rate over the same evaluation episodes for every arm.", ""]

	# W1 accuracy
	acc = {n: accuracy(r) for n, r in robots.items()}
	L += ["## 1. Accuracy on the plans the robot acts on (AUROC, simulator-judged)", "",
	      "| Robot | " + " | ".join(MONITORS) + " | hallucinated steps / decisions |", "| :-- |" + " :-: |" * (len(MONITORS) + 1)]
	for n, a in acc.items():
		L.append(f"| {n} | " + " | ".join(fmt(a.get(k)) for k in MONITORS) + f" | {a['positives']} / {a['decisions']} |")
	means = {k: [a[k] for a in acc.values() if k in a and not math.isnan(a[k])] for k in MONITORS}
	means = {k: sum(v) / len(v) if v else float("nan") for k, v in means.items()}
	L.append("| **Mean** | " + " | ".join(f"**{fmt(means[k])}**" for k in MONITORS) + " | |")
	rivals = [k for k in MONITORS[1:] if not math.isnan(means[k])]
	w1 = not math.isnan(means["LBA"]) and bool(rivals) and all(means["LBA"] > means[k] for k in rivals)
	L += ["", f"**W1 {'passed' if w1 else 'failed'}:** LBA mean AUROC above every raw monitor.", ""]
	# amendment (2026-10-03, before the full run): the snapshot reproduced the real step only approximately on 0-3% of
	# smoke-test steps, so W1 is also reported on exactly reproduced decisions only.
	acc_x = {n: accuracy(r, exact_only=True) for n, r in robots.items()}
	mx = {k: [a[k] for a in acc_x.values() if k in a and not math.isnan(a[k])] for k in MONITORS}
	mx = {k: sum(v) / len(v) if v else float("nan") for k, v in mx.items()}
	share = [float((r["arms"]["H12"]["rec"]["det_err"] > 1e-4).float().mean()) for r in robots.values()
	         if len(r["arms"]["H12"]["rec"]["det_err"])]
	L += ["W1 on exactly reproduced decisions only (mean AUROC): " + ", ".join(f"{k} {fmt(mx[k])}" for k in MONITORS) +
	      (f". Share of decisions not reproduced exactly: {max(share):.3f} at most." if share else "."), ""]

	# success table
	arms = sorted({a for r in robots.values() for a in r["arms"]})
	L += ["## 2. Success rate per arm", "", "| Arm | " + " | ".join(robots) + " | Mean | Warning rate |",
	      "| :-- |" + " :-: |" * (len(robots) + 2)]
	for arm in arms:
		vals, rates = [], []
		for r in robots.values():
			rows = r["arms"].get(arm, {}).get("rows")
			vals.append(sum(x["success"] for x in rows) / len(rows) if rows else float("nan"))
			if rows:
				rates.append(sum(x["warning_rate"] for x in rows) / len(rows))
		ok = [v for v in vals if not math.isnan(v)]
		L.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + f" | **{fmt(sum(ok) / len(ok) if ok else float('nan'))}** | "
		         f"{fmt(sum(rates) / len(rates) if rates else float('nan'))} |")
	L.append("")

	# claims
	lba = f"H12+LBA@{MID}"
	w2, w3 = diff_ci(robots, lba, "H12"), diff_ci(robots, lba, f"H12+random@{MID}")
	w4 = {k: diff_ci(robots, lba, f"H12+{k}@{MID}") for k in MONITORS[1:]}
	ci = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] ({d['models']} robots)"
	L += ["## 3. Pre-registered claims", "",
	      f"- **W2** LBA warning − no warning: {ci(w2)} → {'passed' if w2['ci95'][0] > 0 else 'failed'}",
	      f"- **W3** LBA warning − random warning (same rate): {ci(w3)} → {'passed' if w3['ci95'][0] > 0 else 'failed'}"]
	w4_ok = all(d["mean"] > 0 for d in w4.values() if d["models"])
	L += [f"- **W4** LBA − other monitors (same rate): " + "; ".join(f"{k} {ci(d)}" for k, d in w4.items() if d["models"]) +
	      f" → {'passed' if w4_ok else 'failed'}", ""]

	# exploratory
	L += ["## 4. Exploratory (reported, not claimed)", ""]
	for a, b in [(lba, "H3"), (lba, "L08_H12"), ("H3", "H12"), (f"H12+oracle@{MID}", "H12")]:
		L.append(f"- {a} − {b}: {ci(diff_ci(robots, a, b))}")
	for rate in ("0.1", "0.5"):
		L.append(f"- H12+LBA@{rate} − H12+random@{rate}: {ci(diff_ci(robots, f'H12+LBA@{rate}', f'H12+random@{rate}'))}")
	o = diff_ci(robots, f"H12+oracle@{MID}", "H12")["mean"]
	if o and not math.isnan(o) and abs(o) > 1e-9:
		L.append(f"- Share of the oracle's gain captured by the LBA (at {MID}): {w2['mean'] / o:.2f}")
	det = [float(r["arms"]["H12"]["rec"]["det_err"].max()) for r in robots.values() if len(r["arms"]["H12"]["rec"]["det_err"])]
	if det:
		L.append(f"- Simulator snapshot reproducibility (max obs error): {max(det):.2e}")
	return "\n".join(L)


def main(argv=None):
	print(run(load(argv or sys.argv[1:])))


if __name__ == "__main__":
	main()
