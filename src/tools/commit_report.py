"""Pool the commitment-study results over robots and test the pre-registered claims
(docs/preregistration_commitment.md).

	python -m src.tools.commit_report results/com_*.tar.gz

C0  the setting makes hallucinations costly: success(k1) - success(k4) > 0 and success(oracle@4) - success(random@4) > 0
    (both 95% CIs above 0). If C0 fails, C1-C3 are reported but the setting gives detection little to buy.
C1  the audit pays off under commitment:   success(LBA@4) - success(random@4) > 0     (95% CI above 0)
C2  better than ensemble gates:            success(LBA@4) - success(X@4) > 0 for X in D, M (mean)
C3  value of detection grows with commitment: [LBA - random]@4 > [LBA - random]@2 (mean)
Differences are paired by episode seed within each robot, averaged within robots, then over robots; CIs resample
episodes within each robot.
"""
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from src.preflight.metrics import stratified_auroc
from src.tools.pool_lambda import pooled_ci

GATES = ("oracle", "LBA", "Aa", "D", "M", "PE", "random")
K_MAX = 8


def load(paths) -> dict:
	out = {}
	for p in map(Path, paths):
		root = p
		if p.suffixes[-2:] == [".tar", ".gz"]:
			root = Path(tempfile.mkdtemp())
			with tarfile.open(p) as tf:
				tf.extractall(root)
		com = next(d for d in [root, *root.rglob("*")] if d.is_dir() and (d / "arms").is_dir())
		name = p.name.replace(".tar.gz", "").replace("com_", "")
		cal = com / "calibration.pt"
		out[name] = dict(arms={f.stem: torch.load(f, weights_only=False) for f in sorted((com / "arms").glob("*.pt"))},
		                 lba=json.loads((com / "lba.json").read_text()),
		                 cal=torch.load(cal, weights_only=False) if cal.exists() else None)
	return out


def success(robot: dict, arm: str) -> dict:
	return {r["seed"]: r["success"] for r in robot["arms"][arm]["rows"]} if arm in robot["arms"] else {}


def paired(robots: dict, a: str, b: str) -> list:
	per = []
	for r in robots.values():
		sa, sb = success(r, a), success(r, b)
		seeds = sorted(set(sa) & set(sb))
		if seeds:
			per.append(torch.tensor([sa[s] - sb[s] for s in seeds]))
	return per


def diff_ci(robots: dict, a: str, b: str) -> dict:
	return pooled_ci(paired(robots, a, b))


def gap_of_gaps(robots: dict, a: str, b: str, c: str, d: str) -> dict:
	"""[a - b] - [c - d], paired by seed within robot."""
	per = []
	for r in robots.values():
		s = [success(r, x) for x in (a, b, c, d)]
		seeds = sorted(set.intersection(*(set(x) for x in s)))
		if seeds:
			per.append(torch.tensor([(s[0][k] - s[1][k]) - (s[2][k] - s[3][k]) for k in seeds]))
	return pooled_ci(per)


def actions_per_plan(robot: dict, arm: str) -> float:
	rows = robot["arms"][arm]["rows"]
	return sum(r["steps"] for r in rows) / sum(r["plan_calls"] for r in rows)


def accuracy(robot: dict, target: str = "return") -> dict:
	"""Step-stratified AUROC of each before-acting score on the calibration plans (first K_MAX-1 steps).

	target "return": real return-error labels (as in detection); "decision": act oracle top quarter vs bottom half."""
	cal = robot["cal"]
	if cal is None or (target == "decision" and "act" not in cal):
		return {}
	if target == "return":
		E = cal["E"][:, :K_MAX - 1]
		pos, neg = E > robot["lba"]["eps_hall"], E <= robot["lba"]["eps_clean"]
	else:
		E = cal["act"]
		pos, neg = E > torch.quantile(E, 0.75), E <= torch.quantile(E, 0.5)
	keep = pos | neg
	steps = torch.arange(E.shape[1]).view(1, -1).expand_as(E)
	out = {}
	for k, s in dict(cal["steps"], PE=cal["PE"]).items():
		s = s[:, :K_MAX - 1]
		out[k] = stratified_auroc(s[keep], pos[keep], steps[keep]) if pos[keep].any() and neg[keep].any() else float("nan")
	return out


def fmt_ci(d: dict) -> str:
	return f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] ({d['models']} robots)"


def main(paths):
	robots = load(paths)
	names = list(robots)
	arms = sorted({a for r in robots.values() for a in r["arms"]},
	              key=lambda a: (a[0] != "k", a.split("@")[-1], a))
	print("# Commitment study: results\n")
	print(f"{len(names)} robots. Success = task success over the same evaluation episodes for every arm; "
	      "actions/plan = executed actions per planning call (compute = 1 / actions/plan).\n")
	print("## 1. Success per arm\n")
	print("| Arm | " + " | ".join(names) + " | Mean | Actions/plan |")
	print("| :-- | " + " | ".join(":-:" for _ in names) + " | :-: | :-: |")
	for a in arms:
		vals = [sum(success(robots[n], a).values()) / max(len(success(robots[n], a)), 1) if a in robots[n]["arms"]
		        else float("nan") for n in names]
		ok = [v for v in vals if v == v]
		apl = [actions_per_plan(robots[n], a) for n in names if a in robots[n]["arms"]]
		print(f"| {a} | " + " | ".join("–" if v != v else f"{v:.2f}" for v in vals) +
		      f" | **{sum(ok) / len(ok):.3f}** | {sum(apl) / len(apl):.2f} |")

	confirm = any("act@4" in r["arms"] for r in robots.values())
	targets = (("return", "real return errors"), ("decision", "decision-relevant errors (act oracle)")) if confirm \
		else (("return", "real plan errors"),)
	for sec, (target, label) in enumerate(targets):
		acc = {n: accuracy(r, target) for n, r in robots.items()}
		keys = [k for k in ("LBA", "DBA", "Aa", "D", "M", "PD", "IR", "PE") if any(k in a for a in acc.values())]
		print(f"\n## 2{'ab'[sec] if confirm else ''}. Accuracy of each gate score on {label} "
		      "(calibration plans, steps 1-7, AUROC)\n")
		print("| Robot | " + " | ".join(keys) + " |")
		print("| :-- | " + " | ".join(":-:" for _ in keys) + " |")
		for n in names:
			print(f"| {n} | " + " | ".join(f"{acc[n][k]:.3f}" if k in acc[n] else "–" for k in keys) + " |")
		means = {k: [acc[n][k] for n in names if k in acc[n] and acc[n][k] == acc[n][k]] for k in keys}
		print("| **Mean** | " + " | ".join(f"**{sum(v) / len(v):.3f}**" if v else "–" for v in means.values()) + " |")
	print("\n(PE uses the real next state, so it is only available while acting, not before.)")
	if confirm:
		confirm_claims(robots)
		return

	print("\n## 3. Pre-registered claims\n")
	c0a, c0b = diff_ci(robots, "k1", "k4"), diff_ci(robots, "oracle@4", "random@4")
	c0 = c0a["ci95"][0] > 0 and c0b["ci95"][0] > 0
	print(f"- **C0** committing hurts: k1 − k4 = {fmt_ci(c0a)}; perfect gating helps: oracle@4 − random@4 = "
	      f"{fmt_ci(c0b)} → {'passed' if c0 else 'failed'}")
	c1 = diff_ci(robots, "LBA@4", "random@4")
	print(f"- **C1** LBA@4 − random@4 = {fmt_ci(c1)} → {'passed' if c1['ci95'][0] > 0 else 'failed'}")
	c2 = {x: diff_ci(robots, "LBA@4", f"{x}@4") for x in ("D", "M")}
	ok2 = all(v["models"] > 0 and v["mean"] > 0 for v in c2.values())
	print("- **C2** " + "; ".join(f"LBA@4 − {x}@4 = {fmt_ci(v)}" for x, v in c2.items()) +
	      f" → {'passed' if ok2 else 'failed'}")
	c3 = gap_of_gaps(robots, "LBA@4", "random@4", "LBA@2", "random@2")
	print(f"- **C3** [LBA − random]@4 − [LBA − random]@2 = {fmt_ci(c3)} → {'passed' if c3['mean'] > 0 else 'failed'}")
	if not c0:
		print("\n  C0 failed: the setting gives detection little to buy; C1-C3 are reported, not interpreted.")

	print("\n## 4. Reported, not claimed\n")
	for a, b in (("LBA@2", "random@2"), ("Aa@4", "random@4"), ("Aa@2", "random@2"), ("LBA@4", "Aa@4"),
	             ("PE@4", "random@4"), ("LBA@4", "PE@4"), ("LBA+PE@4", "PE@4"), ("LBA@4", "k4"), ("oracle@4", "k4"),
	             ("oracle@2", "random@2"), ("k1", "k2"), ("k1", "k8"), ("LBA@4", "k2"), ("LBA@4", "k1")):
		print(f"- {a} − {b}: {fmt_ci(diff_ci(robots, a, b))}")
	for b in ("2", "4"):
		o, l = (diff_ci(robots, f"{x}@{b}", f"random@{b}")["mean"] for x in ("oracle", "LBA"))
		if o == o and o != 0:
			print(f"- Share of the oracle's gain over random captured by the LBA at budget {b}: {l / o:.2f}")


def confirm_claims(robots: dict):
	"""docs/preregistration_commitment_confirm.md: K1-K4."""
	ok = lambda d: d["models"] > 0 and d["ci95"][0] > 0
	print("\n## 3. Pre-registered claims (confirmation)\n")
	k1 = diff_ci(robots, "act@4", "random@4")
	print(f"- **K1** act@4 − random@4 = {fmt_ci(k1)} → {'passed' if ok(k1) else 'failed'}")
	k2 = diff_ci(robots, "act@4", "oracle@4")
	print(f"- **K2** act@4 − oracle@4 = {fmt_ci(k2)} → {'passed' if ok(k2) else 'failed'}")
	k3 = {g: diff_ci(robots, f"{g}@4", "random@4") for g in ("Aa", "PE")}
	print("- **K3** " + "; ".join(f"{g}@4 − random@4 = {fmt_ci(v)}" for g, v in k3.items()) +
	      f" → {'passed' if all(ok(v) for v in k3.values()) else 'failed'}")
	k4a, k4b = diff_ci(robots, "DBA@4", "random@4"), diff_ci(robots, "DBA@4", "LBA@4")
	print(f"- **K4** DBA@4 − random@4 = {fmt_ci(k4a)}; DBA@4 − LBA@4 = {fmt_ci(k4b)} → "
	      f"{'passed' if ok(k4a) and k4b['mean'] > 0 else 'failed'}")
	print("\n## 4. Reported, not claimed\n")
	for a, b in (("reg@4", "random@4"), ("act@4", "reg@4"), ("oracle@4", "random@4"), ("LBA@4", "random@4"),
	             ("D@4", "random@4"), ("M@4", "random@4"), ("DBA@4", "Aa@4"), ("DBA@4", "PE@4"), ("k1", "k4"),
	             ("act@4", "k1")):
		print(f"- {a} − {b}: {fmt_ci(diff_ci(robots, a, b))}")
	if any("IR@4" in r["arms"] for r in robots.values()):
		flip_claims(robots)


def flip_claims(robots: dict):
	"""docs/preregistration_decision_flip.md: F1-F3."""
	ok = lambda d: d["models"] > 0 and d["ci95"][0] > 0
	print("\n## 5. Pre-registered claims (decision-flip signals)\n")
	f1 = diff_ci(robots, "IR@4", "random@4")
	print(f"- **F1** IR@4 − random@4 = {fmt_ci(f1)} → {'passed' if ok(f1) else 'failed'}")
	f2 = diff_ci(robots, "PD@4", "D@4")
	print(f"- **F2** PD@4 − D@4 = {fmt_ci(f2)} → {'passed' if ok(f2) else 'failed'}")
	f3 = diff_ci(robots, "IR@4", "LBA@4")
	print(f"- **F3** IR@4 − LBA@4 = {fmt_ci(f3)} → {'passed' if f3['models'] > 0 and f3['mean'] > 0 else 'failed'}")
	print("\nReported, not claimed:\n")
	for a, b in (("PD@4", "random@4"), ("IR@4", "DBA@4"), ("PD@4", "DBA@4"), ("IR@4", "Aa@4"), ("IR@4", "PE@4"),
	             ("IR@4", "act@4"), ("PD@4", "act@4")):
		print(f"- {a} − {b}: {fmt_ci(diff_ci(robots, a, b))}")


if __name__ == "__main__":
	main(sys.argv[1:])
