"""Pool the detector-in-the-planner results over robots and test the pre-registered claims
(docs/preregistration_detector_planning.md).

	python -m src.tools.detplan_report results/plan_*.tar.gz [--monitor results/mon_*.tar.gz] > docs/detplan_results.md

P1  audit-weighted lambda beats the constant lambda:   LBAlam_H12 - L08_H12 > 0                 (95% CI above 0)
P2  the audit's per-candidate information matters:      LBAlam_H12 - LBAlam_H12_shuffled > 0     (95% CI above 0)
P3  dropping suspicious candidates beats the base:      L08_H12_drop25 - L08_H12 > 0             (95% CI above 0)
P4  ... and beats dropping at random:                   L08_H12_drop25 - L08_H12_dropRandom25 > 0 (95% CI above 0)
With --monitor, the standard H3 / H12 arms of the warning-light run (same robots and seeds) are added for context.
"""
import argparse
import math
import tarfile
import tempfile
from pathlib import Path

import torch

from src.tools.pool_lambda import pooled_ci

CLAIMS = [("P1", "LBAlam_H12", "L08_H12"), ("P2", "LBAlam_H12", "LBAlam_H12_shuffled"),
          ("P3", "L08_H12_drop25", "L08_H12"), ("P4", "L08_H12_drop25", "L08_H12_dropRandom25")]
EXPLORATORY = [("LBAlam_pure_H12", "L08_H12"), ("LBAlam_H24", "L08_H24"), ("L08_H24", "L08_H12"),
               ("LBAlam_H24", "LBAlam_H12"), ("LBAlam_H12", "L08_H3"), ("LBAlam_H12", "standard_H3"),
               ("LBAlam_H12", "standard_H12"), ("L08_H12", "standard_H3")]


def _root(p: Path, marker: str) -> Path:
	if p.suffixes[-2:] == [".tar", ".gz"]:
		d = Path(tempfile.mkdtemp())
		with tarfile.open(p) as tf:
			tf.extractall(d)
		p = d
	return next(x for x in [p, *p.rglob("*")] if x.is_dir() and (x / marker).is_dir())


def robot_key(path: Path) -> str:
	name = path.name.replace(".tar.gz", "")
	for prefix in ("plan_", "mon_"):
		name = name.replace(prefix, "", 1) if name.startswith(prefix) else name
	return name


def load(paths, monitor_paths=()) -> dict:
	"""{robot: {arm: {seed: success}}}; warning-light H3/H12 arms renamed standard_H3/standard_H12."""
	out = {}
	for p in map(Path, paths):
		root = _root(p, "arms")
		out.setdefault(robot_key(p), {}).update({f.stem: {r["seed"]: r["success"] for r in torch.load(f, weights_only=False)["rows"]}
		                                         for f in (root / "arms").glob("*.pt")})
	for p in map(Path, monitor_paths):
		root = _root(p, "arms")
		for arm, new in (("H3", "standard_H3"), ("H12", "standard_H12")):
			f = root / "arms" / f"{arm}.pt"
			if f.exists() and robot_key(p) in out:
				out[robot_key(p)][new] = {r["seed"]: r["success"] for r in torch.load(f, weights_only=False)["rows"]}
	return out


def diff(robots: dict, a: str, b: str) -> dict:
	per = []
	for arms in robots.values():
		if a in arms and b in arms:
			seeds = sorted(set(arms[a]) & set(arms[b]))
			per.append(torch.tensor([arms[a][s] - arms[b][s] for s in seeds]))
	return pooled_ci(per)


def run(robots: dict) -> str:
	ci = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] ({d['models']} robots)"
	arms = sorted({a for r in robots.values() for a in r})
	L = ["# Detector in the planner: results", "", f"{len(robots)} robots, success rate on the same evaluation seeds.", "",
	     "## Success rate per arm", "", "| Arm | " + " | ".join(robots) + " | Mean |", "| :-- |" + " :-: |" * (len(robots) + 1)]
	for a in arms:
		vals = [sum(r[a].values()) / len(r[a]) if a in r else float("nan") for r in robots.values()]
		ok = [v for v in vals if not math.isnan(v)]
		mean = f"**{sum(ok) / len(ok):.3f}**" if ok else "–"
		L.append(f"| {a} | " + " | ".join("–" if math.isnan(v) else f"{v:.2f}" for v in vals) + f" | {mean} |")
	L += ["", "## Pre-registered claims", ""]
	for name, a, b in CLAIMS:
		d = diff(robots, a, b)
		L.append(f"- **{name}** {a} − {b}: {ci(d)} → {'passed' if d['models'] and d['ci95'][0] > 0 else 'failed'}")
	L += ["", "## Exploratory (reported, not claimed)", ""]
	for a, b in EXPLORATORY:
		d = diff(robots, a, b)
		if d["models"]:
			L.append(f"- {a} − {b}: {ci(d)}")
	return "\n".join(L)


def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("paths", nargs="+")
	p.add_argument("--monitor", nargs="*", default=[])
	a = p.parse_args(argv)
	print(run(load(a.paths, a.monitor)))


if __name__ == "__main__":
	main()
