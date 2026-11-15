"""Powered value-expansion study (docs/preregistration_value_expansion_powered.md): criteria P1-P5.

	python -m src.tools.ve_report <dir>
<dir> holds one folder per run, named vep_<arm>_<task>_h<h>_s<seed>, each with run/eval.csv and run/final_eval.json.
Learning speed = mean evaluation success at steps >= 50k (as in the pilot); final = success over 100 episodes.
Comparisons are paired by seed, with a 95% bootstrap CI over seed pairs. Partial data is reported as it stands
(an interim look); a criterion is only judged when all of its pre-registered pairs exist.
"""
import csv
import json
import re
import sys
from pathlib import Path

import torch

START = 50_000
CELLS = {"A": ("stack", 5), "B": ("pick", 5), "C": ("stack", 10)}
ARMS = ("none", "const", "D", "Mt", "lba")
SEEDS = range(80, 85)


def load(root: Path) -> dict:
	res = {}
	for d in sorted(root.glob("vep_*")):
		m = re.match(r"vep_(.+)_(stack|pick)_h(\d+)_s(\d+)$", d.name)
		f = d / "run" / "final_eval.json"
		if not m or not f.exists():
			continue
		arm, task, h, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
		rows = list(csv.DictReader(open(d / "run" / "eval.csv")))
		late = [float(r["episode_success"]) for r in rows if float(r["step"]) >= START]
		res[(arm, task, h, seed)] = dict(speed=sum(late) / len(late), final=json.load(open(f))["episode_success"])
	return res


def boot_ci(x: torch.Tensor, n: int = 2000, seed: int = 0):
	g = torch.Generator().manual_seed(seed)
	b = torch.stack([x[torch.randint(len(x), (len(x),), generator=g)].mean() for _ in range(n)])
	return float(x.mean()), float(b.quantile(0.025)), float(b.quantile(0.975))


def pairs(res, a, b, cells, metric):
	out = [res[(a, t, h, s)][metric] - res[(b, t, h, s)][metric]
	       for c in cells for t, h in [CELLS[c]] for s in SEEDS if (a, t, h, s) in res and (b, t, h, s) in res]
	return torch.tensor(out)


def compare(res, a, b, cells, metric="speed", need=None):
	x = pairs(res, a, b, cells, metric)
	need = need or 5 * len(cells)
	if len(x) == 0:
		return f"{a} - {b} ({'+'.join(cells)}, {metric}): no pairs yet", None
	m, lo, hi = boot_ci(x)
	tag = "" if len(x) == need else f"  [interim: {len(x)}/{need} pairs]"
	return (f"{a} - {b} ({'+'.join(cells)}, {metric}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}], {a} higher on "
	        f"{int((x > 0).sum())}/{len(x)}{tag}"), (lo if len(x) == need else None)


def main(root):
	res = load(Path(root))
	for c, (t, h) in CELLS.items():
		print(f"\nCell {c}: {t} h={h}   mean over seeds (n runs)   speed | final")
		for arm in ARMS:
			r = [res[(arm, t, h, s)] for s in SEEDS if (arm, t, h, s) in res]
			if r:
				sp = sum(x["speed"] for x in r) / len(r)
				fi = sum(x["final"] for x in r) / len(r)
				print(f"  {arm:6s} ({len(r)})  {sp:.3f} | {fi:.3f}   seeds: " +
				      " ".join(f"{res[(arm, t, h, s)]['speed']:.2f}" if (arm, t, h, s) in res else " -- " for s in SEEDS))

	print("\nPre-registered criteria (learning speed; a verdict only when all pairs exist)")
	for name, a, b, cells in (("P1", "lba", "D", "AB"), ("P2", "lba", "none", "AB"), ("P3", "lba", "const", "AB"),
	                          ("P5", "lba", "Mt", "AB")):
		line, lo = compare(res, a, b, cells)
		print(f"  {name} {line}" + ("" if lo is None else f" -> {'PASS' if lo > 0 else 'FAIL'}"))
	lc, loc = compare(res, "lba", "D", "C")
	la, _ = compare(res, "lba", "D", "A")
	gc, ga = pairs(res, "lba", "D", "C", "speed"), pairs(res, "lba", "D", "A", "speed")
	verdict = "" if loc is None or len(ga) < 5 else \
		f" -> {'PASS' if loc > 0 and gc.mean() > ga.mean() else 'FAIL'}"
	print(f"  P4 {lc}{verdict}\n     (compare with cell A: {la})")
	print("\nSecondary: final success")
	for a, b, cells in (("lba", "D", "AB"), ("lba", "none", "AB"), ("lba", "const", "AB"), ("lba", "Mt", "AB"), ("lba", "D", "C")):
		print("  " + compare(res, a, b, cells, "final")[0])
	print("\nPer cell (speed)")
	for c in CELLS:
		for b in ("none", "const", "D", "Mt"):
			line, _ = compare(res, "lba", b, c)
			if "no pairs" not in line:
				print("  " + line)


if __name__ == "__main__":
	main(sys.argv[1])
