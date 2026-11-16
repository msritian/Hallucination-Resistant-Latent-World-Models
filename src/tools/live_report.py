"""Live IR study report (docs/preregistration_live_ir.md) from lv_* / lvc_* / lvs_* archives.

	python -m src.tools.live_report N results/lv_*.tar.gz      # confirmation set: claims L1-L3
	python -m src.tools.live_report check results/lvc_*.tar.gz # frontier robots: IRl vs IRs (should be about 0)
	python -m src.tools.live_report stock results/lvs_*.tar.gz # stock TD-MPC2 robots (reported)
"""
import sys

import numpy as np

from src.tools.robust_report import load

B = 5000


def main(part: str, paths: list):
	rng = np.random.default_rng(0)
	S = {}
	for p in paths:
		arms, _ = load(p)
		seeds = sorted(r["seed"] for r in arms["k1"])
		S[p.split("/")[-1].replace(".tar.gz", "")] = {
			a: np.array([{r["seed"]: r["success"] for r in rows}[s] for s in seeds]) for a, rows in arms.items()}
	names = sorted(S)
	want = ["k1", "k2", "k4", "k6", "random@4", "IR@2", "IR@4", "IR@6", "IRs@4", "IRs@6", "PE@2", "PE@4", "PE@6",
	        "IRl@2", "IRl@4", "IRl@6"]
	cols = [a for a in want if all(a in S[n] for n in names)]
	print(f"part {part}: {len(names)} robots\n")
	print(f"{'robot':30s} " + " ".join(f"{a:>7s}" for a in cols))
	for n in names:
		print(f"{n:30s} " + " ".join(f"{S[n][a].mean():7.3f}" for a in cols))
	print(f"{'mean':30s} " + " ".join(f"{np.mean([S[n][a].mean() for n in names]):7.3f}" for a in cols))

	def show(a, b, tag=""):
		if a not in cols or b not in cols:
			return
		d = [S[n][a] - S[n][b] for n in names]
		full = np.mean([v.mean() for v in d])
		bs = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in d]) for _ in range(B)]
		lo, hi = np.percentile(bs, [2.5, 97.5])
		w, l = sum(v.mean() > 0 for v in d), sum(v.mean() < 0 for v in d)
		verdict = ("PASS" if lo > 0 else "FAIL") if tag else ""
		print(f"{tag:4s}{a + ' - ' + b:22s} {full:+.3f} [{lo:+.3f}, {hi:+.3f}]  better {w}, worse {l}  {verdict}")

	print("\nclaims:" if part == "N" else "\ncomparisons:")
	claimed = part in ("N", "stock")
	show("IRl@4", "PE@4", "L1 " if claimed else "")
	show("IRl@6", "PE@6", "L2 " if claimed else "")
	show("IRl@4", "IR@4", "L3 " if claimed else "")
	print("\nreported:")
	for a, b in [("IRl@2", "k1"), ("IRl@2", "k2"), ("IRl@2", "PE@2"), ("IRl@2", "IR@2"), ("IRl@4", "k4"),
	             ("IRl@6", "k6"), ("IRl@4", "random@4"), ("IRl@4", "IRs@4"), ("IRl@6", "IRs@6"), ("PE@2", "k2"),
	             ("PE@6", "k6"), ("k1", "k4")]:
		show(a, b)


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2:])
