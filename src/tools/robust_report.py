"""Robust imagined regret study report (src/robust_regret.py) from the rr_*.tar.gz archives.

	python -m src.tools.robust_report results/rr_*.tar.gz

Closed loop: success of every gate at budgets 4 and 6 next to the earlier arms of the same robots (same seeds).
Offline (calibration plans): how well each score ranks the critic-regret oracle reg (AUROC top quarter vs bottom half,
pooled over steps 1..7, per robot then averaged) and the empirical coverage of the radius maps.
"""
import io
import sys
import tarfile

import numpy as np
import torch

ROB = ("IRs", "RIRo", "RIR", "RIRt", "RIRc")
B = 5000


def load(path: str):
	arms, cal = {}, None
	with tarfile.open(path) as tf:
		for m in tf.getmembers():
			if not m.isfile():
				continue
			if "/arms/" in m.name and m.name.endswith(".pt"):
				arms[m.name.split("/")[-1][:-3]] = torch.load(io.BytesIO(tf.extractfile(m).read()), weights_only=False)["rows"]
			elif m.name.endswith("calibration_robust.pt"):
				cal = torch.load(io.BytesIO(tf.extractfile(m).read()), weights_only=False)
	return arms, cal


def auroc(score, label):
	pos, neg = score[label], score[~label]
	if len(pos) == 0 or len(neg) == 0:
		return float("nan")
	r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
	return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def main(paths):
	rng = np.random.default_rng(0)
	S, CAL = {}, {}
	for p in paths:
		name = p.split("/")[-1].replace(".tar.gz", "")
		arms, cal = load(p)
		want = ["k1", "k4", "k6", "IR@4", "IR@6", "reg@4", "reg@6", "act@4", "act@6"] + [f"{g}@{b}" for b in (4, 6) for g in ROB]
		have = [a for a in want if a in arms]
		seeds = sorted(r["seed"] for r in arms["k1"])
		S[name] = {a: np.array([{r["seed"]: r["success"] for r in arms[a]}[s] for s in seeds]) for a in have}
		if cal is not None:
			CAL[name] = cal
	names = sorted(S)
	cols = [a for a in S[names[0]] if all(a in S[n] for n in names)]
	print(f"robots: {len(names)}\n")
	print(f"{'robot':28s} " + " ".join(f"{a:>7s}" for a in cols))
	for n in names:
		print(f"{n:28s} " + " ".join(f"{S[n][a].mean():7.3f}" for a in cols))
	print(f"{'mean':28s} " + " ".join(f"{np.mean([S[n][a].mean() for n in names]):7.3f}" for a in cols))

	def show(a, b):
		if a not in cols or b not in cols:
			return
		full = np.mean([(S[n][a] - S[n][b]).mean() for n in names])
		bs = [np.mean([(S[n][a] - S[n][b])[rng.integers(0, len(S[n][a]), len(S[n][a]))].mean() for n in names])
		      for _ in range(B)]
		lo, hi = np.percentile(bs, [2.5, 97.5])
		print(f"  {a + ' - ' + b:22s} {full:+.3f} [{lo:+.3f}, {hi:+.3f}]")

	for b in (4, 6):
		print(f"\nbudget {b}:")
		for a, c in [("IRs", "IR"), ("reg", "IRs"), ("RIRo", "IR"), ("RIR", "IR"), ("RIRt", "IR"), ("RIRc", "IR"),
		             ("RIR", "RIRt"), ("RIR", "RIRc"), ("RIR", f"k"), ("reg", "IR")]:
			show(f"{a}@{b}" if a != "k" else f"k{b}", f"{c}@{b}" if c != "k" else f"k{b}")
		show(f"RIR@{b}", f"k{b}")
	show("k1", "k4")

	if CAL:
		print("\noffline, calibration plans: AUROC for ranking reg (top quarter vs bottom half), steps 1..7")
		keys = ["IR0", *ROB]
		print(f"{'robot':28s} " + " ".join(f"{k:>6s}" for k in keys))
		allA = {k: [] for k in keys}
		for n in sorted(CAL):
			c = CAL[n]
			reg = c["reg"].numpy()
			n_s = reg.shape[1]
			y = reg.reshape(-1)
			hi, lo = y >= np.quantile(y, 0.75), y <= np.quantile(y, 0.5)
			m = hi | lo
			row = []
			for k in keys:
				s = c["scores"][k][:, :n_s].numpy().reshape(-1)
				a = auroc(s[m], hi[m])
				allA[k].append(a)
				row.append(a)
			print(f"{n:28s} " + " ".join(f"{a:6.3f}" for a in row))
		print(f"{'mean':28s} " + " ".join(f"{np.nanmean(allA[k]):6.3f}" for k in keys))


if __name__ == "__main__":
	main(sys.argv[1:])
