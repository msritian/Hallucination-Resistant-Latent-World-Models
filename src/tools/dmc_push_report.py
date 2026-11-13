"""DMC disturbance study report (docs/preregistration_dmc_push.md): claims P1-P4 from the dpg_*.tar.gz results.

	python -m src.tools.dmc_push_report results/dpg_*.tar.gz
"""
import io
import json
import sys
import tarfile

import numpy as np
import torch

ARMS = ("k1", "k6", "random@6", "IR@6", "PE@6", "BND@6", "reg@6")
B = 5000


def load(path: str):
	arms, cal, bound = {}, None, None
	with tarfile.open(path) as tf:
		for m in tf.getmembers():
			if not m.isfile():
				continue
			raw = tf.extractfile(m).read()
			if "/arms/" in m.name:
				arms[m.name.split("/")[-1][:-3]] = torch.load(io.BytesIO(raw), weights_only=False)["rows"]
			elif m.name.endswith("calibration.pt"):
				cal = torch.load(io.BytesIO(raw), weights_only=False)
			elif m.name.endswith("bound.json"):
				bound = json.loads(raw)
	return arms, cal, bound


def auroc(score: np.ndarray, label: np.ndarray) -> float:
	pos, neg = score[label], score[~label]
	if len(pos) == 0 or len(neg) == 0:
		return float("nan")
	r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
	return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def main(paths):
	rng = np.random.default_rng(0)
	S, K, info = {}, {}, {}
	for p in paths:
		arms, cal, bound = load(p)
		if not all(a in arms for a in ARMS):
			print(f"skip {p}: arms {sorted(arms)}")
			continue
		t = p.split("/")[-1].replace(".tar.gz", "").replace("dpg_", "")
		seeds = sorted(r["seed"] for r in arms["k1"])
		S[t] = {a: np.array([{r["seed"]: r["success"] for r in arms[a]}[s] for s in seeds]) for a in ARMS}
		K[t] = {a: sum(r["steps"] for r in arms[a]) / sum(r["plan_calls"] for r in arms[a]) for a in ARMS}
		info[t] = (cal, bound)
	tasks = sorted(S)
	print(f"tasks: {len(tasks)}\n")
	print(f"{'task':18s} " + " ".join(f"{a:>9s}" for a in ARMS))
	for t in tasks:
		print(f"{t:18s} " + " ".join(f"{S[t][a].mean():9.3f}" for a in ARMS))
	print(f"{'mean':18s} " + " ".join(f"{np.mean([S[t][a].mean() for t in tasks]):9.3f}" for a in ARMS))
	print(f"{'actions/plan':18s} " + " ".join(f"{np.mean([K[t][a] for t in tasks]):9.2f}" for a in ARMS))

	def diff(a, b):
		full = np.mean([(S[t][a] - S[t][b]).mean() for t in tasks])
		bs = [np.mean([(S[t][a] - S[t][b])[rng.integers(0, len(S[t][a]), len(S[t][a]))].mean() for t in tasks])
		      for _ in range(B)]
		lo, hi = np.percentile(bs, [2.5, 97.5])
		return full, lo, hi

	def show(tag, a, b, claim=True):
		m, lo, hi = diff(a, b)
		v = ("PASS" if lo > 0 else "FAIL") if claim else ""
		print(f"{tag:28s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {v}")

	print("\nclaims:")
	show("P1 k1 - k6", "k1", "k6")
	show("P2 BND@6 - IR@6", "BND@6", "IR@6")
	show("P3 BND@6 - PE@6", "BND@6", "PE@6")
	show("P4 BND@6 - k6", "BND@6", "k6")
	print("\nreported:")
	for a in ("IR@6", "PE@6", "reg@6", "random@6", "BND@6"):
		show(f"{a} - k6", a, "k6", False)
		show(f"{a} - random@6", a, "random@6", False)
	print("\nbound weights and calibration AUROC (reg top quarter vs bottom half, steps with reg > 0 or not):")
	for t in tasks:
		cal, bound = info[t]
		if cal is None or bound is None:
			continue
		ir, pe, reg = (cal[k][:, :7].reshape(-1).double().numpy() for k in ("IR", "PE", "reg"))
		bnd = bound[0] * ir + bound[1] * pe
		hi, lo = reg >= np.quantile(reg, 0.75), reg <= np.quantile(reg, 0.5)
		m = hi | lo
		print(f"  {t:18s} a={bound[0]:.4g} b={bound[1]:.4g}  IR {auroc(ir[m], hi[m]):.3f}  PE {auroc(pe[m], hi[m]):.3f}"
		      f"  BND {auroc(bnd[m], hi[m]):.3f}")


if __name__ == "__main__":
	main(sys.argv[1:])
