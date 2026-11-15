"""Generalisation study report (docs/preregistration_general.md): claims S1/S2 or N1/N2 from gs_* / gn_* archives.

	python -m src.tools.general_report S results/gs_*.tar.gz
	python -m src.tools.general_report N results/gn_*.tar.gz
"""
import io
import sys
import tarfile

import numpy as np
import torch

ARMS = ("k1", "k2", "k4", "k6", "random@4", "IR@2", "IR@4", "IR@6", "PE@4")
B = 5000
MIN_SUCCESS = 0.20


def load(path: str):
	arms, coll = {}, None
	with tarfile.open(path) as tf:
		for m in tf.getmembers():
			if not m.isfile():
				continue
			if "/arms/" in m.name and m.name.endswith(".pt"):
				arms[m.name.split("/")[-1][:-3]] = torch.load(io.BytesIO(tf.extractfile(m).read()), weights_only=False)["rows"]
			elif m.name.endswith("episodes.pt"):
				d = torch.load(io.BytesIO(tf.extractfile(m).read()), weights_only=False)
				coll = float(d["success"][:, -1].float().mean())
	return arms, coll


def main(part: str, paths: list):
	rng = np.random.default_rng(0)
	S, K, excluded = {}, {}, []
	for p in paths:
		name = p.split("/")[-1].replace(".tar.gz", "")
		arms, coll = load(p)
		if not all(a in arms for a in ARMS):
			print(f"incomplete {name}: {sorted(arms)}")
			continue
		if coll is None or coll < MIN_SUCCESS:
			excluded.append((name, coll))
			continue
		seeds = sorted(r["seed"] for r in arms["k1"])
		S[name] = {a: np.array([{r["seed"]: r["success"] for r in arms[a]}[s] for s in seeds]) for a in ARMS}
		K[name] = {a: sum(r["steps"] for r in arms[a]) / sum(r["plan_calls"] for r in arms[a]) for a in ARMS}
	names = sorted(S)
	print(f"part {part}: {len(names)} robots included; excluded (collection success < {MIN_SUCCESS}): {excluded}\n")
	print(f"{'robot':26s} " + " ".join(f"{a:>8s}" for a in ARMS))
	for n in names:
		print(f"{n:26s} " + " ".join(f"{S[n][a].mean():8.3f}" for a in ARMS))
	print(f"{'mean':26s} " + " ".join(f"{np.mean([S[n][a].mean() for n in names]):8.3f}" for a in ARMS))
	print(f"{'actions per plan':26s} " + " ".join(f"{np.mean([K[n][a] for n in names]):8.2f}" for a in ARMS))

	def diff(a, b):
		full = np.mean([(S[n][a] - S[n][b]).mean() for n in names])
		bs = [np.mean([(S[n][a] - S[n][b])[rng.integers(0, len(S[n][a]), len(S[n][a]))].mean() for n in names])
		      for _ in range(B)]
		return (full, *np.percentile(bs, [2.5, 97.5]))

	def show(tag, a, b, claim=True):
		m, lo, hi = diff(a, b)
		print(f"{tag:26s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  " + (("PASS" if lo > 0 else "FAIL") if claim else ""))

	print("\nclaims:")
	show(f"{part}1 IR@4 - random@4", "IR@4", "random@4")
	show(f"{part}2 IR@4 - k4", "IR@4", "k4")
	show(f"{part}2 IR@6 - k6", "IR@6", "k6")
	print("\nreported:")
	show("IR@2 - k2", "IR@2", "k2", False)
	show("PE@4 - random@4", "PE@4", "random@4", False)
	show("PE@4 - k4", "PE@4", "k4", False)
	show("IR@4 - PE@4", "IR@4", "PE@4", False)
	for b in (2, 4, 6):
		show(f"k1 - k{b}", "k1", f"k{b}", False)


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2:])
