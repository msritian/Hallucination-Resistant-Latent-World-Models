"""Wall-clock study report (docs/preregistration_wallclock.md): claims W1-W4 from the tm_*.tar.gz job results.

	python -m src.tools.wallclock_report results/tm_*.tar.gz
"""
import io
import sys
import tarfile

import numpy as np
import torch

ARMS = ("k1", "k2", "k4", "k6", "IR@2", "IR@4", "IR@6")
B = 5000


def load(path: str) -> dict:
	"""{arm: (rows, rec)} of one robot's job archive."""
	out = {}
	with tarfile.open(path) as tf:
		for m in tf.getmembers():
			if m.name.startswith("tim/arms/") and m.name.endswith(".pt"):
				d = torch.load(io.BytesIO(tf.extractfile(m).read()), weights_only=False)
				out[m.name.split("/")[-1][:-3]] = (d["rows"], d["rec"])
	return out


def per_episode(rows: list, field: str) -> dict:
	return {r["seed"]: r[field] for r in rows}


def main(paths: list):
	rng = np.random.default_rng(0)
	robots = {p.split("/")[-1].replace(".tar.gz", ""): load(p) for p in paths}
	robots = {n: d for n, d in robots.items() if all(a in d for a in ARMS)}
	names = sorted(robots)
	print(f"robots: {len(names)}\n")

	# per robot: success and seconds-per-step arrays aligned on seeds, plus gate/plan time totals
	S, T, ratio_gp, kmean = {}, {}, {}, {}
	for n in names:
		d = robots[n]
		seeds = sorted(per_episode(d["k1"][0], "success"))
		S[n] = {a: np.array([per_episode(d[a][0], "success")[s] for s in seeds]) for a in ARMS}
		T[n] = {a: (np.array([per_episode(d[a][0], "seconds")[s] for s in seeds]),
		            np.array([per_episode(d[a][0], "steps")[s] for s in seeds])) for a in ARMS}
		tp = sum(float(d[a][1]["t_plan"].sum()) for a in ("IR@2", "IR@4", "IR@6"))
		tg = sum(float(d[a][1]["t_gate"].sum()) for a in ("IR@2", "IR@4", "IR@6"))
		ratio_gp[n] = tg / tp
		kmean[n] = {a: sum(r["steps"] for r in d[a][0]) / sum(r["plan_calls"] for r in d[a][0]) for a in ARMS}

	def boot(stat):
		"""stat(idx: {robot: episode index array}) -> float; returns mean, 95% CI over a within-robot bootstrap."""
		full = {n: np.arange(len(S[n]["k1"])) for n in names}
		vals = [stat({n: rng.integers(0, len(full[n]), len(full[n])) for n in names}) for _ in range(B)]
		return stat(full), *np.percentile(vals, [2.5, 97.5])

	def sdiff(a, b):
		return lambda idx: np.mean([(S[n][a][idx[n]] - S[n][b][idx[n]]).mean() for n in names])

	def tps(n, a, i):           # seconds per step over the chosen episodes
		sec, st = T[n][a]
		return sec[i].sum() / st[i].sum()

	def tratio(a, b):           # mean over robots of time per step of a / of b
		return lambda idx: np.mean([tps(n, a, idx[n]) / tps(n, b, idx[n]) for n in names])

	# table
	print(f"{'arm':6s} {'success':>8s} {'act/plan':>8s} {'ms/step':>8s} {'speed vs k1':>11s}")
	for a in ARMS:
		print(f"{a:6s} {np.mean([S[n][a].mean() for n in names]):8.3f} {np.mean([kmean[n][a] for n in names]):8.2f} "
		      f"{1000 * np.mean([tps(n, a, slice(None)) for n in names]):8.1f} "
		      f"{np.mean([tps(n, 'k1', slice(None)) / tps(n, a, slice(None)) for n in names]):10.2f}x")
	tpl = {n: 1000 * float(torch.cat([robots[n][a][1]["t_plan"] for a in ARMS]).mean()) for n in names}
	tgt = {n: 1000 * float(torch.cat([robots[n][a][1]["t_gate"] for a in ("IR@2", "IR@4", "IR@6")]).mean()) for n in names}
	print(f"\nper plan: planning {np.mean(list(tpl.values())):.1f} ms, IR gate {np.mean(list(tgt.values())):.1f} ms\n")

	def show(tag, m, lo, hi, ok):
		print(f"{tag}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {'PASS' if ok else 'FAIL'}")

	w1 = float(np.mean(list(ratio_gp.values())))
	print(f"W1 gate time / planning time: {w1:.3f} (< 0.15)  {'PASS' if w1 < 0.15 else 'FAIL'}")
	m, lo, hi = boot(tratio("k1", "IR@2"))
	print(f"W2 speedup k1 / IR@2: {m:.3f} [{lo:.3f}, {hi:.3f}] (CI above 1)  {'PASS' if lo > 1 else 'FAIL'}")
	for b in (2, 6):
		m, lo, hi = boot(sdiff(f"IR@{b}", f"k{b}"))
		show(f"W3 IR@{b} - k{b}", m, lo, hi, lo > 0)
	for b in (2, 6):
		m, lo, hi = boot(tratio(f"IR@{b}", f"k{b}"))
		print(f"W4 time IR@{b} / k{b}: {m:.3f} [{lo:.3f}, {hi:.3f}] (mean < 1.15)  {'PASS' if m < 1.15 else 'FAIL'}")
	print("\nreported:")
	show("IR@4 - k4", *boot(sdiff("IR@4", "k4")), True)
	for b in (2, 4, 6):
		show(f"k1 - IR@{b}", *boot(sdiff("k1", f"IR@{b}")), True)
	print("\nper robot (gate/plan time, success k1 / IR@2 / k2 / IR@6 / k6):")
	for n in names:
		print(f"  {n:28s} {ratio_gp[n]:.3f}  " + " ".join(f"{S[n][a].mean():.2f}" for a in ("k1", "IR@2", "k2", "IR@6", "k6")))


if __name__ == "__main__":
	main(sys.argv[1:])
